import numpy as np
from numpy.fft import rfft2, irfft2, fftfreq, rfftfreq



def rfft2_real_coords(N,dx):
    """ 
    Returns coordinates in real image 

    N, int : FFT size, let it be divisible by 2
    dx, float : distance between adjacent cells

    Returns:
    v : array of floats along one coordinates
    x : 2D array with x values
    y : 2D array with y values
    
 """
    v=np.hstack((np.arange(0,N//2),np.arange(-N//2,0)))*dx
    x=np.outer(v,np.ones(N))
    y=x.T
    return v,x,y

class SimplestGaussBeam:
    def __init__ (self, sigma):
        self.sigma = sigma

    def render (self,N,dx):
        _,x,y = rfft2_real_coords(N,dx)
        if x.max()/self.sigma<3.:
            print ("Beam too big.")
            print ("Beam sigma  = ", self.sigma)
            print ("Box spans = ", x.min(),x.max())
            print ("Box spans in sigma = ", x.min()/self.sigma,x.max()/self.sigma)
            stop()

        img = np.exp(-(x**2+y**2)/(2*self.sigma**2))                   
        return img
    
class VisiCalc:

    def __init__ (self, realmap, dx, beam):
        """ Realmap is in K (assuming you want to add sources later) """
        
        Nx,Ny = realmap.shape
        assert(Nx==Ny)
        self.N = Nx
        self.dx = dx
        beam = beam.render(self.N,dx)
        self.beam = beam / beam.sum() ## normalize
        self.beamA = beam.sum()*dx**2 ## Beam area in steradian
        self.fftmap = rfft2(realmap*self.beam)
        self.fx = fftfreq(self.N,dx) ## frequencies in the x direction
        self.fy = rfftfreq(self.N,dx) ## frequencies in the y direction
        self.df = self.fy[1]-self.fy[0]
        self.fmax = self.fy.max()
        
    def visibility (self,u,v, return_indices = False, linear_interpolation=False):
        """ u, v are in L/lambda and can be arrays
           returns complex viibility in the K units
        """
        
        ## Let's deal with conjugation first
        u = np.copy(np.atleast_1d(u))
        v = np.copy(np.atleast_1d(v))
        if (u.max()>self.fmax-self.df*2) or (v.max()>self.fmax-self.df*2):
            print ("Requested UV too high.")
            print ("umax vmax = %f %f "%(u.max(), v.max()))
            print ('fmax = ',self.fmax)
            stop()
        conjugate = (v<0) ##
        w=np.where(conjugate)
        v[w] *= -1
        u[w] *= -1

        ## we need to have sufficient points to do beyond linear interpolation at lower end
        #assert (np.all(abs(v)>2*self.df))
        #assert (np.all(abs(u)>2*self.df))
        ## il,ih,jl,jh are (lists of) indices boxing the frequency
        
        il = (u/self.df).astype(int)
        jl = (v/self.df).astype(int)
        il[u<0] +=self.N-1
        if (linear_interpolation):
            jh = jl+1
            ih = (il + 1) % self.N
            ## we will now do a two-d linear interpolation.
            ## first weights
            uw = (u-self.fx[il])/self.df
            vw = (v-self.fy[jl])/self.df
            #print ('Here:',il,jl)
            #print (uw)
            assert(np.all(uw>=0)&np.all(uw<=1)&np.all(vw>=0)&np.all(vw<=1))
            
            res = ((1-uw)*(1-vw)*self.fftmap[il,jl]+
               uw*(1-vw)*self.fftmap[ih,jl]+
               (1-uw)*vw*self.fftmap[il,jh]+
               uw*vw*self.fftmap[ih,jh])
        else:
            ## let's do Lanzos 2D interpolation, we need two points at each end
            res = np.zeros(len(u),np.complex)
            sumk = np.zeros(len(u),np.complex)
            uw = (u-self.fx[il])/self.df
            vw = (v-self.fy[jl])/self.df

            for ii in np.arange (-1, +3):
                xi = ii - uw 
                kernx = 2*np.sin(np.pi*xi)*np.sin(np.pi*xi/2)/(np.pi**2*xi**2+1e-200)
                kernx[xi==0]=1.0
                xndx = (il + ii) % self.N
                for jj in np.arange (-1, +3):
                    xj = jj - vw
                    kerny = 2*np.sin(np.pi*xj)*np.sin(np.pi*xj/2)/(np.pi**2*xj**2+1e-200)
                    kerny[xj==0]=1.0
                    ## now we need to pick out where we transit the complex plain
                    jndx = (jl + jj) 
                    data = self.fftmap[xndx, abs(jndx)]
                    ### If would naivetly think I need this, huhm.
                    #w = np.where(jndx<0)
                    #data [w] = np.conj(data[w])
                    res += kernx * kerny * data
                    sumk += kernx * kerny
            res /= sumk
        res[conjugate] = np.conj(res[conjugate])
        if return_indices:
            return res, il, jl
        return res

