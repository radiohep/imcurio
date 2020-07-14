import numpy as np
from numpy.fft import rfft2, irfft2, fftfreq, rfftfreq
import math

def rfft2_real_coords(N, dx):
    """
    Returns coordinates in real image

    N, int : FFT size, let it be divisible by 2
    dx, float : distance between adjacent cells

    Returns:
    v : array of floats along one coordinates
    x : 2D array with x values
    y : 2D array with y values

 """
    v = np.hstack((np.arange(0, N // 2), np.arange(-N // 2, 0))) * dx
    x = np.outer(v, np.ones(N))
    y = x.T
    return v, x, y


class SimplestGaussBeam:
    def __init__(self, sigma):
        self.sigma = sigma

    def render(self, N, dx):
        _, x, y = rfft2_real_coords(N, dx)
        if x.max() / self.sigma < 3.:
            print("Beam too big.")
            print("Beam sigma  = ", self.sigma)
            print("Box spans = ", x.min(), x.max())
            print(
                "Box spans in sigma = ",
                x.min() / self.sigma,
                x.max() / self.sigma)
            stop()

        img = np.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        return img


class VisiCalc:

    def __init__(self, realmap, dx, beam, rad_x_y = None):
        """ Realmap is in K (assuming you want to add sources later) """

        Nx, Ny = realmap.shape
        assert(Nx == Ny)
        self.N = Nx
        self.dx = dx
        beam = beam.render(self.N, dx)
        self.beam = beam / beam.sum()  # normalize
        self.beamA = beam.sum() * dx**2  # Beam area in steradian
        self.fx = fftfreq(self.N, dx)  # frequencies in the x direction
        self.fy = rfftfreq(self.N, dx)  # frequencies in the y direction
        self.df = self.fy[1] - self.fy[0]
        self.fmax = self.fy.max()
        
        if rad_x_y is None:
            self.fftmap = rfft2(realmap * self.beam**2)   #changed to beam squared.
        else:
            self.fftmap = np.zeros((self.N,self.N), np.complex)  
            fy = np.tile(self.fx, (self.N,1))
            fx = np.tile(np.reshape(self.fx,(self.N,1)),(1, self.N))
            l = np.sin(rad_x_y[0])
            m = np.sin(rad_x_y[1])
            l_i = np.rint(l/self.dx).astype(int)         #I'm thinking this would generate the index for corresponding offset angle
            m_i = np.rint(m/self.dx).astype(int)
        
            # if set to 1 like notebook, produces exactly like the notebook file
            realmap[l_i,m_i] = 1   
            # by using l_i,m_i on beam, it gives same value as notebook.
            self.fftmap = realmap[l_i,m_i] * (self.beam[l_i,m_i]**2) * np.exp(-2.*np.pi*1j*(l*fx + m*fy)) 
           
        # self.fftmap = np.delete(self.fftmap,np.s_[self.fy.size:],1)
            
    def visibility(self, u,v, interpolation = 'lasz', opts={}):
        """ u, v are in L/lambda and can be arrays
           returns complex viibility in the K units
        """

        # Let's deal with conjugation first
        u = np.copy(np.atleast_1d(u))
        v = np.copy(np.atleast_1d(v))
        if (u.max() > self.fmax - self.df *
                2) or (v.max() > self.fmax - self.df * 2):
            print("Requested UV too high.")
            print("umax vmax = %f %f " % (u.max(), v.max()))
            print('fmax = ', self.fmax)
            stop()
        conjugate = (v < 0)
        w = np.where(conjugate)
        v[w] *= -1
        u[w] *= -1

        # we need to have sufficient points to do beyond linear interpolation at lower end
        #assert (np.all(abs(v)>2*self.df))
        #assert (np.all(abs(u)>2*self.df))
        # il,ih,jl,jh are (lists of) indices boxing the frequency

        il = (u / self.df).astype(int)
        jl = (v / self.df).astype(int)
        il[u < 0] += self.N - 1
        if (interpolation == 'lin'):
            jh = jl + 1
            ih = (il + 1) % self.N
            # we will now do a two-d linear interpolation.
            # first weights
            uw = (u - self.fx[il]) / self.df
            vw = (v - self.fy[jl]) / self.df
            #print ('Here:',il,jl)
            #print (uw)
            assert(
                np.all(
                    uw >= 0) & np.all(
                    uw <= 1) & np.all(
                    vw >= 0) & np.all(
                    vw <= 1))

            res = ((1 - uw) * (1 - vw) * self.fftmap[il, jl] +
                   uw * (1 - vw) * self.fftmap[ih, jl] +
                   (1 - uw) * vw * self.fftmap[il, jh] +
                   uw * vw * self.fftmap[ih, jh])
        elif interpolation == 'lasz':
            la = 4                   # window length
            if 'a' in opts:
                la = opts['a']
                
            # let's do Lanzos 2D interpolation, we need two points at each end
            
            #Resize the u,v by df:
            u_s = u/self.df
            v_s = v/self.df
            
            #convert fx and fy into indice space:
            #fx_s = (self.fx/self.df).astype(int)  Unneeded.
            #fy_s = (self.fy/self.df).astype(int)
            
           
            
            def L_Kern(x,a):
                if x == 0.0:
                    L = 1.0
                elif x != 0.0 and x < a and x >= -a:
                    L = (a*np.sin(np.pi*x)*np.sin(np.pi*x /a))/(np.pi**2*x**2)
                else:
                    L = 0.0
                 
                return L
            
           
            
            res = np.zeros(len(u),np.complex)  
            
            low_bound_u = (np.floor(u_s) - la + 1).astype(int)
            high_bound_u = (np.floor(u_s) + la + 1).astype(int)       #+1 on high_bound since loop excludes end.
                                  
            low_bound_v = (np.floor(v_s) - la + 1).astype(int)
            high_bound_v = (np.floor(v_s) + la + 1).astype(int)
            
            ### AS: no, no, don't write loops likes this.
            ### Also, you need to make sure boundary conditions are respected!!
            
           
            
           #I can't get away using another for loop here, are there ways to access each value inside the array without using another for loop here?
        
            for kk in np.arange(len(u_s)):  
                for fx_s in np.arange(low_bound_u[kk], high_bound_u[kk]):
                    for fy_s in np.arange(low_bound_v[kk], high_bound_v[kk]):
                        L_x = L_Kern(u_s[kk] - float(fx_s), la)
                        L_y = L_Kern(v_s[kk] - float(fy_s), la)
                        ii = int(math.fmod(fx_s,self.N))           #indices for data
                        jj = int(math.fmod(fy_s,self.N))
                        
                        if fy_s < 0:
                            data = np.conj(self.fftmap[-1*ii][-1*jj])
                        else:
                            data = self.fftmap[ii][jj]
                        
                        res[kk]  += L_x * L_y * data
                        
        else:
            print ("Bad interpolation")
            raise NotImplemented 
                    
        res[conjugate] = np.conj(res[conjugate])
        if 'return_indices' in opts:
            if opts['return_indices']:
                return res, il, jl
        return res
