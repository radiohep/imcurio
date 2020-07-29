import numpy as np
from numpy.fft import rfft2, irfft2, fftfreq, rfftfreq
import math
import scipy.constants

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

    def __call__(self, x, y):
        return np.exp(-(x**2 + y**2) / (2 * self.sigma**2))


class VisiCalc:

    def __init__(self, realmap, source_cat, dx, beamfunc, lam):
        """ Realmap is in K (assuming you want to add sources later) """


        Nx, Ny = realmap.shape
        assert(Nx == Ny)
        self.lam = lam
        self.N = Nx
        self.dx = dx
        self.beamfunc = beamfunc
        beam = beamfunc.render(self.N, dx)
        self.beam = beam 
        self.beamA = beam.sum() * dx**2  # Beam area in steradian
        self.fx = fftfreq(self.N, dx)  # frequencies in the x direction
        self.fy = rfftfreq(self.N, dx)  # frequencies in the y direction
        self.df = self.fy[1] - self.fy[0]
        self.fmax = self.fy.max()
        self.fftmap = rfft2(realmap * self.beam**2)   #changed to beam squared.
        self.source_cat = source_cat
            
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
            la = 5                   # window length
            if 'a' in opts:
                la = opts['a']
                
            #Resize the u,v by df:
            u_s = u/self.df
            v_s = v/self.df

            #convert fx and fy into indice space:
            #fx_s = (self.fx/self.df).astype(int)  Unneeded.
            #fy_s = (self.fy/self.df).astype(int)               

            res = np.zeros(len(u),np.complex)  

            low_bound_u = (np.floor(u_s) - la + 1).astype(int)
            high_bound_u = (np.floor(u_s) + la + 1).astype(int)       #+1 on high_bound since loop excludes end.

            low_bound_v = (np.floor(v_s) - la + 1).astype(int)
            high_bound_v = (np.floor(v_s) + la + 1).astype(int)

            """
            def L_Kern(x,a):
                if x == 0.0:
                    L = 1.0
                elif x != 0.0 and x < a and x >= -a:
                    L = (a*np.sin(np.pi*x)*np.sin(np.pi*x /a))/(np.pi**2*x**2)
                else:
                    L = 0.0
                return L   
            
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
                        """
         
         # Attempts to vectorize the for loop above:
            
            def L_Kern(x,a):
                cond_1 = np.where(x == 0.0)
                cond_2 = np.where((x !=0.0) & (x < a) & (x >= -a))
                cond_3 = np.where((x >= a) | (x < -a))
                
                L = np.zeros((x.shape))
                L[cond_1] = 1.0
                L[cond_2] = (a*np.sin(np.pi*x[cond_2])*np.sin(np.pi*x[cond_2] /a))/(np.pi**2*x[cond_2]**2)
                L[cond_3] = 0.0
                
                return L
            
            
            N = 2*la
            fx_s = np.arange(low_bound_u[0],high_bound_u[0])
            fy_s = np.arange(low_bound_v[0],high_bound_v[0])
           
            for kk in range(1,len(u_s)):
                fx_s = np.vstack((fx_s, np.arange(low_bound_u[kk],high_bound_u[kk])))
                fy_s = np.vstack((fy_s, np.arange(low_bound_v[kk],high_bound_v[kk])))
            
            if len(u_s) != 1 :
                fx_s = np.repeat(fx_s, N, axis =1)
                fy_s = np.tile(fy_s, (1, N))
            
            else:
                fx_s = np.repeat(fx_s, N, axis =0)
                fy_s = np.tile(fy_s, N)                     
                                     
            ii = np.rint(np.fmod(fx_s, self.N)).astype(int)    
            jj = np.rint(np.fmod(fy_s, self.N)).astype(int)    
            
            cond = np.where(jj < 0)
            ii[cond] *= -1
            jj[cond] *= -1
      
            u_s = np.tile(u_s, (N**2,1)).T    #2d
            v_s = np.tile(v_s, (N**2,1)).T
                         
            L_x = L_Kern(u_s - fx_s , la)
            L_y = L_Kern(v_s - fy_s , la)
            
          
            data = self.fftmap[ii,jj]
            data[cond] = np.conj(data[cond])      
            res = np.sum(L_x * L_y * data , axis = 1) 

        else:
            print ("Bad interpolation")
            raise NotImplemented 

        res[conjugate] = np.conj(res[conjugate])
            
        if self.source_cat is not None:
            ## implement source catalog
            # Need 4~5 hours to load all the data, with 0 background flux.
            if self.source_cat.theta_phi_flux.shape[0] == 1:
                for theta,phi,flux in self.source_cat.theta_phi_flux:
                    fluxT = ((self.lam**2)*flux*10**(-26))/(2.*scipy.constants.k*self.beamA)
                    x = np.sin(theta)*np.cos(phi)
                    y = np.sin(theta)*np.sin(phi)
                ## full path difference is should be u*x + v*y + w*z, but w =0
                #z = np.cos(theta)
                
                    beam_sup = self.beamfunc(x,y) 
                ## these are now in radian
                ## above it says that u,v and in L/lambda,so we have
                    res += fluxT * np.exp(-1j*2.0*np.pi*(u*x+v*y)) * beam_sup**2
            else:
                theta_phi_flux = self.source_cat.theta_phi_flux #theta_phi_flux in rad and JY
                shape1 = (len(u),1)
                shape2 = (theta_phi_flux.shape[0],1)
                
                #change to K from JY
                fluxT = np.tile(((self.lam**2)*theta_phi_flux[:,2]*10**(-26))/(2.*scipy.constants.k*self.beamA), shape1) 
                x = np.tile(np.sin(theta_phi_flux[:,0])*np.cos(theta_phi_flux[:,1]), shape1)
                y = np.tile(np.sin(theta_phi_flux[:,0])*np.sin(theta_phi_flux[:,1]), shape1)
                beam_sup = self.beamfunc(x,y)
                
                u_2 = np.tile(u, shape2).T
                v_2 = np.tile(v, shape2).T
                
                res += np.sum(fluxT * np.exp(-1j*2.0*np.pi*(u_2*x+v_2*y)) * beam_sup**2, axis = 1)
                
        if 'return_indices' in opts:
            if opts['return_indices']:
                return res, il, jl
        return res
