import numpy as np
from .simbox import SimBox
from . import visicalc as vc

class TelSim:
    def __init__ (self,N=16, D=6, Ddish=5.5, kind='square'):
        self.N = N
        self.D = D
        self.Ddish = Ddish
        self.get_baselines(kind)


    def get_baselines(self,kind):
        """ gets unique baselines in meters (but doesn't count them)
        uniques baselines on the square can be calculate from the bottome left
        antenna, taking also reflection across z-aix. Hassle anze if you
        don't get this. """
        assert(kind=='square')
        u_m,v_m = [],[]
        for i in range(-self.N+1,self.N):
            for j in range(0,self.N):
                u_m.append(i)
                v_m.append(j)

        self.u_m = np.array(u_m) ## u in meters
        self.v_m = np.array(v_m)

    def get_visibilities (self, sb, pad = None, uv_m = None, verbose = False):
        sigbeam = 0.5 * sb.lams / self.Ddish 
        print ("Beam sigma: %3.2f-%3.2f deg from z=%3.2f-%3.2f"%(sigbeam[0]/np.pi*180,
                                             sigbeam[-1]/np.pi*180., sb.zs[0],sb.zs[-1]))

        boxosig=sb.L_rad/2/sigbeam
        print ("Beam sigma at the edge: %3.2f - %3.2f "%(boxosig[0],boxosig[-1]))

        vis=[]
        N = sb.Nx
        if uv_m is None:
            u_m = self.u_m
            v_m = self.v_m
        else:
            u_m = np.atleast_1d(uv_m[0])
            v_m = np.atleast_1d(uv_m[1])
            
        for i,(lam, dpix,  beam) in enumerate(zip(sb.lams,sb.Dpix_rad,sigbeam)):
            box = sb.box[:,:,i]
            if pad is not None:
                cbox = np.zeros ((pad*N,pad*N))
                ## we need to copy over while keeping the origin
                Nh = N//2
                cbox[:Nh,:Nh] = box[:Nh,:Nh]
                cbox[:Nh,-Nh:] = box[:Nh,-Nh:]
                cbox[-Nh:,:Nh] = box[-Nh:,:Nh]
                cbox[-Nh:,-Nh:] = box[-Nh:,-Nh:]
            else:
                cbox = box
            V = vc.VisiCalc(cbox,dpix,vc.SimplestGaussBeam(beam))
            R,il,jl = V.visibility(u_m/lam,v_m/lam, return_indices=True)
            if verbose:
                if i==0 or i==sb.Nz-1:
                    print ("FFT Indices used at slice %i are (%i,%i)."%(i,il,jl))
            vis.append(R)
        vis = np.array(vis)
        return vis
    
            
