import numpy as np
import simbox
import visicalc as vc

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

    def get_visibilities (self, sb):
        sigbeam = 0.5 * sb.lams / self.Ddish 
        print ("Beam sigma: %3.2f-%3.2f deg from z=%3.2f-%3.2f"%(sigbeam[0]/np.pi*180,
                                             sigbeam[-1]/np.pi*180., sb.zs[0],sb.zs[-1]))

        boxosig=sb.L_rad/2/sigbeam
        print ("Beam sigma at the edge: %3.2f - %3.2f "%(boxosig[0],boxosig[-1]))

        vis=[]
        for i,(lam, dpix,  beam) in enumerate(zip(sb.lams,sb.Dpix_rad,sigbeam)):
            V = vc.VisiCalc(sb.box[:,:,i],dpix,vc.SimplestGaussBeam(beam))
            vis.append(V.visibility(self.u_m/lam,self.v_m/lam))
        vis = np.array(vis)
        return vis
    
            
