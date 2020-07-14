import numpy as np
from .simbox import SimBox
from . import visicalc as vc


class TelSim:
    """Object describing a telescope. It can calculate visibilities and
    baseline distributions given appropriate objects.

    Parameters
    ----------
    N : int, optional
        Linear Array size. If kind=='square', the total number of
        elements is N^2. Default is N=16

    D : float, optional
        Element separation in meters. Default is 6m

    Ddish : float, optional
        Dish size in meters, used to calculate beams, effective
        aperture size, etc, Default is 5.5m

    kind : str, optional
        Placeholder for the type or array, currently only
        'square' is supported as fully filled square array
    """

    def __init__(self, N=16, D=6, Ddish=5.5, kind='square'):
        self.N = N
        self.D = D
        self.Ddish = Ddish
        self.kind = kind
        self._get_baselines()

    def _get_baselines(self):
        """ Gets unique baselines in meters (but doesn't count them).

        Notes
        -----

        Unique baselines on the square can be calculate from the bottom left
        antenna, taking also reflection across z-aix. Hassle anze if you
        don't get this.
        """

        assert(self.kind == 'square')
        u_m, v_m = [], []
        for i in range(-self.N + 1, self.N):
            for j in range(0, self.N):
                u_m.append(i)
                v_m.append(j)

        self.u_m = np.array(u_m)  # u in meters
        self.v_m = np.array(v_m)

    def get_visibilities(self, sb, pad=2, uv_m=None, vopts={}, verbose=1, deg_x_y =None):
        """ Gets visibilities given a simulation box

        Parameters
        ----------
        sb : SimBox
            Simulation box from which to calculate visibilities
        pad : float, optional
            How much to pad the 2D planes before FFTing. 2 is desirable to
            avoid wraparound issue (note that we broke translation symmetry by beam)
        uv_m: tuple of numpy arrays, optional
           Tuple (u_m,v_m) describing baselines of interest. If not specified
           we use all uniqute baselines as given by _get_baselines()
        vopts: dic, optional
           Options to pass to the VisiCalc visibility() function
        verbose: int, optional
            deg_x_y, tuple,(x,y) in degree.
           Babble, babble...
        """
        sigbeam = 0.5 * sb.lams / self.Ddish
        if verbose:
            print("Beam sigma: %3.2f-%3.2f deg from z=%3.2f-%3.2f" %
                  (sigbeam[0] / np.pi * 180, sigbeam[-1] / np.pi * 180., sb.zs[0], sb.zs[-1]))

        boxosig = sb.L_rad / 2 / sigbeam
        if verbose:
            print("Beam sigma at the edge: %3.2f - %3.2f " %
                  (boxosig[0], boxosig[-1]))

        vis = []
        N = sb.Nx
        if uv_m is None:
            u_m = self.u_m
            v_m = self.v_m
        else:
            u_m = np.atleast_1d(uv_m[0])
            v_m = np.atleast_1d(uv_m[1])

        if 'interpolation' not in vopts:
            vopts['interpolation']='lasz'
        
        if deg_x_y is not None:
            offset_x_rad = deg_x_y[0] / 180 * np.pi 
            offset_y_rad = deg_x_y[1] / 180 * np.pi 
            #offset_xi = np.rint(offset_x_rad/sb.Dpix_rad).astype(int) 
            #offset_yi = np.rint(offset_y_rad/sb.Dpix_rad).astype(int)
            #temp = np.zeros((sb.Nx,sb.Nx,sb.Nx))
            #temp[offset_xi,offset_yi,np.arange(sb.Nz)] = sb.box[offset_xi,offset_yi,np.arange(sb.Nz)]
            #sb.box = temp
            
            #sb.box *= 0.0
            #sb.box[offset_xi,offset_yi,np.arange(sb.Nz)]=1.
        for i, (lam, dpix, beam) in enumerate(
                zip(sb.lams, sb.Dpix_rad, sigbeam)):
            box = sb.box[:, :, i]    
            if pad is not None:
                cbox = np.zeros((pad * N, pad * N))
                # we need to copy over while keeping the origin
                Nh = N // 2
                cbox[:Nh, :Nh] = box[:Nh, :Nh]
                cbox[:Nh, -Nh:] = box[:Nh, -Nh:]
                cbox[-Nh:, :Nh] = box[-Nh:, :Nh]
                cbox[-Nh:, -Nh:] = box[-Nh:, -Nh:]
            else:
                cbox = box
            
            if deg_x_y is not None:
                V = vc.VisiCalc(cbox, dpix, vc.SimplestGaussBeam(beam), rad_x_y = (offset_x_rad, offset_y_rad))
            else:
                V = vc.VisiCalc(cbox, dpix, vc.SimplestGaussBeam(beam))
            
            R = V.visibility(u_m / lam, v_m / lam,
                interpolation = vopts['interpolation'], opts=vopts)
            
            if verbose > 1:
                if i == 0 or i == sb.Nz - 1:
                    print(
                        "FFT Indices used at slice %i are (%i,%i)." %
                        (i, il, jl))
            vis.append(R)
        vis = np.array(vis)
        return vis
