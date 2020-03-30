import fitsio
import pyccl as ccl
import numpy as np
from scipy.interpolate import interp1d

class SimBox:
    def __init__ (self, fname, conversions_from = None):
        box, head = fitsio.read(fname, header = True)
        if 'SKY' in head['TTYPE1']:
            kind = 'sky'
        elif 'HI' in head['TTYPE1']:
            kind = 'hi'
        else:
            print ("Bad type")
            stop()
        self.z = head['REDSHIFT']
        self.head=head
        self.Dpix = head['LBOX']/head['NAXIS2']
        self.Nz=head['NAXIS2']
        self.box = np.array([box[i][0] for i in range(len(box))])
        self.Nx = self.box.shape[0]
        if kind == 'hi':
            self.Tb = head['TBAR']
            self.box *= self.Tb
        if 'OMEGAM' in head:
            Omega_m = head['OMEGAM']
            Omega_b = head['OMEGAB']
            hubble = head['HUBBLE']/100
            sigma8 = head['SIGMA8']
            ns = head['ENN_S']
            #Make a neutrinoless cosmology
            self.C = ccl.Cosmology(Omega_b=Omega_b,Omega_c = Omega_m-Omega_b,
                                   h=hubble, sigma8=sigma8, n_s=ns)
            
            self.calculate_conversions()
        if conversions_from is not None:
            o = conversions_from
            self.zs = o.zs
            self.freq = o.freq
            self.lams = o.lams
            self.Dpix_rad = o.Dpix_rad
            self.L_rad = o.L_rad


    def calculate_conversions(self):
            self.rMpc = (ccl.comoving_radial_distance(self.C,1/(1+self.z))+
                         self.Dpix*(np.arange(self.Nz)-self.Nz//2))
            ## Let's calculate everything exactly
            zar = np.linspace(self.z-0.5,self.z+0.5)
            dar = ccl.comoving_radial_distance(self.C,1/(1+zar))
            self.zs = interp1d(dar,zar)(self.rMpc)
            self.freq = 1420.405/(1+self.zs) ## in MHz
            self.lams = 299.8/self.freq
            self.Dpix_rad = self.Dpix/self.rMpc
            self.L_rad = (self.Dpix_rad*self.Nx)
            print ("Box size: %3.2f-%3.2f deg from z=%3.2f-%3.2f"%(
                self.L_rad[0]/np.pi*180.0,self.L_rad[-1]/np.pi*180.0
                ,self.zs[0],self.zs[-1]))

    
            
            
            
