import fitsio
from astropy.cosmology import FlatLambdaCDM
import numpy as np

# Small helper catalog with sources

class SourceCat:

    def __init__(self,input):
        if isinstance(input,str):
            # RA/DEC in degree, FLUX in JY
            RA_DEC_FLUX = fitsio.read(input, columns= ['RA','DEC','FLUX'])
            RA_DEC_FLUX = np.array(RA_DEC_FLUX.tolist())            # 
            RA_DEC_FLUX[:,0] *= (np.pi/180)   #convert to rad
            RA_DEC_FLUX[:,1] = (np.pi/2) - RA_DEC_FLUX[:,1]*(np.pi/180) #convert dec to theta in rad
            self.theta_phi_flux = np.zeros(RA_DEC_FLUX.shape)
           # self.theta_phi_flux = np.zeros((2,3))
            self.theta_phi_flux[:,[0,1,2]] = RA_DEC_FLUX[:,[1,0,2]]    #match theta_phi_flux
            
        elif input.shape[1] == 3:
            self.theta_phi_flux = input # flux in mJy 

            
            
        
    
