import fitsio
from astropy.cosmology import FlatLambdaCDM
import numpy as np

# Small helper catalog with sources

class SourceCat:

    def __init__(self,input, position = None, set_range = None, condition = None):
        #number steps to the end of data, default = 1
        #Condition: include data for certain theta around where telescope is pointing, in tuple in rad 
        #GLEAM covers sky 30 to -90 deg declination
        #Default position of telescope = -30 in declination coordinate 
        if isinstance(input,str):
            # RA/DEC in degree, FLUX in JY
            RA_DEC_FLUX, head = fitsio.read(input, columns= ['RA','DEC','FLUX'], header= True)
            max_size = head['NAXIS2']
            RA_DEC_FLUX = np.array(RA_DEC_FLUX.tolist())
            RA_DEC_FLUX[:,0] *= (np.pi/180.)   #convert to rad
            
            if position is None:
                RA_DEC_FLUX[:,1] = (-30. - RA_DEC_FLUX[:,1])*(np.pi/180.) #convert dec to theta in rad
            else:
                RA_DEC_FLUX[:,1] = (position - RA_DEC_FLUX[:,1])*(np.pi/180.)
            
            if set_range is not None:
                if np.amax(set_range) >= max_size:
                    print('endpoint must be smaller than ', str(max_size))
                    raise UnboundLocalError
                else:
                    self.theta_phi_flux = np.zeros((len(set_range),3))
                    self.theta_phi_flux = RA_DEC_FLUX[set_range,:] 
                    self.theta_phi_flux[:,[0,1,2]] = self.theta_phi_flux[:,[1,0,2]]
                
            elif condition is not None:
                cond = np.where((RA_DEC_FLUX[:,1] > condition[0]) & (RA_DEC_FLUX[:,1] < condition[1]))
                self.theta_phi_flux = RA_DEC_FLUX[cond]
                self.theta_phi_flux[:,[0,1,2]] = RA_DEC_FLUX[cond][:,[1,0,2]]
           
            else:                                                          
                self.theta_phi_flux = np.zeros(RA_DEC_FLUX.shape)
                self.theta_phi_flux[:,[0,1,2]] = RA_DEC_FLUX[:,[1,0,2]]    #match theta_phi_flux
                
        elif input.shape[1] == 3:
            self.theta_phi_flux = input # flux in mJy 

            
            
        
    
