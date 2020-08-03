import fitsio
from astropy.cosmology import FlatLambdaCDM
import numpy as np

# Small helper catalog with sources

class SourceCat:

    def __init__(self,input, position = None, limit = None):
        #number steps to the end of data, default = 1
        #limit: upper limit of theta in DEGREE to collect data
        #GLEAM covers sky 30 to -90 deg declination
        #Position = position of telescope in declination degree
        #Default position of telescope = -30 in declination coordinate so rotate 120 around y-axis
        if isinstance(input,str):
            # RA/DEC in degree, FLUX in JY
            RA_DEC_FLUX, head = fitsio.read(input, columns= ['RA','DEC','FLUX'], header= True)
            alpha = fitsio.read(input, columns = ['ALPHA'])    # Spectral index
            
            max_size = head['NAXIS2']
            
            RA_DEC_FLUX = np.array(RA_DEC_FLUX.tolist())
            alpha = np.array(alpha.tolist())
            
            RA_DEC_FLUX[:,0] *= (np.pi/180.)   #convert to rad for right ascension
            RA_DEC_FLUX[:,1] = (90. - RA_DEC_FLUX[:,1])*(np.pi/180.) #convert dec to theta
            
            #Rotate all the data, convert to cartesian for rotation
            x_old = np.sin(RA_DEC_FLUX[:,1])*np.cos(RA_DEC_FLUX[:,0])
            z_old = np.cos(RA_DEC_FLUX[:,1])
            y_old = np.sin(RA_DEC_FLUX[:,1])*np.sin(RA_DEC_FLUX[:,0])

            if position is None:
                #Rotate around y-axis:
                angle = -120.*np.pi/180.
            else:
                angle = -(90 - position) * np.pi/180.
            
            x_new = np.cos(angle)*x_old + np.sin(angle)*z_old
            z_new = -np.sin(angle)*x_old + np.cos(angle)*z_old
        
            RA_DEC_FLUX[:,1] = np.arccos(z_new)
            RA_DEC_FLUX[:,0] = np.arctan2(y_old,x_new)
                
            if limit is not None:
                cond = np.where(RA_DEC_FLUX[:,1] < (limit*np.pi/180.))
                self.theta_phi_flux = RA_DEC_FLUX[cond]
                self.theta_phi_flux[:,[0,1,2]] = self.theta_phi_flux[:,[1,0,2]]
                self.a_index = alpha[cond]
            else:                                                          
                self.theta_phi_flux = np.zeros(RA_DEC_FLUX.shape)
                self.theta_phi_flux[:,[0,1,2]] = RA_DEC_FLUX[:,[1,0,2]]    #match theta_phi_flux
                self.a_index = alpha
                
        elif input.shape[1] == 3:
            self.theta_phi_flux = input # flux in Jy

            
            
        
    
