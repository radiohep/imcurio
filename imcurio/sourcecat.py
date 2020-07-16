import fitsio
from astropy.cosmology import FlatLambdaCDM
import numpy as np

# Small helper catalog with sources

class SourceCat:

    def __init__(self,input):
        if isinstance(input,str):
            # load the data
            print ("can't yet load the data")
            stop()
        elif input.shape[1] == 3:
            self.theta_phi_flux = input # flux in mJy

            
            
        
    
