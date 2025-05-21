def prep_data(properties,fits):
    import numpy as np
    
    properties_fix = np.full_like(properties, 0)
    # Log-transform viscosities and modes
    properties_fix[:,0]=properties[:,0] #ID
    properties_fix[:,1]=properties[:,1] #N
    properties_fix[:,2]=np.log(properties[:,2]) #mu_d
    properties_fix[:,3]=properties[:,3] #rho_d
    properties_fix[:,4]=np.log(properties[:,4]) #mu_c
    properties_fix[:,5]=properties[:,5] #rho_c
    properties_fix[:,6]=properties[:,6] #sigma
    properties_fix[:,7]=properties[:,7] #phi
    
    fits_fix = np.full_like(fits, 0)
    fits_fix[:,0] = fits[:,0] #ID
    fits_fix[:,1] = np.log(fits[:,1]) #Mo1
    fits_fix[:,2] = fits[:,2] #s1
    fits_fix[:,3] = fits[:,3] #f1
    fits_fix[:,4] = np.log(fits[:,4]) #Mo2
    fits_fix[:,5] = fits[:,5] #s2
    fits_fix[:,6] = fits[:,6] #f2
    fits_fix[:,7] = np.log(fits[:,7]) #Mo3
    fits_fix[:,8] = fits[:,8] #s3
    fits_fix[:,9] = fits[:,9] #f3
    
    indices = fits_fix[:,0] # copy all ID values
    indices = indices.reshape(-1, 1)
    
    return properties_fix, fits_fix, indices