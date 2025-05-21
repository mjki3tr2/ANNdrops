def calc_distribution(x_values,Mo_values,s_values,f_values):
    
    """
    x_values - diameters
    Mo_values - 3 modes
    s_values - 3 standard deviations
    f_values - 3 phase fractions
    
    """
    
    import numpy as np
    from scipy.stats import lognorm
    
    x_values = np.array(x_values)
    y_values = np.concatenate((np.array([[0.0001]]), x_values[:-1]))
    
    # Extract specific fit parameters correctly
    Mo1, Mo2, Mo3 = np.log(Mo_values[0]), np.log(Mo_values[1]), np.log(Mo_values[2])
    s1, s2, s3 = s_values[0], s_values[1], s_values[2]
    f1, f2, f3 = f_values[0], f_values[1], f_values[2]

    # Handle s values and calculate cumulative distributions
    if s3 < 0.001 or f3 < 0.001:  # No distribution for small droplets
        y_small_cum = np.zeros_like(x_values)
    else:
        y_small_cum = (lognorm.cdf(x_values, s3, scale=np.exp(Mo3 + (s3**2))) - lognorm.cdf(y_values, s3, scale=np.exp(Mo3 + (s3**2)))) * f3 * 100

    if s2 < 0.001 or f2 < 0.001:  # No distribution for medium droplets
        y_medium_cum = np.zeros_like(x_values)
    else:
        y_medium_cum = (lognorm.cdf(x_values, s2, scale=np.exp(Mo2 + (s2**2))) - lognorm.cdf(y_values, s2, scale=np.exp(Mo2 + (s2**2)))) * f2 * 100

    if s1 < 0.001 or f1 < 0.001:  # No distribution for large droplets
        y_large_cum = np.zeros_like(x_values)
    else:
        y_large_cum = (lognorm.cdf(x_values, s1, scale=np.exp(Mo1 + (s1**2))) - lognorm.cdf(y_values, s1, scale=np.exp(Mo1 + (s1**2)))) * f1 * 100
   
    # Combine cumulative distributions
    y_total_cum = y_small_cum + y_medium_cum + y_large_cum
    
    return y_small_cum, y_medium_cum, y_large_cum, y_total_cum