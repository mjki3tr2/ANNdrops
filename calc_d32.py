def calc_d32(C, x_vals):
    """
    Computes d32 (Sauter mean diameter) using trapezoidal integration.
    
    C - cumulative distribution from mastersizer
    x_vals - diameters
    
    For each interval i = 1,...,N-1:
      - Bin width: Δd* = x_vals[i] - x_vals[i-1]
      - Effective droplet size for the interval: d_eff = sqrt(x_vals[i] * x_vals[i-1])
      - Convert cumulative volume frequency to number frequency using:
            f_n = [C[i] / ((pi/6)*(d_eff)**3) + C[i-1] / ((pi/6)*(d_eff_prev)**3)] / 2 * Δd*
        (For i == 1, use d_eff for both boundaries)
      - Compute contributions for the third and second moments:
            term_d3 = (f_n_current*(d_eff)**3 + f_n_prev*(d_eff_prev)**3)/2 * Δd*
            term_d2 = (f_n_current*(d_eff)**2 + f_n_prev*(d_eff_prev)**2)/2 * Δd*
    
    Returns:
      d32 = (sum of term_d3) / (sum of term_d2)
    """
    import numpy as np

    N = len(x_vals)
    factor = np.pi / 6.0
    f_n = np.zeros(len(x_vals))
    f_n[0] = ((C[0] / (factor * ((x_vals[0]/2)**3))))/ 2 * x_vals[0]
    for i in range(1, N):
        bin_width = x_vals[i] - x_vals[i-1]
        d_current = np.sqrt(x_vals[i] * x_vals[i-1])
        if i == 1:
            d_prev = x_vals[0]/2
        else:
            d_prev = np.sqrt(x_vals[i-1] * x_vals[i-2])
        
        f_n[i] = ((C[i] / (factor * (d_current**3))) +  (C[i-1] / (factor * (d_prev**3)))) / 2 * bin_width
    
    int_fnd3 = 0.0
    int_fnd2 = 0.0  
    for i in range(1, N):
        bin_width = x_vals[i] - x_vals[i-1]
        d_current = np.sqrt(x_vals[i] * x_vals[i-1])
        if i == 1:
            d_prev = x_vals[0]/2
        else:
            d_prev = np.sqrt(x_vals[i-1] * x_vals[i-2])
         
        term_d3 = (f_n[i] * (d_current**3) + f_n[i-1] * (d_prev**3)) / 2.0 * bin_width
        term_d2 = (f_n[i] * (d_current**2) + f_n[i-1] * (d_prev**2)) / 2.0 * bin_width
        
        int_fnd3 += term_d3
        int_fnd2 += term_d2

    if int_fnd2 == 0:
        return float('nan')  # Still return a scalar float

    return float(int_fnd3 / int_fnd2)  # Explicit float conversion