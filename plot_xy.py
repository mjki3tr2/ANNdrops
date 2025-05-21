def plot_xy(
        y_true,
        y_pred,
        title_str="Predicted vs Actual",
        xlabel_str="Actual",
        ylabel_str="Predicted",
        log=True,
        save_path=None
):
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    import numpy as np
    import os
    
    # For each point, compute error% = |(true - pred)/true| * 100 and then average.
    mae = np.abs((y_true - y_pred)/y_true)*100
    mae_mean = np.mean(mae)
    r2 = r2_score(y_true,y_pred)
    
    # Plots a simple y_pred vs. y_true scatter plus a y=x line.
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='k')
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal: y = x')
    ax.plot([min_val, max_val], [1.2*min_val, 1.2*max_val], 'g--', label ='20% Error')
    ax.plot([min_val, max_val], [0.8*min_val, 0.8*max_val], 'g--')
    
    error_threshold = 0.2
    for i in range(len(y_true)):
        error = abs(y_true[i] - y_pred[i])/y_true[i]
        if error > error_threshold:
            ax.annotate(str(int(i+1)), (y_true[i], y_pred[i]), fontsize=8, color='red')
    
    ax.text(0.05, 0.95, f'Avg % Error: {mae_mean:.2f}%\nR² Score: {r2:.4f}', transform=ax.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_title(title_str)
    ax.set_xlabel(xlabel_str)
    ax.set_ylabel(ylabel_str)
    ax.legend()
    ax.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)