def plot_xy(
        y_true,
        y_pred,
        title_str="Predicted vs Actual",
        xlabel_str="Actual",
        ylabel_str="Predicted",
        index=None,
        log=True,maerun=False,
        x_min=None, x_max=None,
        save_path=None
):
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    import numpy as np
    import os
    
    if maerun:
        # For each point, compute error% = |(true - pred)/true|
        mae = np.abs((y_true - y_pred))
    else:
        # For each point, compute error% = |(true - pred)/true| * 100
        mae = np.abs((y_true - y_pred)/y_true)*100
    mae_mean = np.mean(mae)
    r2 = r2_score(y_true,y_pred)
    
    # Determine the full range for plotting from the combined true values.
    if (not x_min):
        x_min = min(np.min(y_true), np.min(y_pred))
    if (not x_max):
        x_max = max(np.max(y_true), np.max(y_pred))
    x_line = np.linspace(x_min, x_max, 100)
    
    # Plots a simple y_pred vs. y_true scatter plus a y=x line.
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='k')
    
    ax.plot(x_line, x_line, 'r--', label='Ideal: y = x')
    if maerun:
        error_threshold = 0.1
        # Plot the MAE lines.
        ax.plot(x_line, np.add(error_threshold, x_line), 'g--', label ='MAE = 0.1')
        ax.plot(x_line, np.add(-1*error_threshold, x_line), 'g--')
    else:
        error_threshold = 0.2
        # Plot the error lines.
        ax.plot(x_line, (1 + error_threshold)*x_line, 'g--', label ='20% Error')
        ax.plot(x_line, (1 - error_threshold)*x_line, 'g--')
    
    for i in range(len(y_true)):
        if maerun:
            error = abs(y_true[i] - y_pred[i])
        else:
            error = abs(y_true[i] - y_pred[i])/y_true[i]
        if error > error_threshold:
            if index is not None:
                ax.annotate(str(int(index[i])), (y_true[i], y_pred[i]), fontsize=8, color='red')
            else:
                ax.annotate(str(int(i+1)), (y_true[i], y_pred[i]), fontsize=8, color='red')
    
    if maerun:
        ax.text(0.05, 0.95, f'Avg MAE: {mae_mean:.3f}\nR² Score: {r2:.4f}', transform=ax.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    else:
        ax.text(0.05, 0.95, f'Avg % Error: {mae_mean:.2f}%\nR² Score: {r2:.4f}', transform=ax.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_title(title_str)
    ax.set_xlabel(xlabel_str)
    ax.set_ylabel(ylabel_str)
    ax.legend()
    ax.grid(True)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)