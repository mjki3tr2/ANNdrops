def plot_outputs(y_true_train,y_true_valid,y_true_test,y_pred_train,y_pred_valid,y_pred_test,idx_f_train,idx_f_valid,idx_f_test,title,
                     log=True, maerun=False,
                     save_path=None):
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    import os
    
    # Combine all true and predicted values
    y_true_all = np.concatenate([y_true_train, y_true_valid, y_true_test])
    y_pred_all = np.concatenate([y_pred_train, y_pred_valid, y_pred_test])
    
    if maerun:
        # For each point, compute error% = |(true - pred)/true|
        mae = np.abs((y_true_all - y_pred_all))
    else:
        # For each point, compute error% = |(true - pred)/true| * 100
        mae = np.abs((y_true_all - y_pred_all)/y_true_all)*100
    mae_mean = np.mean(mae)
    r2 = r2_score(y_true_all,y_pred_all)
    
    # Determine the full range for plotting from the combined true values.
    x_min = y_true_all.min()
    x_max = y_true_all.max()
    x_line = np.linspace(x_min, x_max, 100)
    
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(8,8))
    
    ax.scatter(y_true_train, y_pred_train, c='blue', label='Train', alpha=0.7, edgecolors='k')
    ax.scatter(y_true_valid, y_pred_valid, c='green', label='Valid', alpha=0.7, edgecolors='k')
    ax.scatter(y_true_test, y_pred_test, c='orange', label='Test', alpha=0.7, edgecolors='k')
    
    # Plot the ideal parity line: y=x.
    ax.plot([x_min, x_max], [x_min, x_max], 'r--', label="Ideal: y=x")
    
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
    
    
    # Annotate outliers in the test set
    for i, idx in enumerate(idx_f_test):
        if maerun:
            error = abs(y_true_test[i] - y_pred_test[i])
        else:
            error = abs(y_true_test[i] - y_pred_test[i])/y_true_test[i]
        if error > error_threshold:
            ax.annotate(str(int(idx)), (y_true_test[i], y_pred_test[i]), fontsize=8, color='red')
    
    for i, idx in enumerate(idx_f_valid):
        if maerun:
            error = abs(y_true_valid[i] - y_pred_valid[i])
        else:
            error = abs(y_true_valid[i] - y_pred_valid[i])/y_true_valid[i]
        if error > error_threshold:
            ax.annotate(str(int(idx)), (y_true_valid[i], y_pred_valid[i]), fontsize=8, color='red')
    
    for i, idx in enumerate(idx_f_train):
        if maerun:
            error = abs(y_true_train[i] - y_pred_train[i])
        else:
            error = abs(y_true_train[i] - y_pred_train[i])/y_true_train[i]
        if error > error_threshold:
            ax.annotate(str(int(idx)), (y_true_train[i], y_pred_train[i]), fontsize=8, color='red')
    
    if maerun:
        ax.text(0.05, 0.95, f'Avg MAE: {mae_mean:.3f}\nR² Score: {r2:.4f}', transform=ax.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    else:
        ax.text(0.05, 0.95, f'Avg % Error: {mae_mean:.2f}%\nR² Score: {r2:.4f}', transform=ax.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)