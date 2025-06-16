def plot_training_history(history, title,
                          y_min=None,y_max=None,
                          save_path=None):
    import matplotlib.pyplot as plt
    import os
    
    # Determine the full range for plotting from the combined true values.
    if (not y_min):
        if 'val_loss' in history.history:
            y_min = 0.9*min(min(history.history['loss']), min(history.history['val_loss']))
        else:
            y_min = 0.9*min(history.history['loss'])
    if (not y_max):
        if 'val_loss' in history.history:
            y_max = 1.1*max(max(history.history['loss']), max(history.history['val_loss']))
        else:
            y_max = 1.1*max(history.history['loss'])
    
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0,len(history.history['loss']))
    ax.set_ylim(y_min, y_max)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close(fig)