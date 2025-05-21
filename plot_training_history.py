def plot_training_history(history, title,save_path=None):
    import matplotlib.pyplot as plt
    import os
    
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
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close(fig)