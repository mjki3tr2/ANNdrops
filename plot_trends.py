def plot_trends(data,y1,y2,y3,
                expx,real1,real2,real3,
                title,xlabel,ylabel,
                log=False,
                save_path=None
          ):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    range=0.2
    
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(8,6))
    ax.fill_between(data, y3 * (1-range), y3 * (1+range), color='b', alpha=0.2)
    ax.plot(data, y3, 'b--', label='Small')
    ax.scatter(expx, real3, color='b', marker='o')
    ax.fill_between(data, y2 * (1-range), y2 * (1+range), color='g', alpha=0.2)
    ax.plot(data, y2, 'g--', label='Medium')
    ax.scatter(expx, real2, color='g', marker='o')
    ax.fill_between(data, y1 * (1-range), y1 * (1+range), color='r', alpha=0.2)
    ax.plot(data, y1, 'r--', label='Large')
    ax.scatter(expx, real1, color='r', marker='o')
    
    #Log x-axis
    ax.set_xscale('log')
    
    if log:
        ax.set_yscale('log')

    # Add labels and a title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    #Limit axis so all are comparable
    #ax.set_xlim([0.001, 10000])
    #ax.set_ylim([0, 20])
    
    ax.legend()
    ax.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)