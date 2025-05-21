def plot_DSD_comp(diameter,DSD_exp,labels,properties,
                  y_small_cum,y_medium_cum,y_large_cum,y_total_cum,
                  index,
                  save_path=None
          ):
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    import numpy as np
    import os
    
    r2 = r2_score(DSD_exp[index-1],y_total_cum[index-1])
    
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(diameter, DSD_exp[index-1], label='Experimental')
    ax.plot(diameter, y_total_cum[index-1], 'k-', label='Total')
    ax.plot(diameter, y_small_cum[index-1], 'b--', label='Small')
    ax.plot(diameter, y_medium_cum[index-1], 'g--', label='Medium')
    ax.plot(diameter, y_large_cum[index-1], 'r--', label='Large')
    
    
    #Log x-axis
    ax.set_xscale('log')

    # Add labels and a title
    ax.set_title(rf"""{labels[0]} = {properties[index-1][0]:.0f}""")
    ax.set_xlabel(r'Diameter / $\mu$m')
    ax.set_ylabel('Vol fraction')

    #Limit axis so all are comparable
    ax.set_xlim([0.001, 10000])
    ax.set_ylim([0, 20])
    
    ax.legend()
    ax.grid(True)

    ax.text(0.02,0.98, rf"""{labels[1]} = {properties[index-1][1]:.0f}
{labels[2]} = {properties[index-1][2]}
{labels[3]} = {properties[index-1][3]}
{labels[4]} = {properties[index-1][4]}
{labels[5]} = {properties[index-1][5]}
{labels[6]} = {properties[index-1][6]}
{labels[7]} = {properties[index-1][7]}
R$^2$ Score = {r2:.4f}""",transform=ax.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)