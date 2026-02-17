import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker  # Add this import
# Import the figure_formatting module
import figure_formatting_v2 as ff
# Set up figure formatting using the function from the module
ff.set_rcParams(ff.master_formatting)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import Optional, Dict, List, Any

# Read the CSV file
def plot_metrics(csv_file):
    # Read the CSV file, handling the whitespace in the headers
    df = pd.read_csv(csv_file, skipinitialspace=True)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))  # Changed to get ax handle
    
    # Plot the metrics on log-log scale
    ax.loglog(df['epoch'], df['training_loss_f'], '-', color='#264653', label='$f_{Loss}^{T}$')   
    ax.loglog(df['epoch'], df['training_f_mae'], '-', color='#2A9D8F', label='$f_{MAE}^{T}$')
    ax.loglog(df['epoch'], df['training_f_rmse'], '-', color='#E76F51', label='$f_{RMSE}^{T}$')
    ax.loglog(df['epoch'], df['validation_loss_f'], '--', color='#264653', label='$f_{Loss}^{V}$')   
    ax.loglog(df['epoch'], df['validation_f_mae'], '--', color='#2A9D8F', label='$f_{MAE}^{V}$')
    ax.loglog(df['epoch'], df['validation_f_rmse'], '--', color='#E76F51', label='$f_{RMSE}^{V}$') 
    ax.loglog(df['epoch'], df['LR'], '-', color='#E9C46A', label=r'$LR \times 10^{-3}$') 
    
    # Fix minor ticks for both axes
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 8) * 0.1, numticks=12))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 8) * 0.1, numticks=12))
    
   
    # Add labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(frameon=True, fontsize=18, facecolor="white", edgecolor="gray", framealpha=0.8)
    
    # Show the plot
    plt.tight_layout()
    plt.savefig("plot_ml_process_forces.svg", bbox_inches='tight')
    plt.show()

# Call the function with your CSV file
plot_metrics('../../MLP-mono_ortho_output/monoclinic_and_orthorhombic_vf_smaller-r_cut_batch-size_8/metrics_epoch.csv')
