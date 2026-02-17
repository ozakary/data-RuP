import json
import matplotlib.pyplot as plt
import numpy as np

# Import the figure_formatting module
import figure_formatting_v2 as ff

# Set up figure formatting using the function from the module
ff.set_rcParams(ff.master_formatting)


def read_training_log(log_file):
    """Read the JSON-based training log file and extract metrics."""
    training_data = []
    validation_data = []
    lr_data = []
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                
                if data.get('mode') == 'opt':
                    # Training data
                    epoch = data.get('epoch')
                    if epoch is not None:
                        training_data.append({
                            'epoch': epoch,
                            'loss': data.get('loss'),
                            'time': data.get('time')
                        })
                
                elif data.get('mode') == 'eval':
                    # Validation data
                    epoch = data.get('epoch')
                    if epoch is not None:
                        validation_data.append({
                            'epoch': epoch,
                            'loss': data.get('loss'),
                            'mae_stress': data.get('mae_stress'),
                            'rmse_stress': data.get('rmse_stress'),
                            'q95_stress': data.get('q95_stress'),
                            'time': data.get('time')
                        })
                
                # Extract learning rate if available
                if 'lr' in data or 'LR' in data:
                    lr_data.append({
                        'epoch': data.get('epoch'),
                        'lr': data.get('lr') or data.get('LR')
                    })
                    
            except json.JSONDecodeError:
                continue
    
    return training_data, validation_data, lr_data


def aggregate_by_epoch(data_list):
    """Aggregate metrics by epoch (average multiple values per epoch)."""
    from collections import defaultdict
    
    epoch_data = defaultdict(list)
    
    for entry in data_list:
        epoch = entry['epoch']
        epoch_data[epoch].append(entry)
    
    aggregated = []
    for epoch in sorted(epoch_data.keys()):
        entries = epoch_data[epoch]
        
        # Average all metrics for this epoch
        avg_entry = {'epoch': epoch}
        keys_to_avg = set()
        for entry in entries:
            keys_to_avg.update(entry.keys())
        keys_to_avg.remove('epoch')
        
        for key in keys_to_avg:
            values = [e[key] for e in entries if key in e and e[key] is not None]
            if values:
                avg_entry[key] = np.mean(values)
        
        aggregated.append(avg_entry)
    
    return aggregated


def plot_stress_metrics(log_file, output_file='plot_ml_process_stress.svg'):
    """Plot stress-related metrics from training log."""
    
    # Read the data
    training_data, validation_data, lr_data = read_training_log(log_file)
    
    # Aggregate by epoch
    training_agg = aggregate_by_epoch(training_data)
    validation_agg = aggregate_by_epoch(validation_data)
    
    # Extract data for plotting
    train_epochs = [d['epoch'] for d in training_agg]
    train_loss = [d['loss'] for d in training_agg]
    
    val_epochs = [d['epoch'] for d in validation_agg]
    val_loss = [d.get('loss', np.nan) for d in validation_agg]
    val_mae = [d.get('mae_stress', np.nan) for d in validation_agg]
    val_rmse = [d.get('rmse_stress', np.nan) for d in validation_agg]
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    
    # Plot training loss
    plt.loglog(train_epochs, train_loss, '-', color='#264653', label=r'$\boldsymbol{s}_{\text{Loss}}^{\text{T}}$')
    
    # Plot validation metrics
    if any(~np.isnan(val_loss)):
        plt.loglog(val_epochs, val_loss, '--', color='#264653', label=r'$\boldsymbol{s}_{\text{Loss}}^{\text{V}}$')
    if any(~np.isnan(val_mae)):
        plt.loglog(val_epochs, val_mae, '--', color='#2A9D8F', label=r'$\boldsymbol{s}_{\text{MAE}}^{\text{V}}$')
    if any(~np.isnan(val_rmse)):
        plt.loglog(val_epochs, val_rmse, '--', color='#E76F51', label=r'$\boldsymbol{s}_{\text{RMSE}}^{\text{V}}$')
    
    # Add learning rate if available
    if lr_data:
        lr_agg = aggregate_by_epoch(lr_data)
        lr_epochs = [d['epoch'] for d in lr_agg]
        lr_values = [d['lr'] * 1e-6 for d in lr_agg]
        plt.loglog(lr_epochs, lr_values, '-', color='#E9C46A', label=r'$LR \times 10^{-6}$')
    
    # Add labels and formatting
    plt.xlabel('Epoch')
    plt.ylabel('Error / eV·Å$^{-3}$')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(frameon=True, fontsize=16, facecolor="white", edgecolor="gray", framealpha=0.8)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    
    print(f"Stress plot saved to {output_file}")
    print(f"Training epochs: {len(train_epochs)}")
    print(f"Validation epochs: {len(val_epochs)}")


if __name__ == "__main__":
    # Update this path to your actual log file
    log_file = './results/fine-tuned_mace-mp-0b3-medium_run-3_train.txt'
    plot_stress_metrics(log_file)
