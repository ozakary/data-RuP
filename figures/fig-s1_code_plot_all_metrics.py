import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

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
                            # Energy metrics
                            'mae_e': data.get('mae_e'),
                            'mae_e_per_atom': data.get('mae_e_per_atom'),
                            'rmse_e': data.get('rmse_e'),
                            'rmse_e_per_atom': data.get('rmse_e_per_atom'),
                            'q95_e': data.get('q95_e'),
                            # Force metrics
                            'mae_f': data.get('mae_f'),
                            'rmse_f': data.get('rmse_f'),
                            'rel_mae_f': data.get('rel_mae_f'),
                            'rel_rmse_f': data.get('rel_rmse_f'),
                            'q95_f': data.get('q95_f'),
                            # Stress metrics
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


def plot_all_metrics(log_file, output_prefix='plot_ml_process'):
    """Plot all metrics (energy, forces, stress) from training log."""
    
    # Read the data
    print(f"Reading training log from: {log_file}")
    training_data, validation_data, lr_data = read_training_log(log_file)
    
    # Aggregate by epoch
    training_agg = aggregate_by_epoch(training_data)
    validation_agg = aggregate_by_epoch(validation_data)
    
    print(f"Found {len(training_agg)} training epochs")
    print(f"Found {len(validation_agg)} validation epochs")
    
    # Extract common data
    train_epochs = [d['epoch'] for d in training_agg]
    train_loss = [d['loss'] for d in training_agg]
    
    val_epochs = [d['epoch'] for d in validation_agg]
    val_loss = [d.get('loss', np.nan) for d in validation_agg]
    
    # Prepare learning rate data if available
    lr_epochs = None
    lr_values = None
    if lr_data:
        lr_agg = aggregate_by_epoch(lr_data)
        lr_epochs = [d['epoch'] for d in lr_agg]
        lr_values = [d['lr'] for d in lr_agg]
    
    # ========== ENERGY PLOT ==========
    print("\nGenerating energy plot...")
    plt.figure(figsize=(10, 5))
    
    val_mae_e = [d.get('mae_e_per_atom', np.nan) for d in validation_agg]
    val_rmse_e = [d.get('rmse_e_per_atom', np.nan) for d in validation_agg]
    
    plt.loglog(train_epochs, train_loss, '-', color='#264653', linewidth=2.0, label=r'$E_{\text{Loss}}^{\text{T}}$')
    
    if any(~np.isnan(val_loss)):
        plt.loglog(val_epochs, val_loss, '--', color='#264653', linewidth=2.0, label=r'$E_{\text{Loss}}^{\text{V}}$')
    if any(~np.isnan(val_mae_e)):
        plt.loglog(val_epochs, val_mae_e, '--', color='#2A9D8F', linewidth=2.0, label=r'$E_{\text{MAE}}^{\text{V}}$')
    if any(~np.isnan(val_rmse_e)):
        plt.loglog(val_epochs, val_rmse_e, '--', color='#E76F51', linewidth=2.0, label=r'$E_{\text{RMSE}}^{\text{V}}$')
    
    if lr_epochs is not None:
        plt.loglog(lr_epochs, [lr * 1e-9 for lr in lr_values], '-', 
                  color='#E9C46A', label=r'$LR \times 10^{-9}$')
    
    plt.xlabel('Epoch')
    plt.ylabel('Error / eV·atom$^{-1}$')
    plt.grid(True, which="both", ls="-", alpha=0.3, linewidth=2.0)
    plt.legend(frameon=True, fontsize=16, facecolor="white", edgecolor="gray", framealpha=0.8)
    plt.tight_layout()
    
    energy_output = f'{output_prefix}_energy.svg'
    plt.savefig(energy_output, bbox_inches='tight')
    print(f"  Saved to: {energy_output}")
    plt.close()
    
    # ========== FORCE PLOT ==========
    print("Generating force plot...")
    plt.figure(figsize=(10, 5))
    
    val_mae_f = [d.get('mae_f', np.nan) for d in validation_agg]
    val_rmse_f = [d.get('rmse_f', np.nan) for d in validation_agg]
    
    plt.loglog(train_epochs, train_loss, '-', color='#264653', linewidth=2.0, label=r'$\vec{f}_{\text{Loss}}^{\text{T}}$')
    
    if any(~np.isnan(val_loss)):
        plt.loglog(val_epochs, val_loss, '--', color='#264653', linewidth=2.0, label=r'$\vec{f}_{\text{Loss}}^{\text{V}}$')
    if any(~np.isnan(val_mae_f)):
        plt.loglog(val_epochs, val_mae_f, '--', color='#2A9D8F', linewidth=2.0, label=r'$\vec{f}_{\text{MAE}}^{\text{V}}$')
    if any(~np.isnan(val_rmse_f)):
        plt.loglog(val_epochs, val_rmse_f, '--', color='#E76F51', linewidth=2.0, label=r'$\vec{f}_{\text{RMSE}}^{\text{V}}$')
    
    if lr_epochs is not None:
        plt.loglog(lr_epochs, [lr * 1e-3 for lr in lr_values], '-', 
                  color='#E9C46A', label=r'$LR \times 10^{-3}$')
    
    plt.xlabel('Epoch')
    plt.ylabel('Error / eV·Å$^{-1}$')
    plt.grid(True, which="both", ls="-", alpha=0.3, linewidth=2.0)
    plt.legend(frameon=True, fontsize=16, facecolor="white", edgecolor="gray", framealpha=0.8)
    plt.tight_layout()
    
    force_output = f'{output_prefix}_forces.svg'
    plt.savefig(force_output, bbox_inches='tight')
    print(f"  Saved to: {force_output}")
    plt.close()
    
    # ========== STRESS PLOT ==========
    print("Generating stress plot...")
    plt.figure(figsize=(10, 5))
    
    val_mae_s = [d.get('mae_stress', np.nan) for d in validation_agg]
    val_rmse_s = [d.get('rmse_stress', np.nan) for d in validation_agg]
    
    plt.loglog(train_epochs, train_loss, '-', color='#264653', linewidth=2.0, label=r'$\boldsymbol{s}_{\text{Loss}}^{\text{T}}$')
    
    if any(~np.isnan(val_loss)):
        plt.loglog(val_epochs, val_loss, '--', color='#264653', linewidth=2.0, label=r'$\boldsymbol{s}_{\text{Loss}}^{\text{V}}$')
    if any(~np.isnan(val_mae_s)):
        plt.loglog(val_epochs, val_mae_s, '--', color='#2A9D8F', linewidth=2.0, label=r'$\boldsymbol{s}_{\text{MAE}}^{\text{V}}$')
    if any(~np.isnan(val_rmse_s)):
        plt.loglog(val_epochs, val_rmse_s, '--', color='#E76F51', linewidth=2.0, label=r'$\boldsymbol{s}_{\text{RMSE}}^{\text{V}}$')
    
    if lr_epochs is not None:
        plt.loglog(lr_epochs, [lr * 1e-6 for lr in lr_values], '-', 
                  color='#E9C46A', label=r'$LR \times 10^{-6}$')
    
    plt.xlabel('Epoch')
    plt.ylabel('Error / eV·Å$^{-3}$')
    plt.grid(True, which="both", ls="-", alpha=0.3, linewidth=2.0)
    plt.legend(frameon=True, fontsize=16, facecolor="white", edgecolor="gray", framealpha=0.8)
    plt.tight_layout()
    
    stress_output = f'{output_prefix}_stress.svg'
    plt.savefig(stress_output, bbox_inches='tight')
    print(f"  Saved to: {stress_output}")
    plt.close()
    
    print("\n" + "="*60)
    print("All plots generated successfully!")
    print("="*60)


if __name__ == "__main__":
    # Update this path to your actual log file
    log_file = './results/fine-tuned_mace-mp-0b3-medium_vf_run-3_train.txt'
    plot_all_metrics(log_file)
