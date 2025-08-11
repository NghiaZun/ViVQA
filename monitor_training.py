"""
Script để monitor training progress và tạo plots
"""
import matplotlib.pyplot as plt
import re
import os
import pandas as pd

def parse_training_log(log_file="training.log"):
    """Parse training log để extract loss values"""
    epochs = []
    losses = []
    best_losses = []
    learning_rates = []
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse dòng: "⏰ Epoch 1/100 | Loss: 2.3456 | Best: 2.1234 | LR: 0.000200"
            match = re.search(r'Epoch (\d+)/\d+ \| Loss: ([\d\.]+) \| Best: ([\d\.]+) \| LR: ([\d\.e-]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                best_loss = float(match.group(3))
                lr = float(match.group(4))
                
                epochs.append(epoch)
                losses.append(loss)
                best_losses.append(best_loss)
                learning_rates.append(lr)
    
    return {
        'epochs': epochs,
        'losses': losses,
        'best_losses': best_losses,
        'learning_rates': learning_rates
    }

def plot_training_progress(data, save_path="training_progress.png"):
    """Tạo plots cho training progress"""
    if data is None:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Loss curve
    ax1.plot(data['epochs'], data['losses'], 'b-', label='Training Loss', alpha=0.7)
    ax1.plot(data['epochs'], data['best_losses'], 'r-', label='Best Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Learning Rate
    ax2.plot(data['epochs'], data['learning_rates'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 3. Loss improvement rate
    if len(data['losses']) > 1:
        loss_diff = [data['losses'][i] - data['losses'][i-1] for i in range(1, len(data['losses']))]
        ax3.plot(data['epochs'][1:], loss_diff, 'orange', alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Change')
        ax3.set_title('Loss Improvement Rate')
        ax3.grid(True, alpha=0.3)
    
    # 4. Training statistics
    ax4.text(0.1, 0.9, f"📊 TRAINING STATISTICS", transform=ax4.transAxes, fontsize=14, fontweight='bold')
    ax4.text(0.1, 0.8, f"Total Epochs: {len(data['epochs'])}", transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.7, f"Best Loss: {min(data['best_losses']):.4f}", transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.6, f"Current Loss: {data['losses'][-1]:.4f}", transform=ax4.transAxes, fontsize=12)
    ax4.text(0.1, 0.5, f"Current LR: {data['learning_rates'][-1]:.2e}", transform=ax4.transAxes, fontsize=12)
    
    if len(data['losses']) > 1:
        improvement = ((data['losses'][0] - data['losses'][-1]) / data['losses'][0]) * 100
        ax4.text(0.1, 0.4, f"Total Improvement: {improvement:.1f}%", transform=ax4.transAxes, fontsize=12)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Plot saved: {save_path}")

def export_training_csv(data, save_path="training_data.csv"):
    """Export training data to CSV"""
    if data is None:
        return
    
    df = pd.DataFrame({
        'epoch': data['epochs'],
        'loss': data['losses'],
        'best_loss': data['best_losses'],
        'learning_rate': data['learning_rates']
    })
    
    df.to_csv(save_path, index=False)
    print(f"✅ Data exported: {save_path}")

def monitor_current_training():
    """Monitor current training session"""
    print("🔍 Monitoring training progress...")
    
    # Check for various log files
    log_files = ["training.log", "nohup.out", "train_output.txt"]
    log_file = None
    
    for file in log_files:
        if os.path.exists(file):
            log_file = file
            break
    
    if log_file is None:
        print("❌ No training log file found. Try redirecting output:")
        print("python train.py > training.log 2>&1")
        return
    
    data = parse_training_log(log_file)
    if data and len(data['epochs']) > 0:
        plot_training_progress(data)
        export_training_csv(data)
        
        # Print summary
        print("\n📈 TRAINING SUMMARY:")
        print(f"Epochs completed: {len(data['epochs'])}")
        print(f"Best loss achieved: {min(data['best_losses']):.4f}")
        print(f"Latest loss: {data['losses'][-1]:.4f}")
        print(f"Current learning rate: {data['learning_rates'][-1]:.2e}")
    else:
        print("❌ No training data found in log file")

if __name__ == "__main__":
    monitor_current_training()
