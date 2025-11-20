"""
Training Progress Visualization
Plots loss curves, weight schedules, and performance metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def plot_training_progress(log_csv, save_path="training_progress.png"):
    """
    Create comprehensive training visualization
    """
    df = pd.read_csv(log_csv)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('VQA Student Training Progress', fontsize=16, fontweight='bold')
    
    # 1. Overall Loss
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(df['epoch'], df['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Overall Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Component Losses (Train)
    ax = axes[0, 1]
    ax.plot(df['epoch'], df['train_F'], 'g-', label='Format', linewidth=2)
    ax.plot(df['epoch'], df['train_A'], 'b-', label='Answer', linewidth=2)
    ax.plot(df['epoch'], df['train_R'], 'orange', label='Reasoning', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Component Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Component Losses (Val)
    ax = axes[0, 2]
    ax.plot(df['epoch'], df['val_F'], 'g-', label='Format', linewidth=2)
    ax.plot(df['epoch'], df['val_A'], 'b-', label='Answer', linewidth=2)
    ax.plot(df['epoch'], df['val_R'], 'orange', label='Reasoning', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Component Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Weight Schedule
    ax = axes[1, 0]
    if 'w_format' in df.columns:
        ax.plot(df['epoch'], df['w_format'], 'g-', label='w_format', linewidth=2)
        ax.plot(df['epoch'], df['w_answer'], 'b-', label='w_answer', linewidth=2)
        ax.plot(df['epoch'], df['w_reason'], 'orange', label='w_reason', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight')
        ax.set_title('Adaptive Weight Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mark curriculum stages if applicable
        if len(df) > 30:
            ax.axvline(x=15, color='red', linestyle='--', alpha=0.5, label='Stage 1→2')
            ax.axvline(x=30, color='purple', linestyle='--', alpha=0.5, label='Stage 2→3')
    else:
        ax.text(0.5, 0.5, 'Weight schedule not available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Weight Schedule (N/A)')
    
    # 5. Learning Rate
    ax = axes[1, 1]
    if 'lr' in df.columns:
        ax.plot(df['epoch'], df['lr'], 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'LR not logged', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Learning Rate (N/A)')
    
    # 6. Loss Improvement
    ax = axes[1, 2]
    train_improvement = df['train_loss'].iloc[0] - df['train_loss']
    val_improvement = df['val_loss'].iloc[0] - df['val_loss']
    ax.plot(df['epoch'], train_improvement, 'b-', label='Train', linewidth=2)
    ax.plot(df['epoch'], val_improvement, 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Reduction')
    ax.set_title('Loss Improvement from Start')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training visualization saved to {save_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total epochs:        {len(df)}")
    print(f"Initial train loss:  {df['train_loss'].iloc[0]:.4f}")
    print(f"Final train loss:    {df['train_loss'].iloc[-1]:.4f}")
    print(f"Best train loss:     {df['train_loss'].min():.4f} (epoch {df['train_loss'].idxmin()+1})")
    print(f"Initial val loss:    {df['val_loss'].iloc[0]:.4f}")
    print(f"Final val loss:      {df['val_loss'].iloc[-1]:.4f}")
    print(f"Best val loss:       {df['val_loss'].min():.4f} (epoch {df['val_loss'].idxmin()+1})")
    print(f"Total improvement:   {df['train_loss'].iloc[0] - df['train_loss'].iloc[-1]:.4f}")
    print("="*60)

def plot_evaluation_results(eval_csv, save_path="evaluation_results.png"):
    """
    Visualize evaluation metrics
    """
    df = pd.read_csv(eval_csv)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('VQA Model Evaluation Results', fontsize=16, fontweight='bold')
    
    # 1. Metric Distribution
    ax = axes[0, 0]
    metrics = ['exact_match', 'token_f1', 'rouge1', 'rougeL']
    means = [df[m].mean() for m in metrics]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = ax.bar(range(len(metrics)), means, color=colors, alpha=0.7)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(['EM', 'Token-F1', 'ROUGE-1', 'ROUGE-L'])
    ax.set_ylabel('Score')
    ax.set_title('Average Metrics')
    ax.set_ylim([0, 1])
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{means[i]:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 2. Format Correctness
    ax = axes[0, 1]
    if 'has_correct_format' in df.columns:
        format_correct = df['has_correct_format'].sum()
        format_incorrect = len(df) - format_correct
        sizes = [format_correct, format_incorrect]
        colors_pie = ['#2ECC71', '#E74C3C']
        explode = (0.1, 0)
        ax.pie(sizes, explode=explode, labels=['Correct Format', 'Incorrect Format'],
               colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Format Correctness\n({format_correct}/{len(df)} correct)')
    else:
        ax.text(0.5, 0.5, 'Format data not available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Format Correctness (N/A)')
    
    # 3. Score Distribution
    ax = axes[1, 0]
    ax.hist(df['token_f1'], bins=20, alpha=0.6, label='Token-F1', color='blue')
    ax.hist(df['rouge1'], bins=20, alpha=0.6, label='ROUGE-1', color='green')
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Performance by Question Length
    ax = axes[1, 1]
    if 'question' in df.columns:
        df['q_len'] = df['question'].str.split().str.len()
        bins = [0, 5, 10, 15, 100]
        labels = ['1-5', '6-10', '11-15', '16+']
        df['q_len_bin'] = pd.cut(df['q_len'], bins=bins, labels=labels)
        
        grouped = df.groupby('q_len_bin')['token_f1'].mean()
        grouped.plot(kind='bar', ax=ax, color='teal', alpha=0.7)
        ax.set_xlabel('Question Length (words)')
        ax.set_ylabel('Average Token-F1')
        ax.set_title('Performance by Question Length')
        ax.set_xticklabels(labels, rotation=0)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Question data not available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Performance by Length (N/A)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Evaluation visualization saved to {save_path}")

def compare_models(old_eval_csv, new_eval_csv, save_path="model_comparison.png"):
    """
    Compare old and new model performance
    """
    df_old = pd.read_csv(old_eval_csv)
    df_new = pd.read_csv(new_eval_csv)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Comparison: Before vs After', fontsize=16, fontweight='bold')
    
    # 1. Metrics comparison
    ax = axes[0]
    metrics = ['exact_match', 'token_f1', 'rouge1']
    old_means = [df_old[m].mean() for m in metrics]
    new_means = [df_new[m].mean() for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, old_means, width, label='Old Model', color='#E74C3C', alpha=0.7)
    bars2 = ax.bar(x + width/2, new_means, width, label='New Model', color='#2ECC71', alpha=0.7)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['EM', 'Token-F1', 'ROUGE-1'])
    ax.legend()
    ax.set_ylim([0, 1])
    
    # Add improvement %
    for i in range(len(metrics)):
        improvement = ((new_means[i] - old_means[i]) / old_means[i]) * 100
        ax.text(i, max(old_means[i], new_means[i]) + 0.05,
                f'+{improvement:.1f}%' if improvement > 0 else f'{improvement:.1f}%',
                ha='center', fontweight='bold',
                color='green' if improvement > 0 else 'red')
    
    # 2. Format correctness comparison
    ax = axes[1]
    if 'has_correct_format' in df_old.columns and 'has_correct_format' in df_new.columns:
        old_format = df_old['has_correct_format'].mean()
        new_format = df_new['has_correct_format'].mean()
        
        bars = ax.bar(['Old Model', 'New Model'], [old_format, new_format],
                      color=['#E74C3C', '#2ECC71'], alpha=0.7)
        ax.set_ylabel('Format Correctness')
        ax.set_title('Format Correctness Comparison')
        ax.set_ylim([0, 1])
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison visualization saved to {save_path}")

if __name__ == "__main__":
    import sys
    
    print("📊 VQA Training & Evaluation Visualization Tool\n")
    
    # Example usage
    print("Usage:")
    print("  1. Plot training progress:")
    print("     python visualize_results.py train <log_csv_path>")
    print("  2. Plot evaluation results:")
    print("     python visualize_results.py eval <eval_csv_path>")
    print("  3. Compare models:")
    print("     python visualize_results.py compare <old_csv> <new_csv>")
    
    if len(sys.argv) < 2:
        print("\n⚠️  No arguments provided. Using example...")
        print("\nTo visualize your results, run:")
        print("  python visualize_results.py train /kaggle/working/train_val_log_ultimate.csv")
        print("  python visualize_results.py eval /kaggle/working/eval_adaptive_v3_results.csv")
    else:
        command = sys.argv[1]
        
        if command == "train" and len(sys.argv) > 2:
            plot_training_progress(sys.argv[2])
        elif command == "eval" and len(sys.argv) > 2:
            plot_evaluation_results(sys.argv[2])
        elif command == "compare" and len(sys.argv) > 3:
            compare_models(sys.argv[2], sys.argv[3])
        else:
            print("❌ Invalid command or missing arguments")
