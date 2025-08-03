"""
Plotting utilities for training loss and evaluation metrics
Visualizes training progress and model performance comparisons
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os
import seaborn as sns
from typing import List, Dict, Optional
from datetime import datetime

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_training_logs(results_dir: str = "../results") -> pd.DataFrame:
    """Load all training log CSV files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_path = os.path.join(project_root, "results")
    
    pattern = os.path.join(results_path, "training_log_*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No training log files found in {results_path}")
        return pd.DataFrame()
    
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        return combined_df
    else:
        return pd.DataFrame()

def load_evaluation_results(results_dir: str = "../results") -> pd.DataFrame:
    """Load all evaluation result CSV files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_path = os.path.join(project_root, "results")
    
    pattern = os.path.join(results_path, "evaluation_results_*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No evaluation result files found in {results_path}")
        return pd.DataFrame()
    
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Add evaluation run timestamp from filename
            filename = os.path.basename(file)
            eval_timestamp = filename.replace('evaluation_results_', '').replace('.csv', '')
            df['eval_timestamp'] = eval_timestamp
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

def plot_training_loss(training_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot training loss curves for all runs"""
    if training_df.empty:
        print("No training data to plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot loss curves for each run
    for run_id in training_df['run_id'].unique():
        run_data = training_df[training_df['run_id'] == run_id]
        model_type = 'Without Attention' if 'no_attention' in run_id else 'With Attention'
        plt.plot(run_data['epoch'], run_data['loss'], 
                label=f"{model_type} ({run_id.split('_')[-1]})", linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Curves', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training loss plot saved to {save_path}")
    
    plt.show()

def plot_model_comparison(eval_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot comparison of evaluation metrics between models"""
    if eval_df.empty:
        print("No evaluation data to plot")
        return
    
    # Get the latest evaluation results
    latest_eval = eval_df[eval_df['eval_timestamp'] == eval_df['eval_timestamp'].max()]
    
    metrics = ['bleu_score', 'accuracy', 'bert_precision', 'bert_recall', 'bert_f1']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Create bar plot
        models = latest_eval['model'].values
        values = latest_eval[metric].values
        
        bars = ax.bar(models, values, color=['skyblue', 'lightcoral'])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, max(values) * 1.2)
        ax.grid(True, alpha=0.3)
    
    # Remove the extra subplot
    fig.delaxes(axes[5])
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    plt.show()

def plot_training_and_eval_timeline(training_df: pd.DataFrame, eval_df: pd.DataFrame, 
                                   save_path: Optional[str] = None):
    """Plot training loss and evaluation results over time"""
    if training_df.empty and eval_df.empty:
        print("No data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Training loss over time
    if not training_df.empty:
        for run_id in training_df['run_id'].unique():
            run_data = training_df[training_df['run_id'] == run_id].sort_values('timestamp')
            model_type = 'Without Attention' if 'no_attention' in run_id else 'With Attention'
            ax1.plot(run_data['timestamp'], run_data['loss'], 
                    label=f"{model_type}", linewidth=2, marker='o', markersize=4)
        
        ax1.set_xlabel('Training Time')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss Over Time', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Evaluation metrics over time
    if not eval_df.empty:
        eval_timestamps = pd.to_datetime(eval_df['eval_timestamp'], format='%Y%m%d_%H%M%S')
        
        metrics_to_plot = ['bleu_score', 'accuracy', 'bert_f1']
        colors = ['blue', 'red', 'green']
        
        for metric, color in zip(metrics_to_plot, colors):
            for model in eval_df['model'].unique():
                model_data = eval_df[eval_df['model'] == model]
                model_timestamps = pd.to_datetime(model_data['eval_timestamp'], format='%Y%m%d_%H%M%S')
                
                ax2.plot(model_timestamps, model_data[metric], 
                        label=f"{metric} ({model})", 
                        color=color, linestyle='-' if model == 'no_attention' else '--',
                        linewidth=2, marker='s', markersize=6)
        
        ax2.set_xlabel('Evaluation Time')
        ax2.set_ylabel('Score')
        ax2.set_title('Evaluation Metrics Over Time', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Timeline plot saved to {save_path}")
    
    plt.show()

def generate_summary_report(training_df: pd.DataFrame, eval_df: pd.DataFrame) -> str:
    """Generate a text summary of the results"""
    report = []
    report.append("=== TRAINING AND EVALUATION SUMMARY ===\n")
    
    if not training_df.empty:
        report.append("TRAINING RESULTS:")
        for run_id in training_df['run_id'].unique():
            run_data = training_df[training_df['run_id'] == run_id]
            model_type = 'Without Attention' if 'no_attention' in run_id else 'With Attention'
            initial_loss = run_data['loss'].iloc[0]
            final_loss = run_data['loss'].iloc[-1]
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            
            report.append(f"  {model_type} ({run_id}):")
            report.append(f"    Initial Loss: {initial_loss:.4f}")
            report.append(f"    Final Loss: {final_loss:.4f}")
            report.append(f"    Improvement: {improvement:.1f}%")
            report.append("")
    
    if not eval_df.empty:
        latest_eval = eval_df[eval_df['eval_timestamp'] == eval_df['eval_timestamp'].max()]
        
        report.append("LATEST EVALUATION RESULTS:")
        for _, row in latest_eval.iterrows():
            model_name = "With Attention" if row['model'] == 'with_attention' else "Without Attention"
            report.append(f"  {model_name}:")
            report.append(f"    BLEU Score: {row['bleu_score']:.4f}")
            report.append(f"    Accuracy: {row['accuracy']:.4f}")
            report.append(f"    BERT F1: {row['bert_f1']:.4f}")
            report.append(f"    Empty Responses: {row['empty_predictions']} ({row['empty_prediction_rate']:.1%})")
            report.append("")
    
    return "\n".join(report)

def plot_training_convergence_analysis(training_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot training convergence analysis showing improvement rates and final performance"""
    if training_df.empty:
        print("No training data for convergence analysis")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Loss improvement rates
    for run_id in training_df['run_id'].unique():
        run_data = training_df[training_df['run_id'] == run_id].sort_values('epoch')
        model_type = 'Without Attention' if 'no_attention' in run_id else 'With Attention'
        
        initial_loss = run_data['loss'].iloc[0]
        improvement_rates = ((initial_loss - run_data['loss']) / initial_loss) * 100
        
        ax1.plot(run_data['epoch'], improvement_rates, 
                label=f"{model_type}", linewidth=2, marker='o', markersize=4)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss Improvement (%)')
    ax1.set_title('Training Loss Improvement Rate', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss reduction velocity (derivative)
    for run_id in training_df['run_id'].unique():
        run_data = training_df[training_df['run_id'] == run_id].sort_values('epoch')
        model_type = 'Without Attention' if 'no_attention' in run_id else 'With Attention'
        
        if len(run_data) > 1:
            loss_velocity = -np.gradient(run_data['loss'].values)  # negative because we want reduction
            ax2.plot(run_data['epoch'], loss_velocity, 
                    label=f"{model_type}", linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Reduction Velocity')
    ax2.set_title('Training Velocity (Loss Reduction Rate)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Final performance comparison
    final_losses = []
    model_names = []
    improvements = []
    
    for run_id in training_df['run_id'].unique():
        run_data = training_df[training_df['run_id'] == run_id]
        model_type = 'Without Attention' if 'no_attention' in run_id else 'With Attention'
        
        initial_loss = run_data['loss'].iloc[0]
        final_loss = run_data['loss'].iloc[-1]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        
        final_losses.append(final_loss)
        model_names.append(model_type)
        improvements.append(improvement)
    
    bars = ax3.bar(model_names, final_losses, color=['skyblue', 'lightcoral'])
    for bar, loss in zip(bars, final_losses):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_ylabel('Final Loss')
    ax3.set_title('Final Training Loss Comparison', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Total improvement comparison
    bars = ax4.bar(model_names, improvements, color=['lightgreen', 'orange'])
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{imp:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title('Total Training Improvement', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Training Convergence Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence analysis plot saved to {save_path}")
    
    plt.show()


def plot_efficiency_analysis(training_df: pd.DataFrame, eval_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot training efficiency vs final performance"""
    if training_df.empty or eval_df.empty:
        print("Need both training and evaluation data for efficiency analysis")
        return
    
    # Get latest evaluation results
    latest_eval = eval_df[eval_df['eval_timestamp'] == eval_df['eval_timestamp'].max()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prepare data for efficiency analysis
    efficiency_data = []
    
    for _, eval_row in latest_eval.iterrows():
        model_name = eval_row['model']
        
        # Find corresponding training run (match by model type)
        training_subset = training_df[
            (training_df['run_id'].str.contains('attention') & (model_name == 'with_attention')) |
            (training_df['run_id'].str.contains('no_attention') & (model_name == 'no_attention'))
        ]
        
        if not training_subset.empty:
            # Get most recent training run for this model
            latest_run = training_subset['run_id'].iloc[-1]
            run_data = training_subset[training_subset['run_id'] == latest_run]
            
            total_epochs = len(run_data)
            final_loss = run_data['loss'].iloc[-1]
            initial_loss = run_data['loss'].iloc[0]
            
            efficiency_data.append({
                'model': 'With Attention' if model_name == 'with_attention' else 'Without Attention',
                'epochs': total_epochs,
                'final_loss': final_loss,
                'bleu_score': eval_row['bleu_score'],
                'bert_f1': eval_row['bert_f1'],
                'accuracy': eval_row['accuracy'],
                'improvement_rate': ((initial_loss - final_loss) / initial_loss) / total_epochs * 100
            })
    
    if not efficiency_data:
        print("No matching training/evaluation data found")
        return
    
    # Plot 1: Training Efficiency vs BLEU Score
    models = [d['model'] for d in efficiency_data]
    improvement_rates = [d['improvement_rate'] for d in efficiency_data]
    bleu_scores = [d['bleu_score'] for d in efficiency_data]
    
    colors = ['skyblue' if 'Without' in model else 'lightcoral' for model in models]
    scatter = ax1.scatter(improvement_rates, bleu_scores, c=colors, s=100, alpha=0.7, edgecolors='black')
    
    for i, model in enumerate(models):
        ax1.annotate(model, (improvement_rates[i], bleu_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax1.set_xlabel('Training Efficiency (% improvement per epoch)')
    ax1.set_ylabel('BLEU Score')
    ax1.set_title('Training Efficiency vs BLEU Performance', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Multi-metric radar comparison
    metrics = ['BLEU Score', 'Accuracy', 'BERT F1']
    
    # Normalize metrics to 0-1 scale for radar chart
    normalized_data = {}
    for d in efficiency_data:
        normalized_data[d['model']] = [
            d['bleu_score'] / max([x['bleu_score'] for x in efficiency_data]),
            d['accuracy'] / max([x['accuracy'] for x in efficiency_data]),
            d['bert_f1'] / max([x['bert_f1'] for x in efficiency_data])
        ]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    ax2 = plt.subplot(122, projection='polar')
    
    colors_radar = ['blue', 'red']
    for i, (model, values) in enumerate(normalized_data.items()):
        values += values[:1]  # Complete the circle
        ax2.plot(angles, values, 'o-', linewidth=2, label=model, color=colors_radar[i])
        ax2.fill(angles, values, alpha=0.25, color=colors_radar[i])
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(0, 1)
    ax2.set_title('Multi-Metric Performance Radar', fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.suptitle('Training Efficiency and Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Efficiency analysis plot saved to {save_path}")
    
    plt.show()

def plot_learning_curve_analysis(training_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot detailed learning curve analysis with smoothing and trend analysis"""
    if training_df.empty:
        print("No training data for learning curve analysis")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Smoothed learning curves with confidence intervals
    for run_id in training_df['run_id'].unique():
        run_data = training_df[training_df['run_id'] == run_id].sort_values('epoch')
        model_type = 'Without Attention' if 'no_attention' in run_id else 'With Attention'
        
        epochs = run_data['epoch'].values
        losses = run_data['loss'].values
        
        # Apply moving average smoothing
        window_size = max(3, len(losses) // 10)
        smoothed_losses = pd.Series(losses).rolling(window=window_size, center=True).mean()
        
        # Plot original and smoothed curves
        ax1.plot(epochs, losses, alpha=0.3, linewidth=1, 
                label=f"{model_type} (raw)")
        ax1.plot(epochs, smoothed_losses, linewidth=3,
                label=f"{model_type} (smoothed)")
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Smoothed Learning Curves', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Plot 2: Learning rate analysis (epoch-to-epoch loss change)
    for run_id in training_df['run_id'].unique():
        run_data = training_df[training_df['run_id'] == run_id].sort_values('epoch')
        model_type = 'Without Attention' if 'no_attention' in run_id else 'With Attention'
        
        if len(run_data) > 1:
            epochs = run_data['epoch'].values[1:]
            loss_changes = np.diff(run_data['loss'].values)
            
            ax2.plot(epochs, loss_changes, linewidth=2, marker='o', markersize=3,
                    label=f"{model_type}")
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Change (Δ Loss)')
    ax2.set_title('Learning Rate Analysis (Epoch-to-Epoch Loss Change)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve analysis saved to {save_path}")
    
    plt.show()

def main():
    """Main function to generate all plots and reports"""
    print("Loading data...")
    training_df = load_training_logs()
    eval_df = load_evaluation_results()
    
    if training_df.empty and eval_df.empty:
        print("No data found to plot. Please run training and evaluation first.")
        return
    
    # Create plots directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    plots_dir = os.path.join(project_root, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate all plots
    if not training_df.empty:
        plot_training_loss(training_df, 
                          save_path=os.path.join(plots_dir, f"training_loss_{timestamp}.png"))
        plot_training_convergence_analysis(training_df,
                                          save_path=os.path.join(plots_dir, f"convergence_analysis_{timestamp}.png"))
        plot_learning_curve_analysis(training_df,
                                    save_path=os.path.join(plots_dir, f"learning_curves_{timestamp}.png"))
    
    if not eval_df.empty:
        plot_model_comparison(eval_df,
                             save_path=os.path.join(plots_dir, f"model_comparison_{timestamp}.png"))
    
    if not training_df.empty and not eval_df.empty:
        plot_training_and_eval_timeline(training_df, eval_df,
                                       save_path=os.path.join(plots_dir, f"timeline_{timestamp}.png"))
        plot_efficiency_analysis(training_df, eval_df,
                                save_path=os.path.join(plots_dir, f"efficiency_analysis_{timestamp}.png"))
    
    # Generate summary report
    summary = generate_summary_report(training_df, eval_df)
    print("\n" + summary)
    
    # Save summary to file
    summary_path = os.path.join(plots_dir, f"summary_report_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Summary report saved to {summary_path}")

if __name__ == "__main__":
    main()