import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

def plot_results(output_dir, raw_reward_data, periodic_log_data, config, 
                 mean_eval_reward, std_eval_reward, 
                 mean_eval_reward_rand_ris, std_eval_reward_rand_ris,
                 mean_eval_reward_rand_bf, std_eval_reward_rand_bf,
                 mean_random_reward,
                 mean_eval_sumrate=0.0, mean_random_sumrate=0.0
                 ):
    """
    Generates plots using periodic data and raw reward data.
    """
    print(f"Generating plots and summary in: {output_dir}")

    periodic_df = pd.DataFrame(periodic_log_data)
    
    # Layout: 1 row, 3 columns (Reward, Sum-rate, Comparison)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Agent Performance Analysis: {config.get('run_settings', {}).get('model_type', '')}", fontsize=16)

    # Plot 1: Training Reward
    if raw_reward_data:
        reward_df = pd.DataFrame({'reward': raw_reward_data})
        rolling_window = 500 if len(reward_df) > 500 else len(reward_df) // 10
        axs[0].plot(reward_df.index, reward_df['reward'], label="Instantaneous", color='blue', alpha=0.15)
        if rolling_window > 0:
            rolling_avg = reward_df['reward'].rolling(window=rolling_window).mean()
            axs[0].plot(rolling_avg.index, rolling_avg, color='blue', label=f"{rolling_window}-Step Avg")
        axs[0].set(xlabel="Step", ylabel="Reward", title="Training Reward")
    elif not periodic_df.empty:
        axs[0].plot(periodic_df['timestep'], periodic_df['avg_reward_since_last_log'], label="Avg Reward", color='blue')
        axs[0].set(xlabel="Step", ylabel="Reward", title="Training Reward")
    else:
        axs[0].text(0.5, 0.5, "No Reward Data", ha='center', va='center')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # Plot 2: Sum-rate
    if not periodic_df.empty and 'recent_avg_sumrate' in periodic_df.columns:
        axs[1].plot(periodic_df['timestep'], periodic_df['recent_avg_sumrate'], label="Avg Sum-rate", color='green')
        axs[1].set(xlabel="Step", ylabel="Sum-rate (bps/Hz)", title="Training Sum-rate")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()
    else:
        axs[1].text(0.5, 0.5, "No Sum-rate Data", ha='center', va='center')

    # Plot 3: Performance Comparison
    # Calculate learning agent average from the last 10% of logs to get a stable "converged" value
    if not periodic_df.empty and 'recent_avg_sumrate' in periodic_df.columns:
        tail_idx = max(1, int(len(periodic_df) * 0.9))
        mean_learning_sumrate = periodic_df['recent_avg_sumrate'].iloc[tail_idx:].mean()
    else:
        mean_learning_sumrate = 0.0

    agents = ['Random', 'Trained (Eval)', 'Trained (Last 10%)']
    sumrates = [mean_random_sumrate, mean_eval_sumrate, mean_learning_sumrate]
    colors = ['gray', 'green', 'blue']
    
    bars = axs[2].bar(agents, sumrates, color=colors)
    axs[2].set(ylabel="Mean Sum-rate (bps/Hz)", title="Benchmark Comparison")
    axs[2].bar_label(bars, fmt='%.2f')
    axs[2].grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "training_plots.png"))
    plt.close(fig)

    summary_data = {
        "final_evaluation_metrics": {
            "trained_agent": { 
                "mean_reward": float(mean_eval_reward), 
                "std_reward": float(std_eval_reward), 
                "mean_sumrate": float(mean_eval_sumrate),
            },
            "diagnostics": {
                "trained_bf_random_ris": { "mean_reward": float(mean_eval_reward_rand_ris), "std_reward": float(std_eval_reward_rand_ris) },
                "random_bf_trained_ris": { "mean_reward": float(mean_eval_reward_rand_bf), "std_reward": float(std_eval_reward_rand_bf) }
            },
            "baseline": { 
                "random_agent": { 
                    "mean_reward": float(mean_random_reward), 
                    "mean_sumrate": float(mean_random_sumrate),
                } 
            }
        },
        "periodic_training_logs": periodic_log_data
    }
    
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
        
    print(f"Successfully saved plots and summary.json to {output_dir}")