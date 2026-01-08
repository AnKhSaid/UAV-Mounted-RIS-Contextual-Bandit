import matplotlib
matplotlib.use("Agg")
import torch
import random
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import gymnasium as gym
from tqdm import tqdm
import time
import json
from collections import deque
import scipy.stats as st
import itertools

import env_registration
from my_env import UAVRISEnv
from classical_optimizers import AO_WMMSE_Eig, AO_WMMSE_Eig_SAA
from model_builder import build_model
from Plotting import plot_results 

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit

import subprocess
import sys
import argparse
from filelock import FileLock
import multiprocessing as mp
import csv
import matplotlib as mpl
from cycler import cycler

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.constrained_layout.use": True,
})
mpl.rcParams["axes.prop_cycle"] = cycler(
    color=[*mpl.colormaps["tab10"].colors]
) + cycler(linestyle=["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"])

def fig_size(column="single", ratio=0.62):
    w = 3.5 if column=="single" else 7.16
    return (w, w*ratio)

class UnifiedLoggingCallback(BaseCallback):
    def __init__(self, total_timesteps: int, n_envs: int, enable_logging: bool = False):
        super().__init__()
        self.enable_logging = enable_logging
        self.log_freq = total_timesteps // 100 if total_timesteps >= 100 else 1
        self.last_log_timestep = 0
        self.periodic_log_data = []
        self.raw_reward_data = [] if self.enable_logging else None
        self.interval_reward_buffer = []
        deque_size = 10 * n_envs
        self.recent_sumrate_buffer = deque(maxlen=deque_size)

    def _on_step(self) -> bool:
        if not self.enable_logging:
            return True
            
        rewards = self.locals.get('rewards')
        infos = self.locals.get('infos')
        
        self.interval_reward_buffer.extend(rewards)
        if self.raw_reward_data is not None:
            self.raw_reward_data.extend(rewards)
            
        for info in infos:
            self.recent_sumrate_buffer.append(info.get('sumrate', 0.0))
            
        if self.num_timesteps >= self.last_log_timestep + self.log_freq or \
           self.num_timesteps == self.training_env.num_envs:
            self.last_log_timestep = self.num_timesteps
            
            log_entry = {
                "timestep": self.num_timesteps,
                "avg_reward_since_last_log": float(np.mean(self.interval_reward_buffer)) if self.interval_reward_buffer else 0.0,
                "recent_avg_sumrate": float(np.mean(self.recent_sumrate_buffer)) if self.recent_sumrate_buffer else 0.0,
            }
            
            self.periodic_log_data.append(log_entry)
            self.interval_reward_buffer.clear()
            
        return True

def run_custom_evaluation(model, env, n_episodes, random_ris=False, random_bf=False, eval_seed_offset=0):
    safety_layer = model.policy.actor.safety_layer
    device = safety_layer.p_max.device

    episode_rewards, episode_avg_sumrates = [], []
    bf_params_count = 2 * env.get_attr('M')[0] * env.get_attr('K')[0]
    
    for i in range(n_episodes):
        env.seed(eval_seed_offset + i) 
        obs, done = env.reset(), np.array([False])
        total_reward, total_sumrate, num_steps = 0.0, 0.0, 0
        
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            
            if random_ris:
                ris_dims = action.shape[1] - bf_params_count
                action[0, bf_params_count:] = np.random.uniform(-1.0, 1.0, size=ris_dims)
            if random_bf:
                action[0, :bf_params_count] = np.random.uniform(-1.0, 1.0, size=bf_params_count)
            
            if random_ris or random_bf:
                action_tensor = torch.as_tensor(action, device=device, dtype=torch.float32)
                with torch.no_grad():
                    action_tensor = safety_layer(action_tensor)
                action = action_tensor.cpu().numpy()
                
            obs, rewards, dones, infos = env.step(action)
            done = dones
            total_reward += rewards[0]
            total_sumrate += infos[0].get('sumrate', 0.0)
            num_steps += 1
            
        episode_rewards.append(total_reward)
        if num_steps > 0:
            episode_avg_sumrates.append(total_sumrate / num_steps)
            
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_sumrate = np.mean(episode_avg_sumrates) if episode_avg_sumrates else 0.0
        
    return mean_reward, std_reward, mean_sumrate

def run_drl_experiment(config, seed: int):
    run_cfg = config['run_settings']
    train_cfg = config['training_params']
    test_cfg = config['testing_params']
    env_cfg = config['env_parameters']
    model_type = run_cfg['model_type']
    
    model_hyperparams = copy.deepcopy(config['model_hyperparameters'].get('shared', {}))
    model_type_upper = model_type.upper()
    if model_type_upper == "TD3":
        specific_hp = config['model_hyperparameters'].get('TD3_shared', {})
    else:
        specific_hp = config['model_hyperparameters'].get(model_type_upper, {})
    for key, value in specific_hp.items():
        if key in model_hyperparams and isinstance(model_hyperparams[key], dict) and isinstance(value, dict):
            model_hyperparams[key].update(value)
        else:
            model_hyperparams[key] = value
            
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    total_timesteps = train_cfg['target_episodes_per_env'] * train_cfg['steps_per_episode'] * train_cfg['n_envs']
    
    rho = env_cfg.get('cascaded_error_rho', 1.0)
    jitter_deg = env_cfg.get('sigma_jitt_deg', 0.0)
    run_name = f"{model_type}_Rho{rho:.2f}_Jit{jitter_deg:.1f}_Seed{seed}_{int(time.time())}"
    run_dir = f"./runs_figures/{run_name}"
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    with open(os.path.join(run_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    def create_env(is_eval=False):
        current_env_cfg = copy.deepcopy(env_cfg)
        if is_eval:
            current_env_cfg['S_samples'] = 1
        env = gym.make("UAV-RIS-v0", **current_env_cfg)
        max_steps = test_cfg['test_steps'] if is_eval else train_cfg['steps_per_episode']
        env = TimeLimit(env, max_episode_steps=max_steps)
        return env

    train_env = make_vec_env(
        lambda: create_env(),
        n_envs=train_cfg['n_envs'],
        seed=seed,
        monitor_dir=logs_dir
    )
    
    monitor_callback = UnifiedLoggingCallback(
        total_timesteps,
        n_envs=train_cfg['n_envs'],
        enable_logging=run_cfg['enable_logging']
    )
    
    model = build_model(
        model_type,
        train_env,
        seed,
        log_dir=os.path.join(run_dir, "tb_logs"),
        device=run_cfg['device'],
        verbose=0,
        model_hyperparams=model_hyperparams,
        env_params=env_cfg
    )
    
    print(f"Starting training for {total_timesteps} total steps...")
    
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=monitor_callback,
        log_interval=4
    )
    
    model.save(os.path.join(run_dir, "trained_model.zip"))
    train_env.close()

    print("\n" + "="*20 + " EVALUATION " + "="*20)
    eval_env = make_vec_env(lambda: create_env(is_eval=True), n_envs=1)
    
    eval_seed = run_cfg.get('eval_seed', 42)
    
    print("\n--- Evaluating: Trained Agent ---")
    mean_reward, std_reward, mean_sumrate = run_custom_evaluation(
        model, eval_env, test_cfg['test_episodes'], eval_seed_offset=eval_seed + 1000
    )
    
    print("\n--- Evaluating: Trained BF, Random RIS ---")
    mean_reward_rand_ris, std_reward_rand_ris, mean_sumrate_rand_ris = run_custom_evaluation(
        model, eval_env, test_cfg['test_episodes'], random_ris=True, eval_seed_offset=eval_seed + 2000
    )

    print("\n--- Evaluating: Random BF, Trained RIS ---")
    mean_reward_rand_bf, std_reward_rand_bf, mean_sumrate_rand_bf = run_custom_evaluation(
        model, eval_env, test_cfg['test_episodes'], random_bf=True, eval_seed_offset=eval_seed + 3000
    )
    
    eval_env.close()
    
    print("\n--- Evaluating: Fully Random Agent ---")
    random_env = make_vec_env(lambda: create_env(is_eval=True), n_envs=1)
    random_rewards, random_avg_sumrates = [], []
    
    safety_layer = model.policy.actor.safety_layer
    device = safety_layer.p_max.device

    for i in range(test_cfg['test_episodes']):
        random_env.seed(eval_seed + i)
        obs = random_env.reset()
        done = np.array([False])
        
        total_reward, total_sumrate, num_steps = 0.0, 0.0, 0
        
        while not done[0]:
            raw_action_1d_np = random_env.action_space.sample()
            raw_action_np = raw_action_1d_np[None, :]
            raw_action_tensor = torch.as_tensor(raw_action_np, device=device, dtype=torch.float32)
            with torch.no_grad():
                projected_action_tensor = safety_layer(raw_action_tensor)
            action = projected_action_tensor.cpu().numpy()
            obs, rewards, dones, infos = random_env.step(action)
            
            done = dones
            total_reward += rewards[0]
            total_sumrate += infos[0].get('sumrate', 0.0)
            
            num_steps += 1
            
        random_rewards.append(total_reward)
        if num_steps > 0:
            random_avg_sumrates.append(total_sumrate / num_steps)
            
    random_env.close()
    
    mean_random_reward = np.mean(random_rewards)
    mean_random_sumrate = np.mean(random_avg_sumrates) if random_avg_sumrates else 0.0
    
    print(f"Result (Reward): {mean_random_reward:.4f}")
    print(f"Result (Sum-rate): {mean_random_sumrate:.4f}")

    print("\n" + "="*20 + " PLOTTING " + "="*20)
    plot_results(
        output_dir=run_dir,
        raw_reward_data=monitor_callback.raw_reward_data,
        periodic_log_data=monitor_callback.periodic_log_data,
        config=config,
        mean_eval_reward=mean_reward,
        std_eval_reward=std_reward,
        mean_eval_reward_rand_ris=mean_reward_rand_ris,
        std_eval_reward_rand_ris=std_reward_rand_ris,
        mean_eval_reward_rand_bf=mean_reward_rand_bf,
        std_eval_reward_rand_bf=std_reward_rand_bf,
        mean_random_reward=mean_random_reward,
        mean_eval_sumrate=mean_sumrate,
        mean_random_sumrate=mean_random_sumrate,
    )
    
    results_dict = {
        "evaluated_agent_mean_sumrate": mean_sumrate,
        "evaluated_agent_mean_reward": mean_reward,
        "trained_bf_random_ris_mean_sumrate": mean_sumrate_rand_ris,
        "random_bf_trained_ris_mean_sumrate": mean_sumrate_rand_bf,
    }
    
    return results_dict, run_dir, seed 

def safe_to_numeric(index):
    try:
        return pd.to_numeric(index)
    except ValueError:
        return index

def run_optimizer_benchmark(config: dict, method_name: str, seed: int) -> dict:
    bench_cfg = config['benchmark_settings']
    env_cfg = config['env_parameters']
    run_cfg = config.get('run_settings', {})
    master_rng = np.random.default_rng(seed)
    verbose = bench_cfg.get('verbose', False)

    common_params = {
        "wmmse_bisection_steps": bench_cfg.get("WMMSE_EIG_Params", {}).get("wmmse_bisection_steps", 30),
        "wmmse_bisection_lam_high": bench_cfg.get("WMMSE_EIG_Params", {}).get("wmmse_bisection_lam_high", 1e6)
    }
    method_params = dict(bench_cfg.get("WMMSE_EIG_Params", {}))
    method_params.update(common_params)
    if "SAA_Params" in bench_cfg:
        method_params["SAA_Params"] = dict(bench_cfg.get("SAA_Params", {}))

    env_cfg_eval = copy.deepcopy(env_cfg)
    if env_cfg.get('cascaded_error_rho', 1.0) == 1.0 and env_cfg.get('sigma_jitt_deg', 0.0) == 0.0:
        env_cfg_eval['S_samples'] = 1
    else:
        env_cfg_eval['S_samples'] = env_cfg.get('S_samples', 1)

    env = gym.make("UAV-RIS-v0", **env_cfg_eval)
    
    all_sum_rates = []
    num_episodes = bench_cfg['num_episodes']

    episode_seeds = master_rng.integers(low=0, high=2**31, size=num_episodes)

    for i in tqdm(range(num_episodes), desc=f"Running {method_name} (Seed {seed})"):
        episode_seed = int(episode_seeds[i])
        
        env.reset(seed=episode_seed)
        u = env.unwrapped 
        
        H_1_Ric = u.k_term_los_1 * u.h1_los_est + u.k_term_nlos_1 * u.h1_nlos_est
        H_2_Ric = u.k_term_los_2 * u.h2_los_est + u.k_term_nlos_2 * u.h2_nlos_est
        H_1_est = u.sqrt_path_loss_1 * H_1_Ric
        H_2_est = u.sqrt_path_loss_2_vec[np.newaxis, :] * H_2_Ric
        sigma2  = float(u.awgn_power_watts)
        
        try:
            if method_name == "WMMSE_EIG_SAA":
                opt = AO_WMMSE_Eig_SAA(env_cfg, method_params, env_u=u, verbose=verbose)
            else:
                opt = AO_WMMSE_Eig(env_cfg, method_params, verbose=verbose)

            G, phi_vec = opt.optimize(H_1_est, H_2_est, sigma2, seed=int(episode_seed))
            
            results = env.unwrapped._calculate_robust_reward(G, phi_vec)
            all_sum_rates.append(results["avg_sum_rate"])
            
        except Exception as e:
            print(f"Warning: Optimizer failed for episode {episode_seed}: {e}")
            continue
            
    env.close()
    mean_sum_rate = np.mean(all_sum_rates) if all_sum_rates else 0.0
    
    return {
        "evaluated_agent_mean_sumrate": mean_sum_rate,
        "trained_bf_random_ris_mean_sumrate": 0.0,
        "seed": seed
    }

def calculate_ci_95(data):
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return 0.0
    
    n = len(data)
    mean = np.mean(data)
    sem = st.sem(data)
    if sem == 0:
        return 0.0
        
    t_val = st.t.ppf(1 - 0.025, df=n-1)
    
    return sem * t_val

def plot_robustness_results(raw_df, output_filename, title, xlabel, model_types, invert_xaxis=False, xticklabels=None):
    print(f"\nGenerating plot: {output_filename}")
    
    raw_df_copy = raw_df.copy()
    
    raw_df_copy.rename(columns={'Rho': 'Rho (CSI Correlation)'}, inplace=True)
    
    grouped_main = raw_df_copy.groupby(['Rho (CSI Correlation)', 'Algorithm'])['Mean Sum-Rate'].agg(
        mean='mean', 
        std='std', 
        count='count'
    ).reset_index()
    
    grouped_main['ci_95'] = grouped_main.apply(
        lambda row: calculate_ci_95(raw_df_copy[
            (raw_df_copy['Rho (CSI Correlation)'] == row['Rho (CSI Correlation)']) & 
            (raw_df_copy['Algorithm'] == row['Algorithm'])
        ]['Mean Sum-Rate']), 
        axis=1
    )
    
    grouped_bf_only = raw_df_copy[raw_df_copy['Mean Sum-Rate (BF-Only)'] > 0].groupby(['Rho (CSI Correlation)', 'Algorithm'])['Mean Sum-Rate (BF-Only)'].agg(
        mean='mean', 
        std='std', 
        count='count'
    ).reset_index()
    
    if not grouped_bf_only.empty:
        grouped_bf_only['ci_95'] = grouped_bf_only.apply(
            lambda row: calculate_ci_95(raw_df_copy[
                (raw_df_copy['Rho (CSI Correlation)'] == row['Rho (CSI Correlation)']) & 
                (raw_df_copy['Algorithm'] == row['Algorithm'])
            ]['Mean Sum-Rate (BF-Only)']), 
            axis=1
        )
        grouped_bf_only['Algorithm'] = grouped_bf_only['Algorithm'] + ' (BF-Only)'
    
    results_df = pd.concat([grouped_main, grouped_bf_only], ignore_index=True)

    try:
        results_df['Rho (CSI Correlation)'] = pd.to_numeric(results_df['Rho (CSI Correlation)'])
        results_df = results_df.sort_values(by='Rho (CSI Correlation)', ascending=not invert_xaxis)
    except Exception:
        pass

    print(results_df.to_string())
    
    fig, ax = plt.subplots(figsize=fig_size("single"))
    
    plot_x_values = results_df['Rho (CSI Correlation)'].unique()

    color_map = {
        "TD3": "C0",
        "DDPG": "C1",
        "AO-WMMSE": "C2",
        "AO-WMMSE-SAA": "C3",
        "TD3 (BF-Only)": "C4",
        "DDPG (BF-Only)": "C5"
    }
    marker_map = {
        "TD3": "o",
        "DDPG": "s",
        "AO-WMMSE": "^",
        "AO-WMMSE-SAA": "d",
        "TD3 (BF-Only)": "x",
        "DDPG (BF-Only)": "+"
    }

    for mt in model_types:
        model_data = results_df[results_df['Algorithm'] == mt]
        if not model_data.empty:
            color = color_map.get(mt, "C" + str(len(color_map) % 10))
            marker = marker_map.get(mt, "v")
            
            if mt in ["AO-WMMSE", "AO-WMMSE-SAA"]:
                y_error = None
            else:
                y_error = model_data['ci_95']
            
            ax.errorbar(
                x=model_data['Rho (CSI Correlation)'],
                y=model_data['mean'], 
                yerr=y_error,
                label=mt, 
                color=color,
                marker=marker,
                linestyle='None',
                capsize=3,
                elinewidth=0.8,
                markeredgewidth=0.8
            )
        else:
            print(f"Warning: No valid data for {mt}. Skipping line in plot.")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean Sum-Rate (bps/Hz)")
    
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.margins(x=0.03)
    
    if xticklabels:
        ax.set_xticks(ticks=plot_x_values, labels=xticklabels, rotation=0)
    elif invert_xaxis:
        ticks = sorted(results_df['Rho (CSI Correlation)'].unique(), reverse=True)
        ax.set_xticks(ticks=ticks)
        ax.invert_xaxis()
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
                loc="outside upper center",
                ncol=3,
                frameon=False,
                handlelength=2.0, handletextpad=0.6, columnspacing=1.0)
    
    pdf_filename = output_filename.replace(".png", ".pdf")
    plt.savefig(pdf_filename, bbox_inches="tight")
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"Successfully saved plot: {output_filename} and {pdf_filename}")
    plt.close(fig)

def plot_convergence_for_level(rho: float, jitter: float, base_config: dict, plot_name_suffix: str, results_dir: str, seeds: list):
    print(f"\n--- Generating Convergence Plot for: {plot_name_suffix} (Rho={rho:.2f}, Jitter={jitter:.1f}) ---")

    drl_models_to_plot = ["TD3", "DDPG"]
    base_run_dir = "./runs_figures"
    n_envs = base_config.get('training_params', {}).get('n_envs', 1)
    
    all_seeds_data = {model: [] for model in drl_models_to_plot}
    
    for model in drl_models_to_plot:
        for seed in seeds:
            patterns = [
                f"{model}_Rho{rho:.2f}_Jit{jitter:.1f}_Seed{seed}_",
            ]
            all_runs_in_basedir = [os.path.join(base_run_dir, d) for d in os.listdir(base_run_dir) if os.path.isdir(os.path.join(base_run_dir, d))]
            relevant_runs = [d for d in all_runs_in_basedir if any(os.path.basename(d).startswith(p) for p in patterns)]

            if relevant_runs:
                latest_run_dir = max(relevant_runs, key=os.path.getmtime)
                logs_subdir = os.path.join(latest_run_dir, "logs")
                monitor_path = os.path.join(logs_subdir, "0.monitor.csv")
                
                if os.path.exists(monitor_path):
                    try:
                        df = pd.read_csv(monitor_path, skiprows=1)
                        if not df.empty and 'r' in df.columns and 'l' in df.columns and 't' in df.columns:
                            df['timestep'] = df['l'].cumsum()
                            df['r'] = pd.to_numeric(df['r'], errors='coerce')
                            all_seeds_data[model].append(df)
                        else:
                            print(f"Warning: Monitor file {monitor_path} is empty or missing columns.")
                    except Exception as e:
                        print(f"Warning: Could not load/process {monitor_path}: {e}")
                else:
                    print(f"Warning: No monitor file found at {monitor_path}")
            else:
                print(f"Warning: No run directory found for {model} @ {plot_name_suffix} (seed={seed}).")
    
    print(f"Found convergence data for {len(all_seeds_data['TD3'])} TD3 seeds and {len(all_seeds_data['DDPG'])} DDPG seeds.")

    fig, ax = plt.subplots(figsize=fig_size("single"))
    
    total_steps = base_config['training_params']['target_episodes_per_env'] * base_config['training_params']['steps_per_episode'] * base_config['training_params']['n_envs']
    max_timestep_plot = min(total_steps, 200000)
    plot_has_drl_data = False
    
    color_map = {"TD3": "C0", "DDPG": "C1"}
    linestyle_map = {"TD3": "-", "DDPG": "--"}
    
    common_timesteps = np.linspace(0, max_timestep_plot, 500)
    
    for model_type, seed_dfs in all_seeds_data.items():
        if not seed_dfs:
            continue
            
        plot_has_drl_data = True
        
        interpolated_raw_dfs = []
        
        for df in seed_dfs:
            df_filtered = df[df['timestep'] <= max_timestep_plot].copy()
            if df_filtered.empty:
                continue
                
            interp_func = np.interp(
                common_timesteps, 
                df_filtered['timestep'], 
                df_filtered['r'],
                left=df_filtered['r'].iloc[0], 
                right=df_filtered['r'].iloc[-1]
            )
            interpolated_raw_dfs.append(interp_func)

        if not interpolated_raw_dfs:
            continue

        stacked_data = np.stack(interpolated_raw_dfs, axis=0)
        
        raw_mean_curve = np.nanmean(stacked_data, axis=0)
        n_seeds_per_step = np.sum(~np.isnan(stacked_data), axis=0)
        std_curve = np.nanstd(stacked_data, axis=0)
        
        ci_95_curve = np.zeros_like(raw_mean_curve)
        df_t = np.maximum(1, n_seeds_per_step - 1)
        t_vals = st.t.ppf(1 - 0.025, df=df_t)
        
        safe_sqrt_n = np.sqrt(n_seeds_per_step, where=n_seeds_per_step > 0, out=np.full_like(n_seeds_per_step, 1.0, dtype=float))
        sem = std_curve / safe_sqrt_n
        sem[n_seeds_per_step == 0] = 0.0 
        
        ci_95_curve = sem * t_vals
        ci_95_curve = np.nan_to_num(ci_95_curve, 0.0)

        window_size_interp = int( (2000 / max_timestep_plot) * 500 )
        window_size_interp = max(1, window_size_interp)
        
        smoothed_mean_curve = pd.Series(raw_mean_curve).rolling(window=window_size_interp, min_periods=1).mean()

        color = color_map.get(model_type, "C2")
        linestyle = linestyle_map.get(model_type, ":")
        
        x_axis_data = common_timesteps / 1000.0
        
        ax.plot(x_axis_data, smoothed_mean_curve,
                label=f"{model_type} (Mean)",
                color=color, 
                linestyle=linestyle)
        
        ax.fill_between(
            x_axis_data, 
            raw_mean_curve - ci_95_curve,
            raw_mean_curve + ci_95_curve, 
            color=color, 
            alpha=0.15, 
            label=f"{model_type} (95% CI Band)"
        )
                
    if not plot_has_drl_data:
        print(f"Warning: No DRL data found to plot for {plot_name_suffix}.")
        ax.text(0.5, 0.5, "No DRL data available.", ha='center', va='center', transform=ax.transAxes)
        
    ax.set_title(rf"($\rho={rho:.2f}, \sigma_j={jitter:.1f}^\circ$) Convergence")
    ax.set_xlabel("Training Timesteps ($10^3$)")
    ax.set_ylabel("Sum-Rate (bps/Hz)")
    
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    for h, l in zip(handles, labels):
        if "95% CI Band" not in l:
            unique_labels[l] = h
             
    ax.legend(unique_labels.values(), unique_labels.keys(), frameon=False, loc='lower right', fontsize=6)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.set_xlim(0, max_timestep_plot / 1000.0)
    ax.margins(x=0.03)
    
    filename_stem = f"paper_figure_training_{plot_name_suffix.replace('.', '_')}"
    
    pdf_path = os.path.join(results_dir, f"{filename_stem}.pdf")
    png_path = os.path.join(results_dir, f"{filename_stem}.png")
    
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")
    print(f"Successfully saved plot: {png_path} and {pdf_path}")
    plt.close(fig)

def main():
    config_path = 'config.yaml'
    try:
        with open(config_path, 'r') as f:
            master_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config not found at {config_path}")
        return

    model_types_to_run = ["TD3", "DDPG", "AO-WMMSE", "AO-WMMSE-SAA"]
    model_types_to_plot = ["TD3", "DDPG", "AO-WMMSE", "AO-WMMSE-SAA", "TD3 (BF-Only)", "DDPG (BF-Only)"]
    
    RESULTS_DIR = "./paper_figures_rho"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("./runs_figures/logs", exist_ok=True)
    
    raw_results_csv_path = os.path.join(RESULTS_DIR, "rho_robustness_raw.csv")
    
    SEEDS_TO_RUN = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90] 
    
    print(f"--- Running {len(SEEDS_TO_RUN)} seeds per experiment: {SEEDS_TO_RUN} ---")

    print("\n" + "="*20 + " BUILDING JOB LIST " + "="*20)
    rho_levels = np.arange(1.0, 0.49, -0.1) 
    
    completed_jobs = set()
    if os.path.exists(raw_results_csv_path):
        try:
            completed_df = pd.read_csv(raw_results_csv_path)
            for _, row in completed_df.iterrows():
                completed_jobs.add((row['Algorithm'], float(row['Rho']), int(row['Seed'])))
            print(f"Loaded {len(completed_jobs)} completed job-seeds from cache.")
        except Exception as e:
            print(f"Warning: Could not read {raw_results_csv_path}. Re-running all jobs. Error: {e}")
            
    drl_jobs = []
    cpu_jobs = []
    
    for rho in rho_levels:
        for model in model_types_to_run:
            for seed in SEEDS_TO_RUN:
                job_tuple = (model, float(rho), int(seed))
                
                if job_tuple in completed_jobs:
                    print(f"Skipping (cached): {model} @ Rho {rho:.2f} (Seed {seed})")
                else:
                    print(f"Adding to queue:   {model} @ Rho {rho:.2f} (Seed {seed})")
                    job_info = (model, float(rho), int(seed)) 
                    if model in ["TD3", "DDPG"]:
                        drl_jobs.append(job_info)
                    else:
                        cpu_jobs.append(job_info)

    total_jobs = len(drl_jobs) + len(cpu_jobs)
    if not total_jobs:
        print("No new experiments to run. All results are cached.")
    else:
        print(f"\n" + "="*20 + f" RUNNING {total_jobs} EXPERIMENTS WITH POPEN " + "="*20)
        
        MAX_DRL_RUNS = 5 
        MAX_CPU_RUNS = 5 
        print(f"Will run up to {MAX_DRL_RUNS} DRL (GPU) jobs and {MAX_CPU_RUNS} CPU jobs in parallel.")

        drl_processes = []
        cpu_processes = []
        
        pbar = tqdm(total=total_jobs, desc="Total Experiments")
        
        while drl_jobs or cpu_jobs or drl_processes or cpu_processes:
            
            for p_list in [drl_processes, cpu_processes]:
                for i in reversed(range(len(p_list))):
                    p = p_list[i]
                    if p.poll() is not None: 
                        p_list.pop(i)
                        pbar.update(1)
            
            while len(drl_processes) < MAX_DRL_RUNS and drl_jobs:
                model, rho, seed = drl_jobs.pop(0)
                cmd = [sys.executable, __file__, "--model", model, "--rho", str(rho), "--seed", str(seed)]
                p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                drl_processes.append(p)

            while len(cpu_processes) < MAX_CPU_RUNS and cpu_jobs:
                model, rho, seed = cpu_jobs.pop(0)
                cmd = [sys.executable, __file__, "--model", model, "--rho", str(rho), "--seed", str(seed)]
                p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                cpu_processes.append(p)
            
            time.sleep(1.0)
            
        pbar.close()

    print("\n" + "="*20 + " PLOTTING RESULTS " + "="*20)
    
    if not os.path.exists(raw_results_csv_path):
        print(f"Error: Raw results file not found: {raw_results_csv_path}. Cannot plot.")
        return
        
    try:
        raw_df_final = pd.read_csv(raw_results_csv_path)
    except Exception as e:
        print(f"Error loading {raw_results_csv_path}: {e}. Cannot plot.")
        return
    
    plot_robustness_results(
        raw_df_final,
        os.path.join(RESULTS_DIR, "paper_figure_rho_robustness.png"),
        title="Performance vs. CSI Quality",
        xlabel=r"CSI Quality Parameter, $\rho$",
        model_types=model_types_to_plot,
        invert_xaxis=True
    )

    print("\n--- Generating All Convergence Plots ---")
    for rho in rho_levels:
        plot_convergence_for_level(
            rho=float(rho), 
            jitter=0.0,
            base_config=master_config, 
            plot_name_suffix=f"rho_{rho:.2f}".replace('.', '_'),
            results_dir=RESULTS_DIR,
            seeds=SEEDS_TO_RUN
        )

    print("\n" + "="*20 + " ALL EXPERIMENTS FINISHED " + "="*20)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Run a single experiment worker.")
    parser.add_argument("--model", type=str, help="Model type to run (e.g., TD3, AO-WMMSE)")
    parser.add_argument("--rho", type=float, help="Rho value to run")
    parser.add_argument("--seed", type=int, help="Seed value to run")
    args = parser.parse_args()

    if args.model and args.rho is not None and args.seed is not None:
        
        model = args.model
        rho = args.rho
        seed = args.seed
        
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        config_path = 'config.yaml'
        try:
            with open(config_path, 'r') as f:
                master_config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"[Worker Error] Config not found at {config_path}")
            sys.exit(1)
            
        run_config = copy.deepcopy(master_config)
        run_config['env_parameters']['sigma_jitt_deg'] = 0.0
        run_config['env_parameters']['cascaded_error_rho'] = float(rho)
        if float(rho) == 1.0:
             run_config['env_parameters']['S_samples'] = 1
        
        try:
            eval_results = {}
            if model == "AO-WMMSE" or model == "AO-WMMSE-SAA":
                method_name_config = "WMMSE_EIG" if model == "AO-WMMSE" else "WMMSE_EIG_SAA"
                run_config['benchmark_settings']['method'] = method_name_config
                eval_results = run_optimizer_benchmark(run_config, method_name_config, seed)
                
            elif model in ["TD3", "DDPG"]:
                run_config['run_settings']['model_type'] = model
                (eval_results, _run_dir, 
                 _seed) = run_drl_experiment(run_config, seed)
            
            RESULTS_DIR = "./paper_figures_rho"
            raw_results_csv_path = os.path.join(RESULTS_DIR, "rho_robustness_raw.csv")
            lock_path = raw_results_csv_path + ".lock"

            with FileLock(lock_path):
                file_exists = os.path.isfile(raw_results_csv_path)
                with open(raw_results_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['Algorithm', 'Rho', 'Seed', 'Mean Sum-Rate', 'Mean Sum-Rate (BF-Only)'])
                    
                    mean_sumrate = eval_results.get("evaluated_agent_mean_sumrate", 0.0)
                    mean_sumrate_bf_only = 0.0
                    if model in ["TD3", "DDPG"]:
                        mean_sumrate_bf_only = eval_results.get("trained_bf_random_ris_mean_sumrate", 0.0)
                    
                    writer.writerow([model, rho, seed, mean_sumrate, mean_sumrate_bf_only])
            
        except Exception as e:
            with open("figure_rho_worker_error.log", "a") as f:
                f.write(f"!!!!!!!! FAILED: {model} @ Rho {rho:.2f} (Seed {seed}). Error: {e} !!!!!!!!\n")
                import traceback
                traceback.print_exc(file=f)
            sys.exit(1)
       
    else:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        
        main()
