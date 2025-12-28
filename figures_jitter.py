# --- Set Matplotlib backend ---
import matplotlib
matplotlib.use("Agg")
import torch
import random
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import copy
import gymnasium as gym
from tqdm import tqdm
import time
import json
from collections import deque
import scipy.stats as st # <-- NEW: For CI calculations
import itertools # <-- NEW: For plot styling

# --- Import from your existing project files ---
# (Ensure these files exist in the same directory)
import env_registration
from my_env import UAVRISEnv
from classical_optimizers import AO_WMMSE_Eig, AO_WMMSE_Eig_SAA
from model_builder import build_model
from Plotting import plot_results 
# --- End project file imports ---

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit
# from stable_baselines3.common.logger import configure as configure_logger # <-- REMOVED

# --- IMPORTS FOR POPEN ---
import subprocess   # For Popen
import sys          # To get python executable path
import argparse     # To parse args in worker mode
from filelock import FileLock # For safe CSV writing
import multiprocessing as mp  # For set_start_method
import csv          # For writing complexity CSV

# ===================================================================
# Paper-Ready Plotting Style
# ===================================================================
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
    "lines.linewidth": 1.2, # Made lines slightly thicker
    "lines.markersize": 3.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.constrained_layout.use": True,
})
# Set cycler to use colors + continuous lines
mpl.rcParams["axes.prop_cycle"] = cycler(
    color=[*mpl.colormaps["tab10"].colors]
) + cycler(linestyle=["-", "-", "-", "-", "-", "-", "-", "-", "-", "-"])

def fig_size(column="single", ratio=0.62):
    w = 3.5 if column=="single" else 7.16
    return (w, w*ratio)
# ===================================================================
# End Plotting Style
# ===================================================================


# ===================================================================
# DRL Experiment Runner (copied from main2.py and modified for timing)
# ===================================================================

# --- UnifiedLoggingCallback from main2.py ---
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
        self.recent_sinr_buffer = deque(maxlen=deque_size)
        self.recent_jains_fairness_buffer = deque(maxlen=deque_size)
        self.recent_rates_per_user_buffer = deque(maxlen=deque_size) 
        self.recent_actor_loss_buffer = deque(maxlen=10)
        self.recent_critic_loss_buffer = deque(maxlen=10)
        self.recent_raw_mu_bf_norm_buffer = deque(maxlen=10)
        self.recent_raw_mu_ris_norm_buffer = deque(maxlen=10)

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
            self.recent_sinr_buffer.append(info.get('total_sinr', 0.0))
            self.recent_jains_fairness_buffer.append(info.get('jains_fairness', 0.0))
            rates = info.get('rates_per_user')
            if rates is not None:
                self.recent_rates_per_user_buffer.append(rates)
            
        loss_dict = self.model.logger.name_to_value
        if 'train/actor_loss' in loss_dict:
            self.recent_actor_loss_buffer.append(loss_dict['train/actor_loss'])
            self.recent_critic_loss_buffer.append(loss_dict['train/critic_loss'])
            if 'train/mean_raw_mu_bf_norm' in loss_dict:
                self.recent_raw_mu_bf_norm_buffer.append(loss_dict['train/mean_raw_mu_bf_norm'])
            if 'train/mean_raw_mu_ris_norm' in loss_dict:
                self.recent_raw_mu_ris_norm_buffer.append(loss_dict['train/mean_raw_mu_ris_norm'])
            
        if self.num_timesteps >= self.last_log_timestep + self.log_freq or \
           self.num_timesteps == self.training_env.num_envs:
            self.last_log_timestep = self.num_timesteps
            
            current_rates_per_user_dict = {}
            if infos:
                current_rates = infos[0].get('rates_per_user')
                if current_rates is not None:
                    current_rates_per_user_dict = {f"user_{i}": float(rate) for i, rate in enumerate(current_rates)}

            log_entry = {
                "timestep": self.num_timesteps,
                "avg_reward_since_last_log": float(np.mean(self.interval_reward_buffer)) if self.interval_reward_buffer else 0.0,
                "recent_avg_sumrate": float(np.mean(self.recent_sumrate_buffer)) if self.recent_sumrate_buffer else 0.0,
                "recent_avg_jains_fairness": float(np.mean(self.recent_jains_fairness_buffer)) if self.recent_jains_fairness_buffer else 0.0,
                "recent_avg_total_sinr": float(np.mean(self.recent_sinr_buffer)) if self.recent_sinr_buffer else 0.0,
                "recent_avg_actor_loss": float(np.mean(self.recent_actor_loss_buffer)) if self.recent_actor_loss_buffer else 0.0,
                "recent_avg_critic_loss": float(np.mean(self.recent_critic_loss_buffer)) if self.recent_critic_loss_buffer else 0.0,
                "recent_avg_raw_mu_bf_norm": float(np.mean(self.recent_raw_mu_bf_norm_buffer)) if self.recent_raw_mu_bf_norm_buffer else 0.0,
                "recent_avg_raw_mu_ris_norm": float(np.mean(self.recent_raw_mu_ris_norm_buffer)) if self.recent_raw_mu_ris_norm_buffer else 0.0,
                "current_rates_per_user": current_rates_per_user_dict
            }
            
            self.periodic_log_data.append(log_entry)
            self.interval_reward_buffer.clear()
            
        return True


# --- Custom Evaluation Logic from main2.py ---
def run_custom_evaluation(model, env, n_episodes, random_ris=False, random_bf=False, eval_seed_offset=0):
    safety_layer = model.policy.actor.safety_layer
    device = safety_layer.p_max.device

    episode_rewards, episode_avg_sumrates, episode_avg_jains = [], [], []
    episode_rates_per_user_list = [] 
    bf_params_count = 2 * env.get_attr('M')[0] * env.get_attr('K')[0]
    
    for i in range(n_episodes):
        # <-- START MODIFICATION: Use seed offset for eval -->
        env.seed(eval_seed_offset + i) 
        # <-- END MODIFICATION -->
        obs, done = env.reset(), np.array([False])
        total_reward, total_sumrate, total_jains, num_steps = 0.0, 0.0, 0.0, 0
        step_rates_per_user_list = [] 
        
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
            total_jains += infos[0].get('jains_fairness', 0.0)
            step_rates = infos[0].get('rates_per_user')
            if step_rates is not None:
                step_rates_per_user_list.append(step_rates)
            num_steps += 1
            
        episode_rewards.append(total_reward)
        if num_steps > 0:
            episode_avg_sumrates.append(total_sumrate / num_steps)
            episode_avg_jains.append(total_jains / num_steps)
            if step_rates_per_user_list:
                avg_rates_for_episode = np.mean(step_rates_per_user_list, axis=0)
                episode_rates_per_user_list.append(avg_rates_for_episode)
            
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_sumrate = np.mean(episode_avg_sumrates) if episode_avg_sumrates else 0.0
    mean_jains = np.mean(episode_avg_jains) if episode_avg_jains else 0.0
    
    mean_rates_per_user_dict = {}
    if episode_rates_per_user_list:
        avg_rates_all_episodes = np.mean(episode_rates_per_user_list, axis=0)
        mean_rates_per_user_dict = {f"user_{i}": float(rate) for i, rate in enumerate(avg_rates_all_episodes)}
        
    return mean_reward, std_reward, mean_sumrate, mean_jains, mean_rates_per_user_dict

# --- Main DRL runner, modified for timing ---
def run_drl_experiment_with_timing(config, seed: int): # <-- START MODIFICATION: Add seed
    """
    Runs a full DRL experiment and returns results AND complexity metrics.
    *** NOW ALSO RUNS RANDOM AGENT AND PLOTTING ***
    """
    run_cfg = config['run_settings']
    train_cfg = config['training_params']
    test_cfg = config['testing_params']
    env_cfg = config['env_parameters']
    model_type = run_cfg['model_type']
    
    model_hyperparams = copy.deepcopy(config['model_hyperparameters'].get('shared', {}))
    model_type_upper = model_type.upper()
    if model_type_upper == "TD3":
        specific_hp = config['model_hyperparameters'].get('TD3_shared', {}) # Use TD3_shared
    else:
        specific_hp = config['model_hyperparameters'].get(model_type_upper, {})
    for key, value in specific_hp.items():
        if key in model_hyperparams and isinstance(model_hyperparams[key], dict) and isinstance(value, dict):
            model_hyperparams[key].update(value)
        else:
            model_hyperparams[key] = value
            
    # <-- START MODIFICATION: Use provided seed -->
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # <-- END MODIFICATION -->
    
    total_timesteps = train_cfg['target_episodes_per_env'] * train_cfg['steps_per_episode'] * train_cfg['n_envs']
    
    # --- Create run directory (using a different base) ---
    rho = env_cfg.get('cascaded_error_rho', 1.0)
    jitter_deg = env_cfg.get('sigma_jitt_deg', 0.0)
    # <-- START MODIFICATION: Add seed to run name -->
    run_name = f"{model_type}_Rho{rho:.2f}_Jit{jitter_deg:.1f}_Seed{seed}_{int(time.time())}"
    # <-- END MODIFICATION -->
    run_dir = f"./runs_figures/{run_name}" # Use a separate dir for figure runs
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

    # <-- START MODIFICATION: Use seed for train_env -->
    train_env = make_vec_env(
        lambda: create_env(),
        n_envs=train_cfg['n_envs'],
        seed=seed, # Use main seed
        monitor_dir=logs_dir
    )
    # <-- END MODIFICATION -->
    
    monitor_callback = UnifiedLoggingCallback(
        total_timesteps,
        n_envs=train_cfg['n_envs'],
        enable_logging=run_cfg['enable_logging']
    )
    
    model = build_model(
        model_type,
        train_env,
        seed, # Pass main seed to model
        log_dir=os.path.join(run_dir, "tb_logs"),
        device=run_cfg['device'],
        verbose=0,
        model_hyperparams=model_hyperparams,
        env_params=env_cfg
    )
    
    # --- *** START MODIFICATION: REMOVED TIMING *** ---
    print(f"Starting training for {total_timesteps} total steps...")
    
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=monitor_callback,
        log_interval=4
    )
    # --- *** END MODIFICATION *** ---
    
    model.save(os.path.join(run_dir, "trained_model.zip"))
    train_env.close()

    # --- Evaluation ---
    print("\n" + "="*20 + " EVALUATION " + "="*20)
    eval_env = make_vec_env(lambda: create_env(is_eval=True), n_envs=1)
    
    # <-- START MODIFICATION: Use eval_seed from config -->
    eval_seed = run_cfg.get('eval_seed', 42)
    
    print("\n--- Evaluating: Trained Agent ---")
    mean_reward, std_reward, mean_sumrate, mean_jains, mean_rates_per_user = run_custom_evaluation(
        model, eval_env, test_cfg['test_episodes'], eval_seed_offset=eval_seed + 1000
    )
    
    print("\n--- Evaluating: Trained BF, Random RIS ---")
    mean_reward_rand_ris, std_reward_rand_ris, mean_sumrate_rand_ris, mean_jains_rand_ris, mean_rates_rand_ris = run_custom_evaluation(
        model, eval_env, test_cfg['test_episodes'], random_ris=True, eval_seed_offset=eval_seed + 2000
    )

    print("\n--- Evaluating: Random BF, Trained RIS ---")
    mean_reward_rand_bf, std_reward_rand_bf, mean_sumrate_rand_bf, mean_jains_rand_bf, mean_rates_rand_bf = run_custom_evaluation(
        model, eval_env, test_cfg['test_episodes'], random_bf=True, eval_seed_offset=eval_seed + 3000
    )
    # <-- END MODIFICATION -->
    
    # --- *** START MODIFICATION: REMOVED INFERENCE TIMING *** ---
    # (Section 2. CAPTURE ONLINE INFERENCE TIME removed)
    # --- *** END MODIFICATION *** ---
    
    eval_env.close()
    
    # --- *** RESTORED: Fully Random Agent Evaluation *** ---
    print("\n--- Evaluating: Fully Random Agent ---")
    random_env = make_vec_env(lambda: create_env(is_eval=True), n_envs=1)
    random_rewards, random_avg_sumrates, random_avg_jains = [], [], []
    random_rates_per_user_list = [] 
    
    safety_layer = model.policy.actor.safety_layer
    device = safety_layer.p_max.device

    for i in range(test_cfg['test_episodes']):
        # <-- START MODIFICATION: Use eval_seed from config -->
        random_env.seed(eval_seed + i)
        # <-- END MODIFICATION -->
        obs = random_env.reset()
        done = np.array([False])
        
        total_reward, total_sumrate, num_steps = 0.0, 0.0, 0
        total_jains = 0.0
        step_rates_per_user_list = [] 
        
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
            total_jains += infos[0].get('jains_fairness', 0.0)
            step_rates = infos[0].get('rates_per_user')
            if step_rates is not None:
                step_rates_per_user_list.append(step_rates)
            
            num_steps += 1
            
        random_rewards.append(total_reward)
        if num_steps > 0:
            random_avg_sumrates.append(total_sumrate / num_steps)
            random_avg_jains.append(total_jains / num_steps)
            if step_rates_per_user_list:
                avg_rates_for_episode = np.mean(step_rates_per_user_list, axis=0)
                random_rates_per_user_list.append(avg_rates_for_episode)
            
    random_env.close()
    
    mean_random_reward = np.mean(random_rewards)
    mean_random_sumrate = np.mean(random_avg_sumrates) if random_avg_sumrates else 0.0
    mean_random_jains = np.mean(random_avg_jains) if random_avg_jains else 0.0
    
    mean_random_rates_per_user_dict = {}
    if random_rates_per_user_list:
        avg_rates_all_episodes = np.mean(random_rates_per_user_list, axis=0)
        mean_random_rates_per_user_dict = {f"user_{i}": float(rate) for i, rate in enumerate(avg_rates_all_episodes)}
    
    print(f"Result (Reward): {mean_random_reward:.4f}")
    print(f"Result (Sum-rate): {mean_random_sumrate:.4f}")
    print(f"Result (Jain's Fairness): {mean_random_jains:.4f}")
    # --- *** END OF RESTORED SECTION *** ---

    # --- *** RESTORED: Plotting and Summary.json Generation *** ---
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
        mean_eval_jains=mean_jains,
        mean_random_jains=mean_random_jains,
        mean_eval_rates_per_user=mean_rates_per_user,
        mean_random_rates_per_user=mean_random_rates_per_user_dict
    )
    # --- *** END OF RESTORED SECTION *** ---
    
    results_dict = {
        "evaluated_agent_mean_sumrate": mean_sumrate,
        "evaluated_agent_mean_reward": mean_reward,
        "trained_bf_random_ris_mean_sumrate": mean_sumrate_rand_ris,
        "random_bf_trained_ris_mean_sumrate": mean_sumrate_rand_bf,
        # --- *** START MODIFICATION: REMOVED TIMING *** ---
        # "offline_time_sec": offline_time_sec,
        # "avg_inference_time_sec": avg_inference_time_sec,
        # "total_timesteps": total_timesteps
        # --- *** END MODIFICATION *** ---
    }
    
    # Return all metrics
    # --- *** START MODIFICATION: REMOVED TIMING *** ---
    return (
        results_dict, 
        run_dir, 
        seed 
    )
    # --- *** END MODIFICATION *** ---


# ===================================================================
# Helper Functions (Plotting and Loading)
# ===================================================================

def safe_to_numeric(index):
    try:
        return pd.to_numeric(index)
    except ValueError:
        return index

# --- NEW OPTIMIZER BENCHMARK RUNNER ---
def run_optimizer_benchmark(config: dict, method_name: str, seed: int) -> dict: # <-- MOD: Add seed
    """
    Runs a non-parallel benchmark for a classical optimizer (AO or AO-SAA).
    Returns mean sum-rate and mean inference time.
    """
    bench_cfg = config['benchmark_settings']
    env_cfg = config['env_parameters']
    run_cfg = config.get('run_settings', {})
    # <-- MOD: Use provided seed for the master_rng -->
    master_rng = np.random.default_rng(seed)
    # <-- END MOD -->
    verbose = bench_cfg.get('verbose', False)

    # Use parameters from the correct config block
    common_params = {
        "wmmse_bisection_steps": bench_cfg.get("WMMSE_EIG_Params", {}).get("wmmse_bisection_steps", 30),
        "wmmse_bisection_lam_high": bench_cfg.get("WMMSE_EIG_Params", {}).get("wmmse_bisection_lam_high", 1e6)
    }
    method_params = dict(bench_cfg.get("WMMSE_EIG_Params", {}))
    method_params.update(common_params)
    if "SAA_Params" in bench_cfg:
        method_params["SAA_Params"] = dict(bench_cfg.get("SAA_Params", {}))

    # Create a single env for robust reward calculation
    env_cfg_eval = copy.deepcopy(env_cfg)
    if env_cfg.get('cascaded_error_rho', 1.0) == 1.0 and env_cfg.get('sigma_jitt_deg', 0.0) == 0.0:
        env_cfg_eval['S_samples'] = 1
        print(f"[{method_name} Runner] Perfect CSI detected. Setting S_samples = 1.")
    else:
        env_cfg_eval['S_samples'] = env_cfg.get('S_samples', 1)

    env = gym.make("UAV-RIS-v0", **env_cfg_eval)
    
    all_sum_rates = []
    # --- *** START MODIFICATION: REMOVED TIMING *** ---
    # all_inference_times = []
    # --- *** END MODIFICATION *** ---
    num_episodes = bench_cfg['num_episodes']

    episode_seeds = master_rng.integers(low=0, high=2**31, size=num_episodes)

    # --- Serial loop (no multiprocessing) ---
    for i in tqdm(range(num_episodes), desc=f"Running {method_name} (Seed {seed})"):
        episode_seed = int(episode_seeds[i])
        
        # Reset env to get new channels (but don't use env's optimizer)
        env.reset(seed=episode_seed)
        u = env.unwrapped # Get internal params
        
        # Build estimated channels *exactly* as in oracle.py
        H_1_Ric = u.k_term_los_1 * u.h1_los_est + u.k_term_nlos_1 * u.h1_nlos_est
        H_2_Ric = u.k_term_los_2 * u.h2_los_est + u.k_term_nlos_2 * u.h2_nlos_est
        H_1_est = u.sqrt_path_loss_1 * H_1_Ric
        H_2_est = u.sqrt_path_loss_2_vec[np.newaxis, :] * H_2_Ric
        sigma2  = float(u.awgn_power_watts)
        
        # Instantiate the correct optimizer
        try:
            if method_name == "WMMSE_EIG_SAA":
                opt = AO_WMMSE_Eig_SAA(env_cfg, method_params, env_u=u, verbose=verbose)
            else: # Default to "WMMSE_EIG"
                opt = AO_WMMSE_Eig(env_cfg, method_params, verbose=verbose)

            # --- *** START MODIFICATION: REMOVED TIMING *** ---
            G, phi_vec = opt.optimize(H_1_est, H_2_est, sigma2, seed=int(episode_seed))
            # --- *** END MODIFICATION *** ---
            
            # Evaluate on the env's true (robust) channel
            results = env.unwrapped._calculate_robust_reward(G, phi_vec)
            all_sum_rates.append(results["avg_sum_rate"])
            
        except Exception as e:
            print(f"Warning: Optimizer failed for episode {episode_seed}: {e}")
            continue
            
    env.close()
    mean_sum_rate = np.mean(all_sum_rates) if all_sum_rates else 0.0
    # --- *** START MODIFICATION: REMOVED TIMING *** ---
    # mean_inference_time_sec = np.mean(all_inference_times) if all_inference_times else 0.0
    # print(f"Avg. {method_name} Inference Time: {mean_inference_time_sec * 1000:.6f} ms")
    # --- *** END MODIFICATION *** ---
    
    return {
        "evaluated_agent_mean_sumrate": mean_sum_rate,
        # --- *** START MODIFICATION: REMOVED TIMING *** ---
        # "avg_inference_time_sec": mean_inference_time_sec,
        # --- *** END MODIFICATION *** ---
        "trained_bf_random_ris_mean_sumrate": 0.0,
        "seed": seed # <-- MOD: Return seed
    }

# --- UPDATED PLOTTING FUNCTIONS ---

def calculate_ci_95(data):
    """ Helper to calculate 95% CI from a list/array of data """
    data = np.asarray(data)
    data = data[~np.isnan(data)] # Remove NaNs
    if len(data) < 2:
        return 0.0
    
    n = len(data)
    mean = np.mean(data)
    sem = st.sem(data) # Standard Error of the Mean
    if sem == 0:
        return 0.0
        
    # Get t-value for 95% CI
    t_val = st.t.ppf(1 - 0.025, df=n-1)
    
    return sem * t_val

def plot_robustness_results(raw_df, output_filename, title, xlabel, model_types, invert_xaxis=False, xticklabels=None):
    """
    Generates the final robustness plot from a completed RAW DataFrame.
    MODIFIED: Uses error bars, markers, and no connecting lines.
    """
    print(f"\nGenerating plot: {output_filename}")
    
    # --- Aggregate raw data to get mean and 95% CI ---
    # Make a copy to avoid SettingWithCopyWarning
    raw_df_copy = raw_df.copy()
    
    # --- *** CHANGE: Group by Jitter *** ---
    raw_df_copy.rename(columns={'Jitter': 'Jitter (Degrees)'}, inplace=True)
    
    # Main results
    grouped_main = raw_df_copy.groupby(['Jitter (Degrees)', 'Algorithm'])['Mean Sum-Rate'].agg(
        mean='mean', 
        std='std', 
        count='count'
    ).reset_index()
    
    # Calculate CI
    grouped_main['ci_95'] = grouped_main.apply(
        lambda row: calculate_ci_95(raw_df_copy[
            (raw_df_copy['Jitter (Degrees)'] == row['Jitter (Degrees)']) & 
            (raw_df_copy['Algorithm'] == row['Algorithm'])
        ]['Mean Sum-Rate']), 
        axis=1
    )
    
    # BF-Only results
    grouped_bf_only = raw_df_copy[raw_df_copy['Mean Sum-Rate (BF-Only)'] > 0].groupby(['Jitter (Degrees)', 'Algorithm'])['Mean Sum-Rate (BF-Only)'].agg(
        mean='mean', 
        std='std', 
        count='count'
    ).reset_index()
    
    if not grouped_bf_only.empty:
        grouped_bf_only['ci_95'] = grouped_bf_only.apply(
            lambda row: calculate_ci_95(raw_df_copy[
                (raw_df_copy['Jitter (Degrees)'] == row['Jitter (Degrees)']) & 
                (raw_df_copy['Algorithm'] == row['Algorithm'])
            ]['Mean Sum-Rate (BF-Only)']), 
            axis=1
        )
        grouped_bf_only['Algorithm'] = grouped_bf_only['Algorithm'] + ' (BF-Only)'
    
    # Combine all aggregated data
    results_df = pd.concat([grouped_main, grouped_bf_only], ignore_index=True)
    # --- End Aggregation ---

    try:
        # --- *** CHANGE: Sort by Jitter *** ---
        results_df['Jitter (Degrees)'] = pd.to_numeric(results_df['Jitter (Degrees)'])
        results_df = results_df.sort_values(by='Jitter (Degrees)', ascending=not invert_xaxis)
    except Exception:
        pass # Keep as is

    print(results_df.to_string())
    
    fig, ax = plt.subplots(figsize=fig_size("single"))
    
    # --- *** CHANGE: Use Jitter for x-axis *** ---
    plot_x_values = results_df['Jitter (Degrees)'].unique()

    # Manually assign styles for consistent legend & B&W friendliness
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
            
            # --- *** START MODIFICATION: Check for optimizers *** ---
            if mt in ["AO-WMMSE", "AO-WMMSE-SAA"]:
                y_error = None
            else:
                y_error = model_data['ci_95']
            # --- *** END MODIFICATION *** ---
            
            ax.errorbar(
                x=model_data['Jitter (Degrees)'], # <-- Plot against Jitter
                y=model_data['mean'], 
                yerr=y_error, # <-- Use new y_error variable
                label=mt, 
                color=color,
                marker=marker,
                linestyle='None', # <-- YOUR REQUEST
                capsize=3, # Makes CI bars easier to see
                elinewidth=0.8, # Thinner error lines
                markeredgewidth=0.8 # Thinner marker lines
            )
        else:
            print(f"Warning: No valid data for {mt}. Skipping line in plot.")
    
    ax.set_xlabel(xlabel) # Use the passed xlabel
    ax.set_ylabel("Mean Sum-Rate (bps/Hz)")
    
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.margins(x=0.03)
    
    if xticklabels:
        ax.set_xticks(ticks=plot_x_values, labels=xticklabels, rotation=0)
    elif invert_xaxis:
        ax.invert_xaxis()
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
                loc="outside upper center",
                ncol=3, # Adjusted for better fit
                frameon=False,
                handlelength=2.0, handletextpad=0.6, columnspacing=1.0)
    
    pdf_filename = output_filename.replace(".png", ".pdf")
    plt.savefig(pdf_filename, bbox_inches="tight")
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"Successfully saved plot: {output_filename} and {pdf_filename}")
    plt.close(fig)

def plot_convergence_for_level(rho: float, jitter: float, base_config: dict, plot_name_suffix: str, results_dir: str, seeds: list):
    """
    Generates a training convergence plot for a specific error level from multiple seeds.
    MODIFIED: 
    - x-axis scaled to 10^3, only TD3/DDPG
    - Plots mean curve + CI band across seeds
    - NO final error bar
    - Uses distinct linestyles
    - *** NEW "RIGOROUS" METHOD ***
    """
    
    print(f"\n--- Generating Convergence Plot for: {plot_name_suffix} (Rho={rho:.2f}, Jitter={jitter:.1f}) ---")

    drl_models_to_plot = ["TD3", "DDPG"]
    base_run_dir = "./runs_figures" # Point to the new run dir
    n_envs = base_config.get('training_params', {}).get('n_envs', 1)
    
    all_seeds_data = {model: [] for model in drl_models_to_plot}
    
    for model in drl_models_to_plot:
        for seed in seeds:
            # Find the latest run directory for this specific (model, rho, jit, seed)
            patterns = [
                f"{model}_Rho{rho:.2f}_Jit{jitter:.1f}_Seed{seed}_",
            ]
            all_runs_in_basedir = [os.path.join(base_run_dir, d) for d in os.listdir(base_run_dir) if os.path.isdir(os.path.join(base_run_dir, d))]
            relevant_runs = [d for d in all_runs_in_basedir if any(os.path.basename(d).startswith(p) for p in patterns)]

            if relevant_runs:
                latest_run_dir = max(relevant_runs, key=os.path.getmtime)
                # print(f"Loading convergence data from: {latest_run_dir}") # Too verbose
                logs_subdir = os.path.join(latest_run_dir, "logs")
                
                # --- We only expect one monitor file (0.monitor.csv) since n_envs=1 ---
                monitor_path = os.path.join(logs_subdir, "0.monitor.csv")
                
                if os.path.exists(monitor_path):
                    try:
                        df = pd.read_csv(monitor_path, skiprows=1)
                        if not df.empty and 'r' in df.columns and 'l' in df.columns and 't' in df.columns:
                            # Use cumulative sum of episode lengths ('l') to get timestep
                            df['timestep'] = df['l'].cumsum()
                            df['r'] = pd.to_numeric(df['r'], errors='coerce') # Ensure reward is numeric
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

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=fig_size("single"))
    
    total_steps = base_config['training_params']['target_episodes_per_env'] * base_config['training_params']['steps_per_episode'] * base_config['training_params']['n_envs']
    max_timestep_plot = min(total_steps, 200000) # Max 200k steps
    plot_has_drl_data = False
    
    color_map = {"TD3": "C0", "DDPG": "C1"}
    linestyle_map = {"TD3": "-", "DDPG": "--"} # <-- YOUR REQUEST: B&W friendly
    
    # Use a common index for interpolation (e.g., 500 points)
    common_timesteps = np.linspace(0, max_timestep_plot, 500)
    
    for model_type, seed_dfs in all_seeds_data.items():
        if not seed_dfs:
            continue
            
        plot_has_drl_data = True
        
        # --- *** START: NEW "RIGOROUS" LOGIC *** ---
        interpolated_raw_dfs = []
        
        for df in seed_dfs:
            # Filter, clean
            df_filtered = df[df['timestep'] <= max_timestep_plot].copy()
            if df_filtered.empty:
                continue
                
            # Interpolate the RAW 'r' column
            interp_func = np.interp(
                common_timesteps, 
                df_filtered['timestep'], 
                df_filtered['r'], # <-- Use raw reward
                left=df_filtered['r'].iloc[0], 
                right=df_filtered['r'].iloc[-1]
            )
            interpolated_raw_dfs.append(interp_func)

        if not interpolated_raw_dfs:
            continue

        # --- Aggregate across seeds ---
        # (n_seeds, n_timesteps)
        stacked_data = np.stack(interpolated_raw_dfs, axis=0)
        
        # Calculate mean/CI from RAW interpolated data
        raw_mean_curve = np.nanmean(stacked_data, axis=0)
        n_seeds_per_step = np.sum(~np.isnan(stacked_data), axis=0)
        std_curve = np.nanstd(stacked_data, axis=0)
        
        # Calculate 95% CI for the curve (this will be noisy)
        ci_95_curve = np.zeros_like(raw_mean_curve)
        df_t = np.maximum(1, n_seeds_per_step - 1)
        t_vals = st.t.ppf(1 - 0.025, df=df_t)
        
        safe_sqrt_n = np.sqrt(n_seeds_per_step, where=n_seeds_per_step > 0, out=np.full_like(n_seeds_per_step, 1.0, dtype=float))
        sem = std_curve / safe_sqrt_n
        sem[n_seeds_per_step == 0] = 0.0 
        
        ci_95_curve = sem * t_vals
        ci_95_curve = np.nan_to_num(ci_95_curve, 0.0)

        # --- Calculate the SMOOTHED mean line ---
        # Scale the 2000-step window to our 500-point interpolated array
        window_size_interp = int( (2000 / max_timestep_plot) * 500 )
        window_size_interp = max(1, window_size_interp)
        
        smoothed_mean_curve = pd.Series(raw_mean_curve).rolling(window=window_size_interp, min_periods=1).mean()

        # --- Plot Mean Curve (Smoothed) and CI Band (Raw) ---
        color = color_map.get(model_type, "C2")
        linestyle = linestyle_map.get(model_type, ":")
        
        # X-axis divided by 1000
        x_axis_data = common_timesteps / 1000.0
        
        ax.plot(x_axis_data, smoothed_mean_curve, # <-- Plot SMOOTHED mean
                label=f"{model_type} (Mean)",
                color=color, 
                linestyle=linestyle)
        
        ax.fill_between(
            x_axis_data, 
            raw_mean_curve - ci_95_curve, # <-- Plot RAW CI
            raw_mean_curve + ci_95_curve, 
            color=color, 
            alpha=0.15, 
            label=f"{model_type} (95% CI Band)"
        )
        # --- *** END: NEW "RIGOROUS" LOGIC *** ---
                
    if not plot_has_drl_data:
        print(f"Warning: No DRL data found to plot for {plot_name_suffix}.")
        ax.text(0.5, 0.5, "No DRL data available.", ha='center', va='center', transform=ax.transAxes)
        
    ax.set_title(rf"($\rho={rho:.2f}, \sigma_j={jitter:.1f}^\circ$) Convergence")
    ax.set_xlabel("Training Timesteps ($10^3$)") # Updated X-axis label
    ax.set_ylabel("Sum-Rate (bps/Hz)")
    
    # --- Create a clean legend ---
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    for h, l in zip(handles, labels):
        if "95% CI Band" not in l: # Keep only the mean lines
            unique_labels[l] = h
             
    ax.legend(unique_labels.values(), unique_labels.keys(), frameon=False, loc='lower right', fontsize=6)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.set_xlim(0, max_timestep_plot / 1000.0) # Scaled X-axis limit
    ax.margins(x=0.03)
    
    filename_stem = f"paper_figure_training_{plot_name_suffix.replace('.', '_')}"
    
    # --- *** START FIX: Use results_dir (lowercase) *** ---
    pdf_path = os.path.join(results_dir, f"{filename_stem}.pdf")
    png_path = os.path.join(results_dir, f"{filename_stem}.png")
    # --- *** END FIX *** ---
    
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")
    print(f"Successfully saved plot: {png_path} and {pdf_path}")
    plt.close(fig)

# ===================================================================
# Main Execution Logic (This is the JITTER script)
# ===================================================================

def main():
    config_path = 'config.yaml'
    try:
        with open(config_path, 'r') as f:
            master_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config not found at {config_path}")
        return

    # --- UPDATED model names ---
    model_types_to_run = ["TD3", "DDPG", "AO-WMMSE", "AO-WMMSE-SAA"]
    model_types_to_plot = ["TD3", "DDPG", "AO-WMMSE", "AO-WMMSE-SAA", "TD3 (BF-Only)", "DDPG (BF-Only)"]
    
    # --- *** CHANGE: Set paths for JITTER *** ---
    RESULTS_DIR = "./paper_figures_jitter"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("./runs_figures/logs", exist_ok=True) # Ensure logs dir exists for DRL runs
    
    raw_results_csv_path = os.path.join(RESULTS_DIR, "jitter_robustness_raw.csv")
    
    # <-- *** YOUR REQUEST: 10 seeds ***
    SEEDS_TO_RUN = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90] 
    
    print(f"--- Running {len(SEEDS_TO_RUN)} seeds per experiment: {SEEDS_TO_RUN} ---")

    # --- 1. Load results cache and build job list ---
    print("\n" + "="*20 + " BUILDING JOB LIST " + "="*20)
    # --- *** CHANGE: Use jitter_levels *** ---
    jitter_levels = np.arange(0, 10.1, 2.0) 
    
    # --- *** NEW: Load raw results to find completed jobs *** ---
    completed_jobs = set()
    if os.path.exists(raw_results_csv_path):
        try:
            completed_df = pd.read_csv(raw_results_csv_path)
            for _, row in completed_df.iterrows():
                # --- *** CHANGE: Use Jitter for cache key *** ---
                completed_jobs.add((row['Algorithm'], float(row['Jitter']), int(row['Seed'])))
            print(f"Loaded {len(completed_jobs)} completed job-seeds from cache.")
        except Exception as e:
            print(f"Warning: Could not read {raw_results_csv_path}. Re-running all jobs. Error: {e}")
            
    # --- Split jobs into DRL (GPU) and CPU lists ---
    drl_jobs = []
    cpu_jobs = []
    
    # --- *** CHANGE: Loop over jitter_levels *** ---
    for jitter in jitter_levels:
        for model in model_types_to_run:
            for seed in SEEDS_TO_RUN:
                job_tuple = (model, float(jitter), int(seed)) # Cache key
                
                if job_tuple in completed_jobs:
                    print(f"Skipping (cached): {model} @ Jitter {jitter:.1f} (Seed {seed})")
                else:
                    print(f"Adding to queue:   {model} @ Jitter {jitter:.1f} (Seed {seed})")
                    # Full job info
                    job_info = (model, float(jitter), int(seed)) 
                    if model in ["TD3", "DDPG"]:
                        drl_jobs.append(job_info)
                    else:
                        cpu_jobs.append(job_info)
    # --- *** END OF CHANGE *** ---

    total_jobs = len(drl_jobs) + len(cpu_jobs)
    if not total_jobs:
        print("No new experiments to run. All results are cached.")
    else:
        print(f"\n" + "="*20 + f" RUNNING {total_jobs} EXPERIMENTS WITH POPEN " + "="*20)
        
        # --- 2. Run jobs with Popen ---
        # <-- *** YOUR REQUEST: 5 DRL + 5 CPU ***
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
                model, jitter, seed = drl_jobs.pop(0)
                # --- *** CHANGE: Pass --jitter and --seed *** ---
                cmd = [sys.executable, __file__, "--model", model, "--jitter", str(jitter), "--seed", str(seed)]
                # print(f"\nStarting DRL job: {' '.join(cmd)}") # Too verbose
                p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # Hide worker output
                drl_processes.append(p)

            while len(cpu_processes) < MAX_CPU_RUNS and cpu_jobs:
                model, jitter, seed = cpu_jobs.pop(0)
                # --- *** CHANGE: Pass --jitter and --seed *** ---
                cmd = [sys.executable, __file__, "--model", model, "--jitter", str(jitter), "--seed", str(seed)]
                # print(f"\nStarting CPU job: {' '.join(cmd)}") # Too verbose
                p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # Hide worker output
                cpu_processes.append(p)
            
            time.sleep(1.0)
            
        pbar.close()

    # --- 3. Plotting (runs after all Popen jobs are done) ---
    print("\n" + "="*20 + " PLOTTING RESULTS " + "="*20)
    
    if not os.path.exists(raw_results_csv_path):
        print(f"Error: Raw results file not found: {raw_results_csv_path}. Cannot plot.")
        return
        
    try:
        raw_df_final = pd.read_csv(raw_results_csv_path)
    except Exception as e:
        print(f"Error loading {raw_results_csv_path}: {e}. Cannot plot.")
        return
    
    # Plot 1: Jitter Robustness (with error bars)
    plot_robustness_results(
        raw_df_final, # <-- MOD: Pass raw dataframe
        os.path.join(RESULTS_DIR, "paper_figure_jitter_robustness.png"),
        title="Performance vs. UAV Jitter",
        xlabel=r"UAV Jitter Standard Deviation, $\sigma_j$ ($^\circ$)",
        model_types=model_types_to_plot,
        invert_xaxis=False # Jitter increases 0 -> 10
    )

    # Plot 2: Convergence plots for ALL jitter levels
    print("\n--- Generating All Convergence Plots ---")
    for jitter in jitter_levels:
        plot_convergence_for_level(
            rho=1.0, # Fixed rho for this sweep
            jitter=float(jitter), 
            base_config=master_config, 
            plot_name_suffix=f"jit_{jitter:.1f}".replace('.', '_'),
            results_dir=RESULTS_DIR,
            seeds=SEEDS_TO_RUN
        )

    # --- *** START MODIFICATION: REMOVED TIMING *** ---
    # (Section 4. Complexity CSV Generation removed)
    # --- *** END MODIFICATION *** ---

    print("\n" + "="*20 + " ALL EXPERIMENTS FINISHED " + "="*20)

# ===================================================================
# Entry Point (MODIFIED)
# ===================================================================
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Run a single experiment worker.")
    # --- *** CHANGE: Add --jitter and --seed *** ---
    parser.add_argument("--model", type=str, help="Model type to run (e.g., TD3, AO-WMMSE)")
    parser.add_argument("--jitter", type=float, help="Jitter value to run")
    parser.add_argument("--seed", type=int, help="Seed value to run")
    args = parser.parse_args()

    if args.model and args.jitter is not None and args.seed is not None:
        # --- WORKER MODE ---
        
        model = args.model
        jitter = args.jitter
        seed = args.seed
        
        # Suppress output from worker
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        # 1. Load config
        config_path = 'config.yaml'
        try:
            with open(config_path, 'r') as f:
                master_config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"[Worker Error] Config not found at {config_path}")
            sys.exit(1)
            
        # 2. Set up run-specific config
        run_config = copy.deepcopy(master_config)
        # --- *** CHANGE: Set jitter, fix rho *** ---
        run_config['env_parameters']['sigma_jitt_deg'] = float(jitter)
        run_config['env_parameters']['cascaded_error_rho'] = 1.0
        if float(jitter) == 0.0:
             run_config['env_parameters']['S_samples'] = 1
        # --- *** END OF CHANGE *** ---
        
        # 3. Run the experiment
        try:
            eval_results = {}
            if model == "AO-WMMSE" or model == "AO-WMMSE-SAA":
                method_name_config = "WMMSE_EIG" if model == "AO-WMMSE" else "WMMSE_EIG_SAA"
                run_config['benchmark_settings']['method'] = method_name_config
                eval_results = run_optimizer_benchmark(run_config, method_name_config, seed)
                
            elif model in ["TD3", "DDPG"]:
                run_config['run_settings']['model_type'] = model
                (eval_results, _run_dir, 
                 _seed) = run_drl_experiment_with_timing(run_config, seed)
            
            # 4. Save sum-rate results to CSV (atomically)
            RESULTS_DIR = "./paper_figures_jitter"
            raw_results_csv_path = os.path.join(RESULTS_DIR, "jitter_robustness_raw.csv")
            lock_path = raw_results_csv_path + ".lock"

            with FileLock(lock_path):
                file_exists = os.path.isfile(raw_results_csv_path)
                with open(raw_results_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        # --- *** CHANGE: Add Jitter to header *** ---
                        writer.writerow(['Algorithm', 'Jitter', 'Seed', 'Mean Sum-Rate', 'Mean Sum-Rate (BF-Only)'])
                    
                    mean_sumrate = eval_results.get("evaluated_agent_mean_sumrate", 0.0)
                    mean_sumrate_bf_only = 0.0
                    if model in ["TD3", "DDPG"]:
                        mean_sumrate_bf_only = eval_results.get("trained_bf_random_ris_mean_sumrate", 0.0)
                    
                    # --- *** CHANGE: Add Jitter to row *** ---
                    writer.writerow([model, jitter, seed, mean_sumrate, mean_sumrate_bf_only])
            
            # --- *** START MODIFICATION: REMOVED TIMING *** ---
            # (Section 5. Save complexity results removed)
            # --- *** END MODIFICATION *** ---

        except Exception as e:
            # Try to write to a log file if all else fails
            with open("figure_jitter_worker_error.log", "a") as f:
                f.write(f"!!!!!!!! FAILED: {model} @ Jitter {jitter:.1f} (Seed {seed}). Error: {e} !!!!!!!!\n")
                import traceback
                traceback.print_exc(file=f)
            sys.exit(1)
        
    else:
        # --- ORCHESTRATOR MODE ---
        # No args were passed, run the main orchestrator
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            print("Multiprocessing start method already set.")
            pass
        
        main()