import os
import numpy as np
import torch
import random
import gymnasium as gym
import env_registration
import datetime
import yaml
import json
import copy
from collections import deque
from model_builder import build_model

import matplotlib
matplotlib.use("Agg")
from Plotting import plot_results

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit


# ===================================================================
# Unified Logging Callback
# ===================================================================
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
            
            # --- CLEANED LOG ENTRY ---
            log_entry = {
                "timestep": self.num_timesteps,
                "avg_reward_since_last_log": float(np.mean(self.interval_reward_buffer)) if self.interval_reward_buffer else 0.0,
                "recent_avg_sumrate": float(np.mean(self.recent_sumrate_buffer)) if self.recent_sumrate_buffer else 0.0,
            }
            # -------------------------
            
            self.periodic_log_data.append(log_entry)
            print(f"\n--- Logging Training Metrics at Timestep {self.num_timesteps} ---")
            print(json.dumps(log_entry, indent=2))
            self.interval_reward_buffer.clear()
            
        return True


# ===================================================================
# Custom Evaluation Logic
# ===================================================================
def run_custom_evaluation(model, env, n_episodes, random_ris=False, random_bf=False):
    # Get the safety layer from the trained model to project random actions
    safety_layer = model.policy.actor.safety_layer
    device = safety_layer.p_max.device

    episode_rewards, episode_avg_sumrates = [], []
    bf_params_count = 2 * env.get_attr('M')[0] * env.get_attr('K')[0]
    
    for _ in range(n_episodes):
        obs, done = env.reset(), np.array([False])
        total_reward, total_sumrate, num_steps = 0.0, 0.0, 0
        
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            
            if random_ris:
                ris_dims = action.shape[1] - bf_params_count
                action[0, bf_params_count:] = np.random.uniform(-1.0, 1.0, size=ris_dims)
            if random_bf:
                action[0, :bf_params_count] = np.random.uniform(-1.0, 1.0, size=bf_params_count)
            
            # Project the action if it contains random components
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


# ===================================================================
# Main Execution Logic
# ===================================================================
def main(config):
    run_cfg = config['run_settings']
    train_cfg = config['training_params']
    test_cfg = config['testing_params']
    env_cfg = config['env_parameters']
    
    model_type = run_cfg['model_type']
    
    # Merge hyperparameters
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
            
    # Set seeds
    seed = run_cfg['seed']
    np.random.seed(20)
    random.seed(20)
    torch.manual_seed(seed)
    
    total_timesteps = train_cfg['target_episodes_per_env'] * train_cfg['steps_per_episode'] * train_cfg['n_envs']
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    jitter_deg = env_cfg.get('sigma_jitt_deg', 0.0)
    rho = env_cfg.get('cascaded_error_rho', 1.0) 
    
    run_name = f"{model_type}_Rho{rho:.2f}_Jit{jitter_deg:.1f}_{current_time}"
    run_dir = f"./runs/{run_name}"
        
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"All artifacts for this run will be saved in: {run_dir}")
    
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

    train_env_unwrapped = make_vec_env(
        lambda: create_env(),
        n_envs=train_cfg['n_envs'],
        seed=run_cfg['training_seed'],
        monitor_dir=logs_dir
    )
    
    env = train_env_unwrapped
    print("VecNormalize is DISABLED.")

    monitor_callback = UnifiedLoggingCallback(
        total_timesteps,
        n_envs=train_cfg['n_envs'],
        enable_logging=run_cfg['enable_logging']
    )
    
    model = build_model(
        model_type,
        env,
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
    env.close()

    # ===================================================================
    # Evaluation
    # ===================================================================
    print("\n" + "="*20 + " EVALUATION " + "="*20)
    
    eval_env_unwrapped = make_vec_env(lambda: create_env(is_eval=True), n_envs=1)
    eval_env = eval_env_unwrapped
        
    print("\n--- Evaluating: Trained Agent ---")
    mean_reward, std_reward, mean_sumrate = run_custom_evaluation(
        model, eval_env, test_cfg['test_episodes']
    )
    print(f"Result (Reward): {mean_reward:.4f} +/- {std_reward:.4f}")
    print(f"Result (Sum-rate): {mean_sumrate:.4f}")
    
    print("\n--- Evaluating: Trained BF, Random RIS ---")
    mean_reward_rand_ris, std_reward_rand_ris, mean_sumrate_rand_ris = run_custom_evaluation(
        model, eval_env, test_cfg['test_episodes'], random_ris=True
    )
    print(f"Result (Reward): {mean_reward_rand_ris:.4f} +/- {std_reward_rand_ris:.4f}")
    print(f"Result (Sum-rate): {mean_sumrate_rand_ris:.4f}")
    
    print("\n--- Evaluating: Random BF, Trained RIS ---")
    mean_reward_rand_bf, std_reward_rand_bf, mean_sumrate_rand_bf = run_custom_evaluation(
        model, eval_env, test_cfg['test_episodes'], random_bf=True
    )
    print(f"Result (Reward): {mean_reward_rand_bf:.4f} +/- {std_reward_rand_bf:.4f}")
    print(f"Result (Sum-rate): {mean_sumrate_rand_bf:.4f}")
    
    eval_env.close()
    
    print("\n--- Evaluating: Fully Random Agent ---")
    random_env = make_vec_env(lambda: create_env(is_eval=True), n_envs=1)
    
    random_rewards, random_avg_sumrates = [], []
    safety_layer = model.policy.actor.safety_layer
    device = safety_layer.p_max.device

    for i in range(test_cfg['test_episodes']):
        random_env.seed(run_cfg['eval_seed'] + i)
        obs = random_env.reset()
        done = np.array([False])
        
        total_reward, total_sumrate, num_steps = 0.0, 0.0, 0
        
        while not done[0]:
            raw_action_1d_np = random_env.action_space.sample()
            raw_action_np = raw_action_1d_np[None, :]
            
            raw_action_tensor = torch.as_tensor(
                raw_action_np, device=device, dtype=torch.float32
            )
            
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
    
    # ===================================================================
    # Plotting
    # ===================================================================
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
        "evaluated_agent_std_reward": std_reward,
        "random_agent_mean_sumrate": mean_random_sumrate,
        "random_agent_mean_reward": mean_random_reward,
        "trained_bf_random_ris_mean_reward": mean_reward_rand_ris,
        "trained_bf_random_ris_mean_sumrate": mean_sumrate_rand_ris,
        "random_bf_trained_ris_mean_reward": mean_reward_rand_bf,
        "random_bf_trained_ris_mean_sumrate": mean_sumrate_rand_bf,
    }
    
    return results_dict, run_dir


if __name__ == '__main__':
    config_path = 'config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config not found at {config_path}")
        exit(1)
        
    main(config)