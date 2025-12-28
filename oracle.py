import os
import yaml
import numpy as np
import gymnasium as gym
import env_registration
from tqdm import tqdm
import multiprocessing as mp

from classical_optimizers import AO_WMMSE_Eig, AO_WMMSE_Eig_SAA

# -------------------- Per-worker Initializer --------------------
def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("MOSEK_NUM_THREADS", "1")

# -------------------- Single-Episode Function --------------------
def run_single_episode(args):
    """
    Run one episode and return (avg_reward, avg_sum_rate).
    """
    episode_seed, env_cfg, bench_cfg = args
    verbose = bench_cfg.get("verbose", False)

    common_params = {
        "wmmse_bisection_steps": bench_cfg.get("wmmse_bisection_steps", 30),
        "wmmse_bisection_lam_high": bench_cfg.get("wmmse_bisection_lam_high", 1e6)
    }

    env = gym.make("UAV-RIS-v0", **env_cfg)
    env.reset(seed=int(episode_seed))
    u = env.unwrapped

    try:
        method = bench_cfg.get("method", "WMMSE_EIG").upper()
        method_params = dict(bench_cfg.get("WMMSE_EIG_Params", {}))
        method_params.update(common_params)

        if "SAA_Params" in bench_cfg:
            method_params["SAA_Params"] = dict(bench_cfg.get("SAA_Params", {}))

        H_1_Ric = u.k_term_los_1 * u.h1_los_est + u.k_term_nlos_1 * u.h1_nlos_est
        H_2_Ric = u.k_term_los_2 * u.h2_los_est + u.k_term_nlos_2 * u.h2_nlos_est
        H_1_est = u.sqrt_path_loss_1 * H_1_Ric
        H_2_est = u.sqrt_path_loss_2_vec[np.newaxis, :] * H_2_Ric
        sigma2  = float(u.awgn_power_watts)

        if method == "WMMSE_EIG_SAA":
            opt = AO_WMMSE_Eig_SAA(env_cfg, method_params, env_u=u, verbose=verbose)
        else:
            opt = AO_WMMSE_Eig(env_cfg, method_params, verbose=verbose)

        G, phi_vec = opt.optimize(H_1_est, H_2_est, sigma2, seed=int(episode_seed))
        results = env.unwrapped._calculate_robust_reward(G, phi_vec)
        
        env.close()
        return (float(results["avg_reward"]), float(results["avg_sum_rate"]))

    except Exception as e:
        env.close()
        print(f"Error in episode {episode_seed}: {e}")
        return (np.nan, np.nan)

# -------------------- Main Driver --------------------
def main():
    cfg_path = "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    bench = cfg.get("benchmark_settings", {})
    if not bench.get("enable", False):
        print("Benchmark testing is disabled in config.yaml.")
        return

    env_cfg = cfg["env_parameters"]
    run_cfg = cfg.get("run_settings", {})
    eval_seed = int(run_cfg.get("eval_seed", 42))

    rng = np.random.default_rng(eval_seed)
    num_eps = int(bench.get("num_episodes", 100))
    ep_seeds = rng.integers(low=0, high=2**31 - 1, size=num_eps, dtype=np.int64)

    default_workers = max(1, (os.cpu_count() or 2) - 1)
    workers = int(bench.get("parallel_workers", default_workers))
    workers = max(1, min(workers, num_eps))
    method = bench.get("method", "WMMSE_EIG").upper()
    
    print("=====================================================")
    print("--- Starting Classical Benchmark Test (Parallel) ---")
    print(f"Method:                 {method}")
    print(f"Episodes:               {num_eps}")
    print(f"Parallel Workers:       {workers}")
    print("=====================================================")

    jobs = [(int(s), dict(env_cfg), dict(bench)) for s in ep_seeds]
    results_R, results_SR = [], []

    all_results = []
    if workers > 1:
        with mp.Pool(workers, initializer=_init_worker) as pool:
            all_results = list(tqdm(pool.imap(run_single_episode, jobs), total=num_eps, desc="Running Benchmark"))
    else:
        all_results = [run_single_episode(j) for j in tqdm(jobs, desc="Running Benchmark (Sequential)")]

    for res in all_results:
        if res[0] is not np.nan:
            results_R.append(res[0])
            results_SR.append(res[1])

    print("\n--- Benchmark Results ---")
    if results_R and results_SR:
        R = np.array(results_R, dtype=np.float64)
        SR = np.array(results_SR, dtype=np.float64)
        
        print(f"Avg Reward:   {R.mean():.6f} ± {R.std(ddof=1):.6f}")
        print(f"Avg Sum-Rate: {SR.mean():.6f} ± {SR.std(ddof=1):.6f}")
    else:
        print("No episodes were successfully completed.")
    print("=====================================================")

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()