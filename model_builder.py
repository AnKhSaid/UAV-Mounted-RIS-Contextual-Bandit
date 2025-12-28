from typing import Any, Dict, Union, Tuple, Optional
import numpy as np
from stable_baselines3.common.noise import ActionNoise
from TD3 import ConstrainedTD3, ConstrainedTD3Policy
from ddpg import ConstrainedDDPG, ConstrainedDDPGPolicy


# -------------------------
# Hybrid Gaussian (BF vs RIS)
# -------------------------
class HybridGaussianNoise(ActionNoise):
    """
    Independent Gaussian noise for the beamformer and RIS parts.
    Sizes inferred from env_params.
    """
    def __init__(self, env_params: dict, sigma_bf: float = 0.1, sigma_ris: float = 0.1):
        super().__init__()
        self.n_bf = 2 * int(env_params["BS_antennas"]) * int(env_params["num_users"])
        self.n_ris_params = 2 * int(env_params["RIS_elements"])
        
        self.sigma_bf = float(sigma_bf)
        self.sigma_ris = float(sigma_ris)

    def __call__(self) -> np.ndarray:
        n = self.n_bf + self.n_ris_params
        noise = np.zeros(n, dtype=np.float32)
        if self.n_bf > 0:
            noise[: self.n_bf] = np.random.normal(0.0, self.sigma_bf, size=self.n_bf)
        if self.n_ris_params > 0:
            noise[self.n_bf :] = np.random.normal(0.0, self.sigma_ris, size=self.n_ris_params)
        return noise

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"HybridGaussianNoise(sigma_bf={self.sigma_bf}, sigma_ris={self.sigma_ris})"


# -------------------------
# Helpers
# -------------------------
def _as_tuple_train_freq(train_freq: Union[int, Tuple[int, str], list]) -> Union[int, Tuple[int, str]]:
    if isinstance(train_freq, list):
        return (int(train_freq[0]), str(train_freq[1]))
    return train_freq


def _make_internal_noise(env_params: Dict[str, Any], params: Dict[str, Any]) -> Optional[ActionNoise]:
    an = params.get("action_noise", {})
    sigma_bf = float(an.get("sigma_bf", 0.0))
    sigma_ris = float(an.get("sigma_ris", 0.0))
    if sigma_bf == 0.0 and sigma_ris == 0.0:
        return None
    return HybridGaussianNoise(env_params, sigma_bf=sigma_bf, sigma_ris=sigma_ris)


# -------------------------
# Public builder
# -------------------------
def build_model(
    model_type: str,
    env,
    seed: Optional[int],
    log_dir: Optional[str],
    device: str,
    verbose: int,
    model_hyperparams: Dict[str, Any],
    env_params: Dict[str, Any],
):
    """
    Build a contextual bandit agent (TD3 or DDPG style).
    """
    params = model_hyperparams
    train_freq = _as_tuple_train_freq(params.get("train_freq", (1, "step")))
    internal_noise = _make_internal_noise(env_params, params)

    # Pass env_params + internal noise object into the policy
    pk = dict(params.get("policy_kwargs", {}))
    pk["env_params"] = env_params
    pk["action_noise_obj"] = internal_noise

    if model_type.upper() == "TD3":
        policy_delay = int(params.get("policy_delay", 2))

        return ConstrainedTD3(
            policy=ConstrainedTD3Policy,
            env=env,
            seed=seed,
            tensorboard_log=log_dir,
            device=device,
            verbose=verbose,
            learning_rate=params.get("learning_rate", 1e-3),
            batch_size=int(params.get("batch_size", 100)),
            buffer_size=int(params.get("buffer_size", 1_000_000)),
            learning_starts=int(params.get("learning_starts", 1000)),
            train_freq=train_freq,
            gradient_steps=int(params.get("gradient_steps", -1)),
            action_noise=None, # Internal noise used instead
            policy_kwargs=pk,
            policy_delay=policy_delay,
        )

    elif model_type.upper() == "DDPG":
        return ConstrainedDDPG(
            policy=ConstrainedDDPGPolicy,
            env=env,
            seed=seed,
            tensorboard_log=log_dir,
            device=device,
            verbose=verbose,
            learning_rate=params.get("learning_rate", 1e-3),
            batch_size=int(params.get("batch_size", 100)),
            buffer_size=int(params.get("buffer_size", 1_000_000)),
            learning_starts=int(params.get("learning_starts", 1000)),
            train_freq=train_freq,
            gradient_steps=int(params.get("gradient_steps", 1)),
            action_noise=None,
            policy_kwargs=pk,
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'TD3' or 'DDPG'.")