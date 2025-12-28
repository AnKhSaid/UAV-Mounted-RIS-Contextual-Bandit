from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.td3.policies import TD3Policy


class DifferentiableSafetyLayer(nn.Module):
    """
    Projects actions onto the feasible set:
      1) Beamformer power: ||g||^2 <= P_max
      2) RIS: unit-modulus (per element)
    """

    def __init__(self, bf_dims: int, ris_dims: int, p_max: float, eps: float = 1e-8):
        super().__init__()
        self.bf_dims = int(bf_dims)
        self.ris_dims = int(ris_dims)
        self.register_buffer("p_max", torch.tensor(p_max, dtype=torch.float32))
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float32))

    def forward(self, raw_action: torch.Tensor) -> torch.Tensor:
        bf = raw_action[..., : self.bf_dims]
        ris = raw_action[..., self.bf_dims :]
        bf_proj = self._project_beamformer(bf)
        ris_proj = self._project_ris(ris)
        return torch.cat([bf_proj, ris_proj], dim=-1)

    def _project_beamformer(self, bf_action: torch.Tensor) -> torch.Tensor:
        power = torch.sum(bf_action ** 2, dim=-1, keepdim=True)
        scale = torch.minimum(
            torch.ones_like(power), torch.sqrt(self.p_max / (power + self.eps))
        )
        return bf_action * scale

    def _project_ris(self, ris_action: torch.Tensor) -> torch.Tensor:
        if self.ris_dims == 0:
            return ris_action
        bsz = ris_action.shape[0]
        n = self.ris_dims // 2
        ris2 = ris_action.view(bsz, n, 2)
        norms = torch.norm(ris2, dim=-1, keepdim=True)
        return (ris2 / (norms + self.eps)).view(bsz, -1)


class ConstrainedActor(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        env_params: Dict,
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        action_noise_obj: Optional[ActionNoise] = None,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.normalize_images = normalize_images
        self.action_noise = action_noise_obj

        action_dim = int(np.prod(action_space.shape))
        layers: List[nn.Module] = []
        last = int(features_dim)
        for h in net_arch:
            layers += [nn.Linear(last, h), activation_fn()]
            last = h
        layers += [nn.Linear(last, action_dim)]
        self.mu = nn.Sequential(*layers)

        bf_dims = 2 * env_params["BS_antennas"] * env_params["num_users"]
        ris_dims = 2 * env_params["RIS_elements"]

        self.safety_layer = DifferentiableSafetyLayer(
            bf_dims=bf_dims,
            ris_dims=ris_dims,
            p_max=env_params["P_max"],
        )

    def set_training_mode(self, mode: bool):
        self.train(mode)
        self.features_extractor.train(mode)
        self.mu.train(mode)
        return self

    @torch.no_grad()
    def _sample_noise(self, batch: int, act_dim: int, device, dtype) -> Optional[torch.Tensor]:
        if self.action_noise is None:
            return None
        if batch <= 1:
            n = np.asarray(self.action_noise(), dtype=np.float32)
            return torch.as_tensor(n, device=device, dtype=dtype).view(1, act_dim)
        arr = [np.asarray(self.action_noise(), dtype=np.float32) for _ in range(batch)]
        n = np.stack(arr, axis=0)
        return torch.as_tensor(n, device=device, dtype=dtype)

    def forward(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        return_pre_projection: bool = False,
    ) -> torch.Tensor:
        z = self.features_extractor(observation)
        raw_mu = self.mu(z)

        raw_for_projection = raw_mu
        if not deterministic:
            noise = self._sample_noise(
                batch=raw_mu.shape[0],
                act_dim=raw_mu.shape[1],
                device=raw_mu.device,
                dtype=raw_mu.dtype,
            )
            if noise is not None:
                raw_for_projection = raw_mu + noise

        projected_action = self.safety_layer(raw_for_projection)

        if return_pre_projection:
            return projected_action, raw_mu
        else:
            return projected_action

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.forward(observation, deterministic=deterministic)


class ConstrainedTD3Policy(TD3Policy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        env_params: Dict,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        action_noise_obj: Optional[ActionNoise] = None,
    ):
        self.env_params = env_params
        self._action_noise_obj = action_noise_obj
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> ConstrainedActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return ConstrainedActor(
            **actor_kwargs,
            env_params=self.env_params,
            action_noise_obj=self._action_noise_obj,
        ).to(self.device)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.actor(observation, deterministic=deterministic)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)


class ConstrainedTD3(TD3):
    """
    Contextual Bandit agent with safety projection and twin critics.
    This implementation ignores bootstrapping and targets Q(s,a) ≈ r(s,a).
    """

    def __init__(
        self,
        policy: Union[str, Type[ConstrainedTD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 100,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, any]] = None,
        policy_delay: int = 2,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=0.0,
            gamma=0.0,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_delay=policy_delay,
            target_policy_noise=0.0,
            target_noise_clip=0.0,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    @staticmethod
    def _as_list_q(q_out):
        if isinstance(q_out, (list, tuple)):
            return list(q_out)
        return [q_out]

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(
            [self.policy.actor.optimizer, self.policy.critic.optimizer]
        )

        for _ in range(gradient_steps):
            self._n_updates += 1

            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            target_q = replay_data.rewards

            current_q_list = self._as_list_q(
                self.policy.critic(replay_data.observations, replay_data.actions)
            )
            critic_loss = sum(F.mse_loss(cq, target_q) for cq in current_q_list)
            
            self.policy.critic.optimizer.zero_grad()
            critic_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), 10.0)
            
            self.policy.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                actor_actions, _ = self.policy.actor(
                    replay_data.observations,
                    deterministic=True,
                    return_pre_projection=True,
                )

                q_actor_out = self.policy.critic(
                    replay_data.observations, actor_actions
                )
                q1 = self._as_list_q(q_actor_out)[0]

                actor_loss = -q1.mean()

                self.policy.actor.optimizer.zero_grad()
                actor_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 5.0)
                
                self.policy.actor.optimizer.step()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")