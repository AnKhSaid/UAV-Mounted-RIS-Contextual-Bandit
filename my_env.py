import gymnasium as gym
from gymnasium import spaces
import numpy as np

class UAVRISEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, **env_parameters):
        super().__init__()
        self.__dict__.update(env_parameters)

        # Parameter mapping
        self.p_max = self.P_max
        self.k_factor_1 = self.K_factor_1
        self.k_factor_2 = self.K_factor_2
        self.alpha_1 = self.alpha_1
        self.alpha_2 = self.alpha_2
        self.eta_nlos = self.eta_NLoS
        self.kappa_2 = self.kappa_2
        self.sigma_jitt_deg = self.sigma_jitt_deg
        self.cascaded_error_rho = self.cascaded_error_rho
        self.p_bs = self.p_BS
        self.q_uav = self.q_UAV
        self.dist_norm_factor = env_parameters.get('distance_normalization_factor', 100.0)

        # Dimensions and Types
        self.M = self.BS_antennas
        self.N = self.RIS_elements
        self.K = self.num_users
        self.dtype_real = np.float32
        self.dtype_complex = np.complex64

        # Physics Constants
        self.uav_jitter_std_rad = self.dtype_real(np.deg2rad(self.sigma_jitt_deg))
        c = 3e8
        self.lambda_c = c / self.carrier_frequency
        self.k0_wavenumber = self.dtype_real(2 * np.pi / self.lambda_c)
        self.awgn_power_watts = self.dtype_real(10**(self.awgn_power_dbw / 10))

        # Rician K-Factors
        self.k_term_los_1 = self.dtype_real(np.sqrt(self.k_factor_1 / (self.k_factor_1 + 1)))
        self.k_term_nlos_1 = self.dtype_real(np.sqrt(1 / (self.k_factor_1 + 1)))
        self.k_term_los_2 = self.dtype_real(np.sqrt(self.k_factor_2 / (self.k_factor_2 + 1)))
        self.k_term_nlos_2 = self.dtype_real(np.sqrt(1 / (self.k_factor_2 + 1)))

        # Antenna and RIS Geometry
        d_bs = self.bs_antenna_spacing * self.lambda_c
        self.bs_ant_pos = np.zeros((self.M, 3), dtype=self.dtype_real)
        self.bs_ant_pos[:, 2] = np.arange(self.M, dtype=self.dtype_real) * d_bs
        self.bs_ant_pos -= self.bs_ant_pos.mean(axis=0)

        d_ris = self.ris_element_spacing * self.lambda_c
        self.ris_elem_pos = np.zeros((self.N, 3), dtype=self.dtype_real)
        ris_elements_per_side = int(np.sqrt(self.N))
        if ris_elements_per_side**2 != self.N:
            raise ValueError(f"RIS_elements (N={self.N}) must be a perfect square.")
        
        indices = np.arange(self.N, dtype=np.int32)
        self.ris_elem_pos[:, 0] = (indices % ris_elements_per_side) * d_ris
        self.ris_elem_pos[:, 1] = (indices // ris_elements_per_side) * d_ris
        self.ris_elem_pos = self.ris_elem_pos.astype(self.dtype_real, copy=False)
        self.ris_elem_pos -= self.ris_elem_pos.mean(axis=0)

        # Space Definitions
        bf_action_dim = 2 * self.M * self.K
        ris_action_dim = 2 * self.N
        self.action_dim = bf_action_dim + ris_action_dim
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=self.dtype_real)

        channel_obs_dim = 2 * (self.N * self.M + self.N * self.K)
        obs_dim = channel_obs_dim + self.K
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=self.dtype_real)

        # Buffers
        self.h_cascaded_true_buffer = np.empty((self.K, self.N, self.M), dtype=self.dtype_complex)
        self.cascaded_error_buffer = np.empty((self.K, self.N, self.M), dtype=self.dtype_complex)
        self.stats_keys = ["reward", "sum_rate"]
        self.scalar_stats_buffer = np.empty((self.S_samples, len(self.stats_keys)), dtype=self.dtype_real)

        self._cached_observation = None

    def _get_obs(self):
        h1_rician_est = self.k_term_los_1 * self.h1_los_est + self.k_term_nlos_1 * self.h1_nlos_est
        h2_rician_est = self.k_term_los_2 * self.h2_los_est + self.k_term_nlos_2 * self.h2_nlos_est
        
        h1_channel_est = self.sqrt_path_loss_1 * h1_rician_est
        h2_channel_est = self.sqrt_path_loss_2_vec[np.newaxis, :] * h2_rician_est
        
        norms_1 = np.linalg.norm(h1_channel_est, axis=1, keepdims=True).astype(self.dtype_real)
        h1_normalized = h1_channel_est / norms_1

        norms_2 = np.linalg.norm(h2_channel_est, axis=1, keepdims=True).astype(self.dtype_real)
        h2_normalized = h2_channel_est / norms_2

        h1_obs_flat = np.concatenate([h1_normalized.real.flatten(), h1_normalized.imag.flatten()])
        h2_obs_flat = np.concatenate([h2_normalized.real.flatten(), h2_normalized.imag.flatten()])
        channel_obs = np.concatenate([h1_obs_flat, h2_obs_flat])
            
        distances_obs_normalized = self.dists_ris_to_users / self.dist_norm_factor
        final_obs = np.concatenate([distances_obs_normalized, channel_obs])
            
        return final_obs.astype(self.dtype_real)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        user_loc_indices = self.np_random.choice(len(self.possible_user_locations), size=self.K, replace=False)
        self.user_positions = np.asarray(self.possible_user_locations, dtype=self.dtype_real)[user_loc_indices]
        
        self._calculate_large_scale_components()
        self._generate_estimated_small_scale_channels()
        self._cached_observation = self._get_obs()
        
        return self._cached_observation, {}

    def step(self, action):
        bf_action_dim = 2 * self.M * self.K
        bf_action, ris_action = np.split(action, [bf_action_dim])
        
        bf_real, bf_imag = np.split(bf_action, 2)
        bf_matrix_g = (bf_real.reshape(self.M, self.K) + 1j * bf_imag.reshape(self.M, self.K)).astype(self.dtype_complex)
        
        ris_action_reshaped = ris_action.reshape(self.N, 2)
        ris_phases = (ris_action_reshaped[:, 0] + 1j * ris_action_reshaped[:, 1]).astype(self.dtype_complex)
        
        next_observation = self._cached_observation
        
        results = self._calculate_robust_reward(bf_matrix_g, ris_phases)
        reward = results["avg_reward"]
        
        info = {
            'sumrate': results["avg_sum_rate"]
        }
        
        return next_observation, reward, False, False, info

    def _calculate_steering_vector(self, element_positions, direction_vector):
        """Calculates the steering vector for given positions and directions."""
        proj = element_positions @ direction_vector.T
        return np.exp(-1j * self.k0_wavenumber * proj).astype(self.dtype_complex)

    def _calculate_large_scale_components(self):
        """Calculates distances, path loss, and LoS unit vectors."""
        vec_bs_to_ris = np.array(self.q_uav, dtype=self.dtype_real) - np.array(self.p_bs, dtype=self.dtype_real)
        self.dist_bs_to_ris = self.dtype_real(np.linalg.norm(vec_bs_to_ris))
        self.unit_vec_bs_to_ris = vec_bs_to_ris / self.dist_bs_to_ris
        
        elevation_arg = (self.q_uav[2] - self.p_bs[2]) / self.dist_bs_to_ris
        elevation_deg = np.rad2deg(np.arcsin(elevation_arg))
        prob_los_1 = 1 / (1 + self.urban_a * np.exp(-self.urban_b * (elevation_deg - self.urban_a)))
        
        path_loss_1 = (prob_los_1 + (1 - prob_los_1) * self.eta_nlos) * (self.dist_bs_to_ris**(-self.alpha_1))
        self.sqrt_path_loss_1 = self.dtype_real(np.sqrt(path_loss_1))
        
        vecs_ris_to_users = self.user_positions - np.array(self.q_uav, dtype=self.dtype_real)
        self.dists_ris_to_users = np.linalg.norm(vecs_ris_to_users, axis=1).astype(self.dtype_real)
        self.unit_vecs_ris_to_users = vecs_ris_to_users / self.dists_ris_to_users[:, np.newaxis]
        
        path_loss_2 = self.kappa_2 * (self.dists_ris_to_users**(-self.alpha_2))
        self.sqrt_path_loss_2_vec = self.dtype_real(np.sqrt(path_loss_2))

    def _generate_estimated_small_scale_channels(self):
        """Generates the estimated LoS and NLoS channel components."""
        a_ris_rx_los = self._calculate_steering_vector(self.ris_elem_pos, self.unit_vec_bs_to_ris).flatten()
        self.a_bs_tx_los_nominal = self._calculate_steering_vector(self.bs_ant_pos, self.unit_vec_bs_to_ris).flatten()
        
        self.h1_los_est = np.outer(a_ris_rx_los, self.a_bs_tx_los_nominal.conj())
        self.h2_los_est = self._calculate_steering_vector(self.ris_elem_pos, self.unit_vecs_ris_to_users)

        sqrt_2 = self.dtype_real(np.sqrt(2.0))
        self.h1_nlos_est = (self.np_random.standard_normal(size=(self.N, self.M), dtype=self.dtype_real) + 
                            1j * self.np_random.standard_normal(size=(self.N, self.M), dtype=self.dtype_real)
                           ).astype(self.dtype_complex) / sqrt_2
                           
        self.h2_nlos_est = (self.np_random.standard_normal(size=(self.N, self.K), dtype=self.dtype_real) + 
                            1j * self.np_random.standard_normal(size=(self.N, self.K), dtype=self.dtype_real)
                           ).astype(self.dtype_complex) / sqrt_2

    def _apply_jitter(self):
        """Applies a random rotation to the RIS and calculates new LoS channels."""
        roll, pitch, yaw = self.np_random.normal(0, self.uav_jitter_std_rad, 3)
        
        c_r, s_r = np.cos(roll), np.sin(roll)
        c_p, s_p = np.cos(pitch), np.sin(pitch)
        c_y, s_y = np.cos(yaw), np.sin(yaw)
        
        rot_x = np.array([[1,0,0],[0,c_r,-s_r],[0,s_r,c_r]], dtype=self.dtype_real)
        rot_y = np.array([[c_p,0,s_p],[0,1,0],[-s_p,0,c_p]], dtype=self.dtype_real)
        rot_z = np.array([[c_y,-s_y,0],[s_y,c_y,0],[0,0,1]], dtype=self.dtype_real)
        
        rotation_matrix = rot_z @ rot_y @ rot_x
        rotated_ris_elem_pos = self.ris_elem_pos @ rotation_matrix.T
        
        a_ris_rx_los_true = self._calculate_steering_vector(rotated_ris_elem_pos, self.unit_vec_bs_to_ris).flatten()
        a_bs_tx_los_true = self.a_bs_tx_los_nominal
        
        h1_los_true_sample = np.outer(a_ris_rx_los_true, a_bs_tx_los_true.conj())
        h2_los_true_sample = self._calculate_steering_vector(rotated_ris_elem_pos, self.unit_vecs_ris_to_users)
        
        return h1_los_true_sample, h2_los_true_sample

    def _calculate_robust_reward(self, bf_matrix_g, ris_phases):
        """Calculates the average sum-rate applying channel estimation errors and UAV jitter."""
        sqrt_rho_term = self.dtype_real(np.sqrt(1 - self.cascaded_error_rho**2))
        ris_phase_matrix = np.diag(ris_phases)

        path_loss_1_val = float(self.sqrt_path_loss_1**2)
        path_loss_2_vec = (self.sqrt_path_loss_2_vec**2).astype(self.dtype_real)
        error_variance_k = (path_loss_1_val * path_loss_2_vec)[:, None, None]
        error_std_dev = np.sqrt(error_variance_k / 2.0).astype(self.dtype_real)

        for s in range(self.S_samples):
            if self.uav_jitter_std_rad > 0:
                h1_los_true_sample, h2_los_true_sample = self._apply_jitter()
            else:
                h1_los_true_sample, h2_los_true_sample = self.h1_los_est, self.h2_los_est
            
            h1_jitt_rician = self.k_term_los_1 * h1_los_true_sample + self.k_term_nlos_1 * self.h1_nlos_est
            h2_jitt_rician = self.k_term_los_2 * h2_los_true_sample + self.k_term_nlos_2 * self.h2_nlos_est
            
            h1_jitt_channel = self.sqrt_path_loss_1 * h1_jitt_rician
            h2_jitt_channel = self.sqrt_path_loss_2_vec[np.newaxis, :] * h2_jitt_rician

            h_cascaded_jitt_true = h1_jitt_channel[None, :, :] * h2_jitt_channel.T.conj()[:, :, None]

            self.cascaded_error_buffer.real = error_std_dev * self.np_random.standard_normal(
                size=(self.K, self.N, self.M), dtype=self.dtype_real)
            self.cascaded_error_buffer.imag = error_std_dev * self.np_random.standard_normal(
                size=(self.K, self.N, self.M), dtype=self.dtype_real)
            
            np.multiply(self.cascaded_error_rho, h_cascaded_jitt_true, out=self.h_cascaded_true_buffer)
            self.h_cascaded_true_buffer += sqrt_rho_term * self.cascaded_error_buffer
            
            stats = self._calculate_sum_rate(bf_matrix_g, ris_phase_matrix, 
                                             H_cascaded_channel=self.h_cascaded_true_buffer)
            
            for i, key in enumerate(self.stats_keys):
                self.scalar_stats_buffer[s, i] = stats[key]

        avg_scalar_stats = np.mean(self.scalar_stats_buffer, axis=0)
        
        results = {
            "avg_reward": avg_scalar_stats[0], 
            "avg_sum_rate": avg_scalar_stats[1],
        }
        return results
    
    def _calculate_sum_rate(self, bf_matrix_g, ris_phase_matrix, H_cascaded_channel):
        """Calculates the sum-rate for a given set of channels and actions."""
        effective_channel_matrix = np.einsum('knm,nn,mj->kj', 
                                             H_cascaded_channel, 
                                             ris_phase_matrix, 
                                             bf_matrix_g, 
                                             dtype=self.dtype_complex)

        signal_power = np.abs(np.diag(effective_channel_matrix))**2
        total_received_power = np.sum(np.abs(effective_channel_matrix)**2, axis=1, dtype=self.dtype_real)
        interference_power = total_received_power - signal_power
        
        sinr = signal_power / (interference_power + self.awgn_power_watts)
        rates = np.log1p(sinr) / self.dtype_real(np.log(2.0))
        sum_rate = np.sum(rates, dtype=self.dtype_real)
        
        stats = {
            "reward": sum_rate, 
            "sum_rate": float(sum_rate)
        }
        return stats