from __future__ import annotations
import numpy as np

# ------------------------------- Utilities -----------------------------------

def _as_c64(x):
    return np.asarray(x, dtype=np.complex64)

def _as_f32(x):
    return np.asarray(x, dtype=np.float32)

def _project_unit_modulus(v: np.ndarray) -> np.ndarray:
    return np.exp(1j * np.angle(v)).astype(np.complex64)

def _fro2(x: np.ndarray) -> float:
    return float(np.real(np.vdot(x, x)))

# ------------------------------ Rate Helper ----------------------------------

def _sum_rate_bits(Heff: np.ndarray, G: np.ndarray, sigma2: float) -> float:
    S = Heff @ G
    desired = np.abs(np.diag(S))**2
    total = np.sum(np.abs(S)**2, axis=1)
    interf = total - desired
    sinr = desired / (interf + sigma2)
    return float(np.sum(np.log2(1.0 + sinr)))

# ---------------------------- Channel Builders --------------------------------

def _build_Heff_from_cascaded(H_cascaded: np.ndarray, v: np.ndarray) -> np.ndarray:
    # Heff[k,:] = v^T H_k
    Heff = np.einsum('knm,n->km', H_cascaded, v, optimize=True)
    return _as_c64(Heff)

def _build_cascaded_from_est(H1_est: np.ndarray, H2_est: np.ndarray) -> np.ndarray:
    N, M = H1_est.shape
    K = H2_est.shape[1]
    H = np.zeros((K, N, M), dtype=np.complex64)
    for k in range(K):
        H[k] = (np.conj(H2_est[:, k])[:, None] * H1_est)
    return H

# ---------------------------- WMMSE Primitives --------------------------------

def _wmmse_receivers_weights(Heff: np.ndarray, G: np.ndarray, sigma2: float):
    S = Heff @ G
    sig = np.diag(S)
    p_tot = np.sum(np.abs(S)**2, axis=1) + sigma2

    u = (np.conj(sig) / p_tot).astype(np.complex64)
    e = 1.0 - 2.0 * np.real(u * sig) + (np.abs(u)**2) * p_tot
    e = np.maximum(e, 1e-12)
    w = (1.0 / e).astype(np.float32)
    return _as_c64(u), _as_f32(w)

def _wmmse_precoder_update(Heff: np.ndarray, u: np.ndarray, w: np.ndarray,
                           Pmax: float, lam_hi: float = 1e6, n_bisect: int = 30) -> np.ndarray:
    K, M = Heff.shape[0], Heff.shape[1]
    U = np.diag(u)
    W = np.diag(w)

    Hw  = Heff.conj().T @ U.conj().T @ W
    HUH = Heff.conj().T @ U.conj().T @ W @ U @ Heff

    def solve_for_lambda(lmbd: float) -> np.ndarray:
        A = HUH + (lmbd * np.eye(M, dtype=np.complex64))
        G = np.linalg.solve(A, Hw).astype(np.complex64)
        return G

    G0 = solve_for_lambda(0.0)
    pow0 = _fro2(G0)
    if pow0 <= Pmax:
        if pow0 > 1e-12:
            G0 *= np.sqrt(Pmax / pow0)
        return G0

    lam_lo, lam_hi_local = 0.0, lam_hi
    G = G0
    for _ in range(n_bisect):
        lam_mid = 0.5 * (lam_lo + lam_hi_local)
        G = solve_for_lambda(lam_mid)
        powG = _fro2(G)
        if powG > Pmax:
            lam_lo = lam_mid
        else:
            lam_hi_local = lam_mid

    powG = _fro2(G)
    if powG > 1e-12:
        G *= np.sqrt(Pmax / powG)
    return G

# ------------------------------ AO: EIG Method --------------------------------

class AO_WMMSE_Eig:
    """
    Alternating optimization: WMMSE (BS) + Eigen-heuristic (RIS).
    """
    def __init__(self, env_params: dict, params: dict, verbose: bool = False):
        self.verbose = verbose
        self.Pmax = float(env_params.get("P_max", 1.0))
        self.ao_max = int(params.get("ao_max", 100))
        self.wmmse_inner = int(params.get("wmmse_inner", 12))
        self.wmmse_bisection_steps = int(params.get("wmmse_bisection_steps", 30))
        self.wmmse_bisection_lam_high = float(params.get("wmmse_bisection_lam_high", 1e6))

    def optimize(self, H1_est: np.ndarray, H2_est: np.ndarray, sigma2: float, seed: int = 0):
        rng = np.random.default_rng(int(seed))
        N, M = H1_est.shape
        K = H2_est.shape[1]
        Hc_nom = _build_cascaded_from_est(_as_c64(H1_est), _as_c64(H2_est))

        v = np.ones((N,), dtype=np.complex64)
        G = (1/np.sqrt(M)) * (rng.standard_normal((M, K)) + 1j * rng.standard_normal((M, K))).astype(np.complex64)
        G *= np.sqrt(self.Pmax / max(_fro2(G), 1e-12))

        if self.verbose:
            print("--- Starting AO WMMSE-Eig Optimization ---")

        for it in range(self.ao_max):
            # BS WMMSE
            Heff = _build_Heff_from_cascaded(Hc_nom, v)
            for _ in range(self.wmmse_inner):
                u, w = _wmmse_receivers_weights(Heff, G, sigma2)
                G = _wmmse_precoder_update(Heff, u, w, self.Pmax,
                                           lam_hi=self.wmmse_bisection_lam_high,
                                           n_bisect=self.wmmse_bisection_steps)

            # RIS eigen-heuristic
            A = np.zeros((N, N), dtype=np.complex64)
            for k in range(K):
                Hk = Hc_nom[k]
                for j in range(K):
                    a_kj = Hk @ G[:, j]
                    A += np.outer(a_kj, a_kj.conj())
            A = 0.5 * (A + A.conj().T)
            _, vecs = np.linalg.eigh(A.astype(np.complex128))
            v = _project_unit_modulus(np.conj(vecs[:, -1]))

            if self.verbose:
                Heff_nom = _build_Heff_from_cascaded(Hc_nom, v)
                sr_nom = _sum_rate_bits(Heff_nom, G, sigma2)
                print(f"[AO Iter {it+1:03d}] Rate: {sr_nom:.6f}")

        return _as_c64(G), _as_c64(v)
    

# ------------------------------ AO: EIG + SAA (Robust) -----------------------

class AO_WMMSE_Eig_SAA:
    """
    Robust Alternating Optimization via Sample Average Approximation (SAA).
    """
    def __init__(self, env_params: dict, params: dict, env_u, verbose: bool = False):
        self.verbose = verbose
        self.Pmax = float(env_params.get("P_max", 1.0))
        self.ao_max = int(params.get("ao_max", 100))
        self.wmmse_inner = int(params.get("wmmse_inner", 12))
        self.wmmse_bisection_steps = int(params.get("wmmse_bisection_steps", 30))
        self.wmmse_bisection_lam_high = float(params.get("wmmse_bisection_lam_high", 1e6))
        
        saa = params.get("SAA_Params", {})
        self.num_scenarios = int(saa.get("num_scenarios", 16))
        self.resample_each_iter = bool(saa.get("resample_each_iter", True))

        self.u = env_u
        pl1 = float(self.u.sqrt_path_loss_1**2)
        pl2_vec = (self.u.sqrt_path_loss_2_vec**2).astype(np.float32)
        self._err_std = np.sqrt((pl1 * pl2_vec)[:, None, None] / 2.0).astype(np.float32)
        self._sqrt_rho_term = np.float32(np.sqrt(1.0 - self.u.cascaded_error_rho**2))

    def _Heff_from(self, Hc, v):
        return np.einsum('knm,n->km', Hc, v, optimize=True).astype(np.complex64)

    def _sum_rate_bits(self, Heff, G, sigma2):
        S = Heff @ G
        desired = np.abs(np.diag(S))**2
        total = np.sum(np.abs(S)**2, axis=1)
        interf = total - desired
        sinr = desired / (interf + sigma2)
        return float(np.sum(np.log2(1.0 + sinr)))

    def _apply_jitter_once(self, rng):
        u = self.u
        roll, pitch, yaw = rng.normal(0.0, float(u.uav_jitter_std_rad), 3)
        c_r, s_r = np.cos(roll), np.sin(roll)
        c_p, s_p = np.cos(pitch), np.sin(pitch)
        c_y, s_y = np.cos(yaw), np.sin(yaw)
        rot_x = np.array([[1,0,0],[0,c_r,-s_r],[0,s_r,c_r]], dtype=np.float32)
        rot_y = np.array([[c_p,0,s_p],[0,1,0],[-s_p,0,c_p]], dtype=np.float32)
        rot_z = np.array([[c_y,-s_y,0],[s_y,c_y,0],[0,0,1]], dtype=np.float32)
        R = rot_z @ rot_y @ rot_x

        elems = u.ris_elem_pos @ R.T
        proj1 = elems @ u.unit_vec_bs_to_ris.T
        a_ris_rx_los_true = np.exp(-1j * u.k0_wavenumber * proj1).astype(np.complex64).flatten()
        a_bs_tx_los_true = u.a_bs_tx_los_nominal
        h1_los_true = np.outer(a_ris_rx_los_true, a_bs_tx_los_true.conj())

        proj2 = elems @ u.unit_vecs_ris_to_users.T
        h2_los_true = np.exp(-1j * u.k0_wavenumber * proj2).astype(np.complex64)
        return h1_los_true, h2_los_true

    def _sample_cascaded_channels(self, rng):
        u = self.u
        S = self.num_scenarios
        out = []
        for _ in range(S):
            if float(u.uav_jitter_std_rad) > 0.0:
                h1_los_true, h2_los_true = self._apply_jitter_once(rng)
            else:
                h1_los_true, h2_los_true = u.h1_los_est, u.h2_los_est

            h1 = u.k_term_los_1 * h1_los_true + u.k_term_nlos_1 * u.h1_nlos_est
            h2 = u.k_term_los_2 * h2_los_true + u.k_term_nlos_2 * u.h2_nlos_est
            h1 = u.sqrt_path_loss_1 * h1
            h2 = u.sqrt_path_loss_2_vec[np.newaxis, :] * h2

            Hc_true = (h1[None, :, :] * h2.T.conj()[:, :, None]).astype(np.complex64)
            noise_real = self._err_std * rng.standard_normal(size=Hc_true.shape).astype(np.float32)
            noise_imag = self._err_std * rng.standard_normal(size=Hc_true.shape).astype(np.float32)
            E = noise_real + 1j * noise_imag
            Hc = (u.cascaded_error_rho * Hc_true + self._sqrt_rho_term * E).astype(np.complex64)
            out.append(Hc)
        return out

    def _wmmse_receivers_weights(self, Heff, G, sigma2):
        S = Heff @ G
        sig = np.diag(S)
        p_tot = np.sum(np.abs(S)**2, axis=1) + sigma2
        u = (np.conj(sig) / p_tot).astype(np.complex64)
        e = 1.0 - 2.0 * np.real(u * sig) + (np.abs(u)**2) * p_tot
        e = np.maximum(e, 1e-12)
        w = (1.0 / e).astype(np.float32)
        return u, w

    def _precoder_update_saa(self, Heff_list, U_list, W_list):
        M = Heff_list[0].shape[1]
        Hw = np.zeros((M, len(U_list[0])), dtype=np.complex64)
        HUH = np.zeros((M, M), dtype=np.complex64)
        S = float(len(Heff_list))
        for Heff, u, w in zip(Heff_list, U_list, W_list):
            U = np.diag(u)
            W = np.diag(w)
            Hw += Heff.conj().T @ U.conj().T @ W
            HUH += Heff.conj().T @ U.conj().T @ W @ U @ Heff
        Hw /= S
        HUH /= S

        def solve_for_lambda(lmbd: float):
            A = HUH + (lmbd * np.eye(M, dtype=np.complex64))
            return np.linalg.solve(A, Hw).astype(np.complex64)

        G = solve_for_lambda(0.0)
        powG = float(np.real(np.vdot(G, G)))
        if powG <= self.Pmax and powG > 1e-12:
            G *= np.sqrt(self.Pmax / powG)
            return G
        
        lam_lo, lam_hi = 0.0, self.wmmse_bisection_lam_high
        for _ in range(self.wmmse_bisection_steps):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            G = solve_for_lambda(lam_mid)
            powG = float(np.real(np.vdot(G, G)))
            if powG > self.Pmax:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
        
        if powG > 1e-12:
            G *= np.sqrt(self.Pmax / powG)
        return G

    def optimize(self, H1_est: np.ndarray, H2_est: np.ndarray, sigma2: float, seed: int = 0):
        rng = np.random.default_rng(int(seed))
        N, M = H1_est.shape
        K = H2_est.shape[1]

        v = np.ones((N,), dtype=np.complex64)
        G = (1/np.sqrt(M)) * (rng.standard_normal((M, K)) + 1j * rng.standard_normal((M, K))).astype(np.complex64)
        G *= np.sqrt(self.Pmax / max(float(np.real(np.vdot(G, G))), 1e-12))

        samples = self._sample_cascaded_channels(rng)

        for it in range(self.ao_max):
            if self.resample_each_iter and it > 0:
                samples = self._sample_cascaded_channels(rng)

            # SAA BS WMMSE
            Heff_list = [self._Heff_from(Hc, v) for Hc in samples]
            for _ in range(self.wmmse_inner):
                U_list, W_list = zip(*(self._wmmse_receivers_weights(Heff, G, sigma2) for Heff in Heff_list))
                G = self._precoder_update_saa(Heff_list, list(U_list), list(W_list))

            # SAA RIS Eigen
            A_total = np.zeros((N, N), dtype=np.complex64)
            for Hc in samples:
                As = np.zeros((N, N), dtype=np.complex64)
                for k in range(K):
                    Hk = Hc[k]
                    for j in range(K):
                        a = Hk @ G[:, j]
                        As += np.outer(a, a.conj())
                A_total += 0.5 * (As + As.conj().T)
            A_total /= float(len(samples))
            _, vecs = np.linalg.eigh(A_total.astype(np.complex128))
            v = _project_unit_modulus(np.conj(vecs[:, -1]))

            if self.verbose:
                print(f"[AO-SAA {it+1:03d}] Optimization Step Complete")

        return G.astype(np.complex64), v.astype(np.complex64)