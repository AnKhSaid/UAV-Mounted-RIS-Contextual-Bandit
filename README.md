# UAV-Mounted RIS Optimization via Constrained Contextual Bandits

This repository implements a **Deep Reinforcement Learning (DRL)** framework for optimizing a UAV-mounted Reconfigurable Intelligent Surface (RIS). It utilizes **Constrained Contextual Bandits** (specifically modified TD3 and DDPG agents) to maximize sum-rate performance under robust channel conditions, accounting for specific challenges like **UAV Jitter** and **CSI (Channel State Information) Errors**.

The approach integrates a differentiable safety layer to strictly enforce beamforming power constraints and RIS unit-modulus constraints during the learning process.

## Citation

If you use this code in your research, please cite our preprint available on arXiv:

**[Preprint] Robust Optimization for UAV-Mounted RIS: A Constrained Contextual Bandit Approach** [https://arxiv.org/abs/2512.24773](https://arxiv.org/abs/2512.24773)


---

## Repository Structure



These scripts are **mutually independent**. Run the one corresponding to the experiment you want to perform. They all rely on the **Core Modules** and `config.yaml`.

**`main.py`**: Trains a single agent based on the current `config.yaml` settings.

**`figures_jitter.py`**: **Jitter Sweep:** Runs parallel training sessions sweeping over UAV jitter values ($\gamma=0$ to $10^\circ$). Generates robustness plots specifically for jitter analysis. Aggregates 10 seeded sessions for the results. 

**`figures_rho.py`**: **CSI Error Sweep:** Runs parallel training sessions sweeping over the correlation coefficient $\rho$.Generates robustness plots specifically for channel estimation error. Aggregates 10 seeded sessions for the results.

**`figures_combined.py`**: **Combined Robustness:** Sweeps over mixed error levels (simultaneous Jitter and CSI error) to produce robustness figures. Aggregates 10 seeded sessions for the results.

**`oracle.py`**: **Benchmarks:** Runs classical optimization baselines (**AO-WMMSE** and **SAA**) using CPU multiprocessing. No DRL involved.

### Core Modules

These files provide the logic and environment.

#### Configuration & Environment
* **`config.yaml`**: Central control file. Contains all hyperparameters.
* **`my_env.py`**: Defines the Gymnasium environment `UAV-RIS-v0`. Implements the system model.
* **`env_registration.py`**: Registers the custom environment with Gymnasium.

#### Agents & Algorithms
* **`TD3.py`**: Custom **TD3** implementation modified for Contextual Bandits ($\gamma=0$) with a **Differentiable Safety Layer** for action projection.
* **`ddpg.py`**: Custom **DDPG** implementation with equivalent safety constraints.
* **`classical_optimizers.py`**: Implementations of **AO-WMMSE** and **SAA** for the Oracle runner.
* **`model_builder.py`**: Factory script that initializes the correct agent (TD3 vs DDPG) and injects dependencies.

#### Utilities
* **`Plotting.py`**: Tools for generating training curves, bar charts, and JSON summaries.

---
## Installation & Usage

### 1. Prerequisites

Ensure you have Python 3.8+ installed. Install the dependencies using the requirements file:

```bash
pip install -r requirements.txt
```

### 2. Running Experiments

**To run a standard training session:**

1. Modify `config.yaml` to set your desired parameters.
2. Run:
```bash
python main.py
```

**To reproduce paper figures (Robustness Analysis):** These scripts automatically spawn parallel workers (CPU for baselines, GPU for DRL).

```bash
# For Jitter robustness analysis
python figures_jitter.py

# For CSI Error (Rho) analysis
python figures_rho.py

# For Combined Error analysis
python figures_combined.py
```

**To run Classical Benchmarks (Oracle):**

1. Ensure `benchmark_settings: enable: True` is set in `config.yaml`.
2. Run:
```bash
python oracle.py
```
