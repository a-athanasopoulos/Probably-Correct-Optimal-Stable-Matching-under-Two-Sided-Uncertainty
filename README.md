# README

This repository contains the code accompanying the paper **Probably Correct Optimal Stable Matching under Two-Sided Uncertainty**, submitted to **UAI** (anonymous authors).

## Repository structure

```
PACOS/
├─ algorithms/
│  ├─ learning/
│  │  ├─ extra_utils/
│  │  │  ├─ __init__.py
│  │  │  ├─ utils.py
│  │  │  └─ utils_partial_pref.py
│  │  ├─ __init__.py
│  │  ├─ agregator.py
│  │  ├─ BasePac.py
│  │  ├─ E_NS.py
│  │  ├─ E_psm.py
│  │  ├─ EE_psm.py
│  │  ├─ U.py
│  │  ├─ U_Delta.py
│  │  ├─ U_psm.py
│  │  └─ utils.py
│  └─ matching_algo/
│     └─ __init__.py
└─ exp/
   ├─ exp_1/
   ├─ exp_2/
   └─ __init__.py
```

### `algorithms/learning/`

Core implementations of the learning algorithms used in the paper.

### `algorithms/matching_algo/`

Matching-related primitives (PSM/POSM integration and other matching subroutines).

### `exp/`

Experiment drivers and configurations.

* `exp_1/`: scripts/configs for the first experiment of Section 4.
* `exp_2/`: scripts/configs for the last experiment of Section 7.

## Installation

Create a virtual environment and install dependencies (if you maintain a `requirements.txt`):

```bash
pip install -r requirements.txt
```

## Running Experiments

### Experiment 1: Baseline Comparisons
The first set of simulations from **Section 4** can be run using the Jupyter notebook:
`exp_1_compare_etc.ipynb`
---

### Experiment 2: Market Size Scaling
To generate the plots for **Section 7**, follow these three steps:

#### Step 1: Generate Instances
Create the random instances using `generate_instances.ipynb`.
* This script saves a `.pkl` file with instances for different settings of $\Delta_{min}$ and various market sizes.
* **Storage Path:** `./exp_2/instances`

#### Step 2: Run Simulations
Run the simulation script `exp_final.py`.
> **Note:** Ensure your relative paths are correctly configured before execution.
* Results are saved by default to: `./exp_2/results`

#### Step 3: Generate Plots
Run `plot_final.ipynb` to process the simulation data and generate the final figures.
