# CLAUDE.md — Agent Background for ssEinterpolator

## What This Package Does

**ssEinterpolator** builds a **Reduced Order Model (ROM)** of slow slip event (SSE) simulations on the Cascadia subduction zone. The core problem: running full geophysical simulations is expensive, so this package creates a fast surrogate model that can predict simulation output for any new parameter combination by interpolating from a set of pre-run simulations.

**Input**: A grid of full numerical simulations, each run at different forward model parameters (e.g., friction law coefficients).
**Output**: A trained ROM that can predict the time series (slip rate, state variable, slip) for new parameter values without running new simulations.

---

## Scientific Domain

**Slow Slip Events (SSEs)** are episodes of aseismic fault creep on subduction zones (here, Cascadia). They occur episodically and are tracked via three physical quantities at each along-fault depth:
- **Slip rate** (`sr`): rate of fault slip, spans many orders of magnitude → log-transformed
- **State variable** (`state`): friction state variable from the rate-and-state friction law
- **Slip** (`slip`): cumulative fault displacement

Each simulation produces time series of these three quantities at ~200+ depth points. A single simulation run generates data shaped `[n_depths, n_timesteps]`.

---

## Architecture Overview

The pipeline has four stages:

```
Raw Simulations → Latent Space → POD → RBF → Predictions
```

### Stage 1: Latent Space Construction (`interpolation.py`)

Each simulation's SSE is compressed into a fixed-length vector via a two-level parametrization:

**Level 1 — Intrinsic parameter `u` (spatial)**
For each depth, compute the arc length through (state, log-slip-rate) space over the SSE cycle:
```
u_i = cumsum(sqrt(Δstate² + Δlog(sr)²)) / total_arc_length   ∈ [0, 1]
```
Fit a cubic B-spline mapping: `u → (state, sr, slip)` using `scipy.interpolate.make_lsq_spline`.

**Level 2 — Time parametrization**
Fit a cubic B-spline mapping: `t → u` where `t ∈ [0,1]` is normalized time over the SSE.

Both splines' knots and coefficients are concatenated into a single **latent vector** per (simulation, SSE, depth). The full latent matrix has shape `[n_simulations, latent_dim]`.

Key detail: a densification step (`insert_dense_u`) is applied before spline fitting to prevent underfitting in sparse regions.

### Stage 2: POD (`rom.py: build_pod`)

Apply SVD to the latent matrix:
```
U, S, Vh = SVD(Latent.T)
A = Latent @ U      # shape: [n_sims, r]
```
`U` are the POD basis vectors (shared structure across all simulations). `A` are the parameter-dependent coefficients. Truncate to `r` modes.

### Stage 3: ROM Training (`rom.py: build_rom`)

Train one `scipy.interpolate.RBFInterpolator` per SSE index, mapping normalized forward model parameters → POD coefficients:
```
RBF: f_params_normalized → A
```

### Stage 4: Prediction (`rom.py: predict`)

For new parameters `w`:
1. RBF predicts `A_pred`
2. Reconstruct latent: `latent_pred = A_pred @ U.T`
3. Inverse transform via spline evaluation → `(state, sr, slip)` time series

---

## Key Classes

### `ROM` (`src/ssEinterpolator/rom.py`)

The central orchestrator. Important attributes:

| Attribute | Description |
|-----------|-------------|
| `D` | Dict of `Data` objects keyed by param tuple — all raw simulation data |
| `D_sses` | Same data split into individual SSE windows |
| `latent` | List (per SSE) of latent matrices `[n_sims, latent_dim]` |
| `A`, `U`, `S`, `V` | POD components (lists per SSE) |
| `RBFs` | List of trained RBFInterpolator objects (one per SSE) |
| `f_params` | Forward model parameter array, shape `[n_sims, n_features]` |
| `f_params_normalized` | Z-score normalized version of `f_params` |
| `lf` | Depth array (along-fault positions) |

Important constructor parameters:

| Parameter | Purpose |
|-----------|---------|
| `load_f` | Callback: `load_f(str_param) → Data` — loads simulation from disk |
| `t_to_u_knot_l` | Number of knots for time→u spline (controls temporal resolution) |
| `u_to_par_knot_l` | Number of knots for u→(state,sr,slip) spline (controls spatial resolution) |
| `sses_num` | How many SSE cycles to extract from each simulation |
| `along_dp_sses_depth_detector` | Depth index used to detect SSE onset (default: 195) |
| `sses_detector_threshold` | Log SR threshold for SSE detection (default: -4) |
| `n_workers` | Parallel workers for latent construction (default: 30) |

### `Data` (`src/ssEinterpolator/data.py`)

Simple container:
```python
Data.sr      # [n_depths, n_time]
Data.state   # [n_depths, n_time]
Data.slip    # [n_depths, n_time]
Data.t       # [n_time] — time in years
Data.params  # [n_features] — parameter vector
```

---

## Key Functions

### `interpolation.py`

| Function | Direction | Description |
|----------|-----------|-------------|
| `interpolate_to_latent()` | Forward | Converts all `Data` objects → latent matrices (parallel) |
| `interpolate_to_latent_single_along_stk()` | Forward | Single depth: constructs u→pars spline |
| `interpolate_time_parametric_space()` | Forward | Single depth: constructs t→u spline |
| `inverse_interpolation()` | Inverse | Latent vectors → reconstructed `(sr, state, slip)` |
| `inverse_interpolate_to_latent_single_along_stk()` | Inverse | Single SSE reconstruction |
| `insert_dense_u()` | Helper | Densifies sparse regions of `u` before spline fitting |

### `utils.py`

| Function | Description |
|----------|-------------|
| `find_slip_events()` | Detects SSE onset times from log slip rate time series by thresholding |

---

## Data Flow in Detail

```
load_f(str_param) → Data object
         ↓
rom.D[param_tuple] = Data
         ↓
split_data_by_sse() → rom.D_sses[param_tuple][sse_idx]
         ↓
interpolate_to_latent() for each SSE index
  └── per simulation, per depth (parallel):
      ├── normalize state, log(sr), slip to [0,1]
      ├── compute arc-length parameter u
      ├── densify u (insert_dense_u)
      ├── fit u→(state,sr,slip) spline → store tck
      ├── fit t→u spline → store tck
      └── concatenate tck components → latent vector
         ↓
rom.latent[sse_idx] = matrix [n_sims, latent_dim]
         ↓
build_pod(r) → SVD → rom.A, rom.U, rom.S, rom.V
         ↓
build_rom() → RBFInterpolator per SSE
         ↓
predict(w) → list of Data objects (one per SSE)
```

---

## Spline Encoding Format

The latent vector for one simulation/SSE/depth packs these spline components in order:
```
[u_to_pars_knots | u_to_pars_coefficients_state | u_to_pars_coefficients_sr | u_to_pars_coefficients_slip |
 t_to_u_knots    | t_to_u_coefficients]
```
Plus stored normalization constants (min/max for each variable, time bounds).

These are stored/accessed via `interpolate_to_latent_single_along_stk` return value and consumed by `inverse_interpolate_to_latent_single_along_stk`.

---

## Normalization Conventions

- **Slip rate**: `log10(|sr|)` then min-max scaled to [0,1] per SSE
- **State variable**: min-max scaled to [0,1] per SSE
- **Slip**: min-max scaled to [0,1] per SSE
- **Time**: normalized to [0,1] over the SSE window
- **Parameters** (`f_params`): Z-score normalized for RBF training

---

## Current State of the Code

The branch `run_time_improve` has uncommitted modifications to:
- `src/ssEinterpolator/interpolation.py` — likely performance work
- `src/ssEinterpolator/rom.py` — likely corresponding ROM changes

The `io.py` file is a stub with placeholder `load_simulation_data()` — actual data loading is done via the user-provided `load_f` callback passed to `ROM.__init__()`.

---

## Preprocessing CLI (`standrtize_data.py`)

Standalone script run before training to:
1. Detect SSE onsets in raw `.npy` simulation output
2. Build a refined time grid (denser near SSE events)
3. Interpolate raw data to the new grid

```bash
python standrtize_data.py <input_dir> <output_dir> <prefix> <sr_idx> <t_start> <t_end>
```

---

## Dependencies

| Library | Role |
|---------|------|
| `numpy` | Array ops, SVD |
| `scipy.interpolate` | `make_lsq_spline`, `splev`, `RBFInterpolator` |
| `joblib` | Parallel latent construction |
| `matplotlib` | Visualization |

---

## Terminology Quick Reference

| Term | Meaning |
|------|---------|
| SSE | Slow Slip Event — one episodic slip cycle |
| ROM | Reduced Order Model — the fast surrogate |
| POD | Proper Orthogonal Decomposition — SVD-based basis |
| RBF | Radial Basis Function interpolation |
| `u` | Intrinsic arc-length parameter ∈ [0,1] |
| `sr` | Slip rate |
| `lf` | Along-fault depth positions |
| `tck` | Scipy spline tuple (knots, coefficients, degree) |
| `latent` | Fixed-length vector encoding a simulation's spline parameters |
| `A` | POD amplitude matrix (parameter-dependent) |
| `U` | POD basis matrix (shared across all simulations) |
| `f_params` | Forward model parameter vectors (friction coefficients, etc.) |
