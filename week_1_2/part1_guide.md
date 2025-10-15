# `part1.py` Guide


## Prerequisites
- Python 3.9+ with the repo's dependencies installed (notably `numpy`, `matplotlib`, `tqdm`, and the `simulation_and_control` package shipped with the lab).
- A valid `pandaconfig.json` in `week_1_2/` describing the robot configuration expected by PyBullet.
- PyBullet GUI access (the script renders by default and listens for the `q` key to abort a trajectory).
- Write access to `week_1_2/checkpoints/` and `week_1_2/plots_part1/` if you keep the default save locations.


## Quick Start
1. Activate the lab's virtual environment.
2. `cd` to `week_1_2`
   ```bash
   cd week_1_2
   ```
3. From `week_1_2` directory, run the script:
   ```bash
   python part1.py
   ```
4. The script boots the simulator, plays the prescribed trajectories, and logs joint regressors and torques.
5. When collection finishes, the combined parameter vector `a` is saved to `week_1_2/checkpoints/a_part1.npy`, and evaluation plots appear (and are persisted if `save_plots=True`).

> Tip: Press `q` in the PyBullet window to stop the current trajectory early. The run continues with the next trajectory unless you close the simulator.


## Runtime Configuration
Settings live in the `Part1Config` dataclass (see `week_1_2/part1_test.py:15`). Update them either by editing the defaults, or by constructing a `Part1Config` in a custom entrypoint and calling `collect_data(config)`.

- `conf_file_name`: Name of the PyBullet configuration file. Defaults to `pandaconfig.json`.
- `cur_dir`: Absolute directory containing the configs and output folders. Defaults to `week_1_2/`.
- `source_names`: Dynamics sources passed to `PinWrapper`. Leave as `["pybullet"]` unless you have additional models.
- `kp`, `kd`: Feedback linearization gains. Tune if the robot oscillates or responds sluggishly.
- `collection_max_time_per_trajectory`: Seconds to run each generated trajectory.
- `skip_initial`: Number of samples ignored when building the dataset (helps bypass transients right after reset).
- `num_trajectories`: Counts per trajectory type when generating training data (`generate_trajectories` expects a 5-element list).
- `evaluation_trajectories`: Same as above, but for the evaluation pass.
- `train_mix` / `evaluation_mix`: When `True`, mixes different trajectory types during data generation.
- `run_evaluation`: If `False`, skips the evaluation phase entirely.
- `save_model`: Toggle for writing `a` to disk.
- `save_plots`: Toggle for persisting evaluation plots under `plots_part1/`.
- `show_plots`: Toggle for interactive plot display via Matplotlib.
- `model_save_path`: Where to store the learned parameter vector.
- `plots_save_dir`: Target directory for plots.


## Data Collection
We have implement different trajectory generators in file `utils.trajectory_generator`

## Evaluation & Plotting
1. Set `run_evaluation = True`, the simulator will resets and generates new trajectories for evaluation.
2. Regressors and torques collected from the simulatiuon will be feed into `utils.draw_graphs.draw_plots`, which compares measured torques against the model predictions.
3. Plots are saved under `plots_part1/` (one file per joint) and optionally shown interactively.

