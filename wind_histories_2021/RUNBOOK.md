# Compare 2, 5 and 10 training years on Tufts HPC

Upload `wind_histories_2021.tar.gz`. It contains the regional development data for 2008–2020, a separate 2021 test file, and the model and Slurm code. The 2-, 5- and 10-year models all forecast the same 2018 weather. Only the three 10-year model types advance to 2019 calibration and 2021 testing. Each task requests two CPU cores, 4 GB RAM and up to two hours. The training array runs up to six tasks at once.

## 1. Sign in and upload

Use your Tufts research-cluster account. Connect to the Tufts VPN off campus. Open OnDemand in Chrome or Firefox and sign in with your Tufts username. In **Files → Home Directory**, click **Upload** and choose `wind_histories_2021.tar.gz`. If your home quota is tight, use **Change directory** to upload to your writable research storage instead. Allow several GB of free space for the Python environment, package cache and outputs. [Tufts OnDemand and upload instructions](https://rtguides.it.tufts.edu/hpc/access/15-ondemand.html)

## 2. Open the shell and unpack

In OnDemand choose **Clusters → Tufts HPC Shell Access**. The shell starts in your cluster home. If you uploaded to research storage, first use `cd` to enter that exact directory. In a fresh directory containing the uploaded archive, run:

```bash
tar -xzf wind_histories_2021.tar.gz
cd wind_histories_2021
ls
```

Expect to see `submit.sh`, `job.sbatch`, `followup.py`, `regional_2008_2020.nc`, `holdout_2021.nc` and `SHA256SUMS`. Each short training window is sliced from the same continuous regional file. For a data rebuild, `prepare_histories.py` reads WeatherBench directly with xarray, dask, Zarr v2 and HTTP fsspec.

## 3. Submit the workflow

```bash
bash submit.sh
```

Copy the **Run directory** path printed on the first line. The next five lines give Slurm job IDs for `setup`, `smoke`, `train`, `compare` and `finish`. You can close the terminal once all five IDs appear. The jobs run under your Tufts account and the dependencies control their order. [Tufts batch job guide](https://rtguides.it.tufts.edu/hpc/slurm/batchjob.html)

| Stage | What happens |
| --- | --- |
| setup | Verifies package checksums, creates an isolated Python 3.12 environment with Miniforge, installs CPU PyTorch and records installed package versions. |
| smoke | Exercises all nine model/history combinations on 32 daily examples each. Uses 2020 to rehearse the final scoring path. |
| train | Runs 33 trials per training history, 99 in total. Every checkpoint records its training span, normalization and sample count. |
| compare | Selects a configuration per model within each history on 2018, writes the validation comparison and freezes only the 10-year finalists. |
| finish | Calibrates 90% wind intervals on 2019 and scores the selected 10-year checkpoints on 2021. |

The default setup needs access to conda-forge, PyPI and the PyTorch downloads host. It installs CPU PyTorch 2.5.1, torchvision 0.20.1 and LPIPS 0.1.4 and fetches pretrained SqueezeNet weights into the run directory before training. A compatible shared Python environment can be supplied via the absolute path in `WIND_PYTHON`. The setup checks the imports and loads the metric weights. [PyTorch CPU packages](https://pytorch.org/get-started/previous-versions/)

## 4. Monitor your run

```bash
squeue -u "$USER"
```

`PD` means pending and `R` means running. A pending dependency is normal while an earlier stage is running. Set the shell variable below to the actual run directory printed in step 3. Inspect job IDs and logs:

```bash
run=/replace/with/the/printed/run/directory
cat "$run/jobs.txt"
ls "$run"/*.log
```

Open an individual log with `less` and its exact filename. If a job leaves the queue, look up its final status, substituting its numeric ID for `12345`:

```bash
sacct -j 12345 --format=JobID,JobName,State,ExitCode,Elapsed
```

## 5. Read and download the results

Once `compare` completes, open `validation_by_history.csv`. At each lead, compare the nine rows (three models × three histories). `validation_physics_gain.csv` reports plain minus constrained RMSE at each size. Positive values favor the constraints. These are development comparisons on the same dates used for tuning. `validation_by_history_per_seed.csv` shows the spread across seeds. Once `finish` completes, read the separate final-test tables:

```bash
cat "$run/table1_24h.csv"
cat "$run/table1_48h.csv"
```

The rows show the conventional LSTM, plain PredRNN, and PredRNN with both physical penalties, in that order. Lower RMSE and LPIPS and higher SSIM are better. `primary_gain.csv` gives the paired wind-RMSE difference between plain PredRNN and the same PredRNN with **both** physical losses. A positive gain favors the joint constraints. `summary.csv` gives the actual errors in the mass and dry-static-energy diagnostics plus 90% wind interval coverage and widths.

Use **Files** in OnDemand to open the run directory and download the CSVs. For the paper keep `primary_gain.csv`, `summary.csv`, `per_seed.csv`, `frozen.json`, `validation_trials.csv`, `learning_curves.csv` and `packages.txt`. The fitted weights are in `trial-*.pt` and calibration radii in `intervals.pt`. The `smoke/` subdirectory contains the short rehearsal.

## Experimental design

The training region, 11 atmospheric fields, two-layer PredRNN, optimizer, batch size of 32, issue times and forecast leads match the first pilot. Forecasts use maps from 24 h earlier, 12 h earlier and the issue time. Each complete case starts at 00 UTC. Inputs and targets stay within their assigned split.

| Period | Role | Complete daily cases |
| --- | --- | ---: |
| 2016–2017 | Two-year training window | 728 |
| 2013–2017 | Five-year training window | 1,823 |
| 2008–2017 | Ten-year training window, prespecified for final testing | 3,650 |
| 2018 | Select epochs and joint physics weights separately within each history | 362 |
| 2019 | Calibrate 90% intervals for the ten-year models by grid cell, lead and wind component | 362 |
| 2021 | Final retrospective test for the ten-year models | 362 |

The earlier pilot used 2020 for testing. The production follow-up leaves that year out. The new 2021 file was checked for correct timestamps, spatial coverage, finite fields and above-ground pressure levels during packaging. The training and selection stages only use the development file. [WeatherBench 2 data source](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)

**Table 1 design.** The conventional LSTM receives flattened regional maps. Both PredRNN rows use the same two-layer convolutional model with temporal and zigzag spatiotemporal memory, the same starting parameters for a given seed, the same sampling schedule and the same forecast loss. Only the third row adds both physical penalties. The LSTM-to-plain comparison measures the change of architecture. The plain-to-joint comparison measures the combined effect of the two constraints.

Within each training window, independently estimate the normalization and diagnostic scales from that window. All histories use the same maximum epoch budget, so a longer history entails more gradient updates per epoch. All comparisons use one fixed wind-speed image range derived from 2016–2017, the training period shared by all three windows. The frozen 10-year manifest includes the selected checkpoints and this scale. The final stages verify checkpoint training span, case count and seeds before opening the test year.

**Objective.** All three use Adam with an initial learning rate of 0.0001 and RMSE on standardized forecast fields. Both PredRNN rows also use a shared memory-decoupling weight of 0.1. The decoupling term uses the absolute cosine similarity between projected memory writes, following the published PredRNN implementation. During training the probability of a true context input rises linearly from 0.5 to 1 and the probability of a true decoder input falls from 0.5 to 0 across the 60-epoch budget. For validation and testing, the entire input history is observed and all future inputs are model outputs. The linear schedule, residual readout, regional dimensions and decoupling weight are explicit implementation choices. [PredRNN source](https://github.com/thuml/predrnn-pytorch/blob/master/core/models/predrnn_v2.py)

**Both constraints.** The mass residual uses 500 hPa horizontal wind divergence plus the pressure derivative of vertical velocity between 400 and 600 hPa. The dry-static-energy diagnostic uses a 12-hour time difference of specific heat times temperature plus geopotential, horizontal advection and vertical advection in pressure coordinates. Each penalty is the area-weighted squared error relative to that diagnostic computed on ERA5, scaled by its training variance. The residuals can be nonzero in the coarse reference fields.

**Tuning.** Train one conventional LSTM, one plain PredRNN and nine joint-weight settings per seed. The two weights independently take values 0.003, 0.01 or 0.03, so both are active in every constrained trial. Seeds are 42, 43 and 44. Each job gets up to 60 epochs, with patience 10. Select the best epoch per seed and one weight pair across seeds by the mean of the 24- and 48-hour validation wind vector RMSE. A tie favors smaller mass weight followed by smaller energy weight. Thus hyperparameter tuning produces exactly one constrained row in the test table. The joint model has a larger validation search budget. The paired seven-day block bootstrap uses identical test days across seeds and the two PredRNN rows.

**Table metrics on real weather.** RMSE is area-weighted 500 hPa vector wind error in m/s over 15 interior cells. For SSIM and LPIPS, derive scalar wind speed from the same two components at each cell, divide by the maximum interior speed from the training years, then clip both images to [0,1]. Compute SSIM using valid 3×3 windows, population covariance and constants 0.01² and 0.03². For LPIPS v0.1, bilinearly resize the 3×5 speed map to 48×80, repeat grayscale across RGB, map to [-1,1] and apply pretrained SqueezeNet. Average image scores across days and then seeds. These image scores summarize coarse speed-map appearance. Interpolation adds no weather information, and opposite wind directions can give identical speed images. [Original SSIM](https://ece.uwaterloo.ca/~z70wang/publications/ssim.html) · [Original LPIPS](https://github.com/richzhang/PerceptualSimilarity)

## Recovery

Read the first failed stage's log and its `sacct` status. Failed dependencies cancel downstream jobs. If a training task hit the time limit, increase the wall time in the submission command for the retry according to your allocation (for example, `SBATCH_TIMELIMIT=04:00:00 bash submit.sh "$run"`). If the module version is unavailable or package downloads are blocked, use a compatible group environment through `WIND_PYTHON` or consult Tufts Research Technology about software access. Submit retries from the extracted package directory after the original jobs are terminal:

```bash
bash submit.sh "$run"
```

Completed trial checkpoints are reused. An interrupted trial restarts from its beginning. The retry uses the same experimental settings. Once 2021 scores have been seen, use a later untouched year for testing any design changes motivated by those scores.
