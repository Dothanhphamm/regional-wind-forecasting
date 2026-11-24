# Regional wind forecasting · First research release

Open **[the four-page research report](report/regional-wind-report.pdf)** to read the findings. An [editable Word version](report/regional-wind-report.docx) accompanies it. This release covers the completed three-model Tufts experiment with nested two-, five- and ten-year training histories and the final 2021 test. The earlier exploratory 2020 pilot is outside this release.

The ten-year constrained PredRNN lowered the plain PredRNN’s mean 2021 vector wind RMSE by 0.098 m/s at 24 hours and 0.149 m/s at 48 hours. Only ten-year models reached 2021. All three training lengths can be compared on the common 2018 validation dates.

## 1. Inspect the results

Unzip the archive and open the resulting `regional-wind-v1` folder. You can read the report and CSVs immediately. From a terminal in that folder, run this command with Python 3 to verify the file contents against the release manifest.

```bash
python3 verify_release.py
```

| Question | File or directory |
| --- | --- |
| How did training history affect validation? | [2018 results](wind-6afo1x/validation_by_history.csv) and [paired model differences](wind-6afo1x/validation_physics_gain.csv) |
| What were the final model scores? | [24-hour table](wind-6afo1x/table1_24h.csv), [48-hour table](wind-6afo1x/table1_48h.csv) and [all leads and diagnostics](wind-6afo1x/summary.csv) |
| How uncertain was the physics gain? | [Paired bootstrap intervals](wind-6afo1x/primary_gain.csv) and [individual seeds](wind-6afo1x/per_seed.csv) |
| Which models were selected? | [Final settings and checkpoint indices](wind-6afo1x/frozen.json), [99 trial scores](wind-6afo1x/validation_trials.csv) and [learning curves](wind-6afo1x/learning_curves.csv) |
| Where are the data and code? | [Experiment and two NetCDF files](wind_histories_2021/), [model definitions](wind_histories_2021/paper_models.py), [selection and scoring](wind_histories_2021/followup.py) |
| Where are the fitted models? | [Tufts results](wind-6afo1x/) holds all 99 trial checkpoints, the calibration radii, all ten CSVs, the final manifest, Slurm IDs and the environment record. |

In the raw tables, `lstm` means the conventional flattened-map LSTM, `plain` the regional PredRNN and `both` the same PredRNN with mass and dry-static-energy penalties. Wind RMSE uses both wind components at 500 hPa. Physics gain is plain PredRNN error minus constrained PredRNN error. A positive value favors the constrained model. Three seeds contributed to each mean.

## 2. Reproduce the scores from the fitted checkpoints

Use Python 3.12 with the scientific environment from [the original setup script](wind_histories_2021/job.sbatch). If you have a compatible trained-run environment, activate it and skip the installation. For a fresh CPU environment on Linux, start inside the release folder with these commands.

```bash
python3.12 -m venv .wind-env
source .wind-env/bin/activate
python -m pip install numpy==1.26.4 scipy==1.15.1 pandas==2.2.2 xarray==2023.6.0
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
python -m pip install lpips==0.1.4
```

On macOS use the default PyPI index for the torch and torchvision installation line. With the environment active, choose a fresh output directory and run the replay on a workstation or an allocated compute node. On Tufts, the equivalent command as a batch job is also shown below.

```bash
python replay.py replay-results
```

```bash
sbatch --export=ALL --partition=batch --cpus-per-task=2 --mem=4G --time=02:00:00 --job-name=wind-replay --wrap='python replay.py replay-results'
```

The replay checks the release hashes, copies the 99 checkpoints to the new folder, repeats validation selection for all histories, recalibrates 90% prediction intervals using 2019 and scores the nine ten-year finalist fits on 2021. It then compares the frozen selections, calibration radii and all ten CSVs with the Tufts outputs, allowing for floating-point variation across CPU implementations. The supplied Torch cache contains the SqueezeNet weights used by LPIPS. The replay performs zero training epochs.

## 3. Submit a fresh training run on Tufts

Upload and extract this ZIP on Tufts, then open the extracted folder in a cluster shell. From its root, run the following commands. The original [Tufts runbook](wind_histories_2021/RUNBOOK.md) covers monitoring, recovery and output interpretation. Its first two sections describe transfer of the historical tarball. This ZIP already contains that tarball’s extracted directory.

```bash
cd wind_histories_2021
bash submit.sh
```

The workflow creates a new run directory, provisions its CPU environment and submits the smoke check, the 99 training trials, selection and final scoring. Each trial requests two CPU cores and 4 GB of RAM. Up to six training tasks can run concurrently. Save the printed run directory and Slurm IDs. The [original package list](wind-6afo1x/packages.txt) records the completed Linux environment. The setup script includes the installable specifications.

## 4. Rebuild the report

From a working copy of this release, use Python with matplotlib and python-docx installed. Run the following commands in the release root. They regenerate Figure 1 from the bootstrap CSV and rebuild the editable report with its tables from the archived scores. To export the Word file to PDF, use Word’s PDF export or LibreOffice. The command below writes the new PDF to a separate folder.

```bash
python report/plot_gain.py
python report/build_report.py
mkdir -p report-export
soffice --headless --convert-to pdf --outdir report-export report/regional-wind-report.docx
```

## Verification of this release

A separate CPU replay on macOS selected the same nine ten-year finalists from the 99 Tufts checkpoints. Recomputed 2019 interval radii agreed within 0.000028 m/s. All ten regenerated CSVs passed the numeric comparison, including the paired 2021 bootstrap intervals. The largest difference across the twelve mean 2021 vector RMSE scores was below 0.000001 m/s. Training is preserved through the original checkpoints, per-epoch curves and trial settings. This check reran the three stages after training.

## Provenance

The code/data directory contains the byte-for-byte members of the original Tufts submission. The results directory contains the byte-for-byte members of the downloaded Tufts output archive. Both directories use their original names. The report, replay helper, checksums and report-building scripts were added for this release.

| Original archive | SHA-256 |
| --- | --- |
| `wind_histories_2021.tar.gz` | `77cb533b813136022a4898692c26ce3fd32016e20a9ee142ba311b86b42e0788` |
| `wind-6afo1x-results.tar.gz` | `d59477f700c8cb7381057e3fa29feba7f51b72b5c83b2c6d253b53ff798ae92f` |

Weather data contain modified Copernicus Climate Change Service information for 2008–2021, distributed as ERA5 through WeatherBench 2. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data in this release. The exact Zarr source URL is recorded in both NetCDF files and in the preparation code. [Data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html) · [Data licence](sources/WeatherBench-ERA5-LICENSE.txt)

The LPIPS cache uses the ImageNet-1K V1 SqueezeNet 1.1 weights provided by Torchvision. [Weight documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.squeezenet1_1.html) · [Torchvision notice](sources/Torchvision-LICENSE.txt) · [SqueezeNet notice](sources/SqueezeNet-LICENSE.txt)
