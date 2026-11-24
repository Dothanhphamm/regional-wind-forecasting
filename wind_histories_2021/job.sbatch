#!/bin/bash -l
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
stage=$1
run=$2
export TORCH_HOME="$run/torch-cache"
if [[ $stage == setup ]]; then
    sha256sum --check SHA256SUMS
    if [[ -n ${WIND_PYTHON:-} ]]; then
        py=$WIND_PYTHON
        [[ $py == /* && -x $py ]]
    else
        module load miniforge/25.3.0
        export CONDA_PKGS_DIRS="$run/conda-cache"
        py="$run/env/bin/python"
        if [[ ! -x $py ]]; then
            conda create -y --prefix "$run/env" --override-channels -c conda-forge python=3.12 pip
        fi
        "$py" -m pip install --no-cache-dir 'numpy==1.26.4' 'scipy==1.15.1' 'pandas==2.2.2' 'xarray==2023.6.0'
        "$py" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1' 'torchvision==0.20.1'
        "$py" -m pip install --no-cache-dir 'lpips==0.1.4'
    fi
    "$py" -c 'import numpy, scipy, pandas, xarray, torch, lpips; lpips.LPIPS(net="squeeze", verbose=False).eval(); print(torch.__version__)'
    "$py" -m pip freeze > "$run/packages.txt"
    echo "$py" > "$run/python-path"
    exit 0
fi
py=$(<"$run/python-path")
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
case "$stage" in
    smoke) "$py" -u followup.py smoke "$run/smoke" ;;
    train) "$py" -u followup.py train "$run" --index "$SLURM_ARRAY_TASK_ID" ;;
    compare) "$py" -u followup.py freeze "$run" ;;
    finish)
        "$py" -u followup.py calibrate "$run"
        "$py" -u followup.py score "$run"
        ;;
    *) echo "Unknown stage $stage" >&2; exit 2 ;;
esac
