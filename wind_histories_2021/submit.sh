#!/usr/bin/env bash
# Run from a Tufts login shell. All computation is submitted to Slurm.
set -euo pipefail
cd "$(dirname "$0")"
project=$(pwd -P)
mkdir -p runs
if [[ $# == 0 ]]; then
    run=$(mktemp -d "$project/runs/wind-XXXXXX")
elif [[ $# == 1 && -d $1 ]]; then
    run=$(cd "$1" && pwd -P)
    [[ $run == "$project"/runs/wind-* && -f $run/jobs.txt ]] || {
        echo "Expected an existing wind run from this package." >&2; exit 2; }
    # Prevent overlapping retries of the same run.
    while read -r stage id; do
        [[ -z $(squeue -h -j "$id" -o '%T' 2>/dev/null || true) ]] || {
            echo "Job $id from this run is still queued. See RUNBOOK.md for recovery." >&2; exit 2; }
    done < "$run/jobs.txt"
else
    echo "Usage: bash submit.sh [runs/wind-XXXXXX]" >&2; exit 2
fi
echo "Run directory: $run"
queue_job() {
    local stage=$1 result
    shift
    result=$(sbatch --parsable --job-name="wind-$stage" --output="$run/%x-%A_%a.log" \
                    "$@" job.sbatch "$stage" "$run")
    job_id=${result%%;*}
    [[ $job_id =~ ^[0-9]+$ ]]
    echo "$stage $job_id" >> "$run/jobs.txt"
    echo "$stage $job_id"
}
queue_job setup
queue_job smoke --dependency="afterok:$job_id" --kill-on-invalid-dep=yes
queue_job train --array=0-98%6 --dependency="afterok:$job_id" --kill-on-invalid-dep=yes
queue_job compare --dependency="afterok:$job_id" --kill-on-invalid-dep=yes
queue_job finish --dependency="afterok:$job_id" --kill-on-invalid-dep=yes
