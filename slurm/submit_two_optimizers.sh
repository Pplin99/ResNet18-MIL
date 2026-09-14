#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/appsnew/home/jlpan/ResNet}"
SLURM_SCRIPT="$PROJECT_ROOT/slurm/train_optimizer.slurm"
SEED="${SEED:-20260811}"
WEIGHTS="${WEIGHTS:-imagenet}"
BACKBONE="${BACKBONE:-resnet50}"

case "$BACKBONE" in
  resnet18|resnet50) ;;
  *) echo "Unsupported BACKBONE=$BACKBONE" >&2; exit 2 ;;
esac

test -f "$SLURM_SCRIPT"
mkdir -p "$PROJECT_ROOT/runs/slurm" "$PROJECT_ROOT/runs/submissions"

adamw_job="$(sbatch --parsable --job-name="${BACKBONE}_adamw" \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",BACKBONE="$BACKBONE",OPTIMIZER=adamw,SEED="$SEED",WEIGHTS="$WEIGHTS" \
  "$SLURM_SCRIPT")"
sgd_job="$(sbatch --parsable --job-name="${BACKBONE}_sgd" \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",BACKBONE="$BACKBONE",OPTIMIZER=sgd,SEED="$SEED",WEIGHTS="$WEIGHTS" \
  "$SLURM_SCRIPT")"

stamp="$(date +%Y%m%d_%H%M%S)"
record="$PROJECT_ROOT/runs/submissions/submission_${stamp}.txt"
{
  echo "submitted_at=$(date --iso-8601=seconds)"
  echo "project_root=$PROJECT_ROOT"
  echo "backbone=$BACKBONE"
  echo "seed=$SEED"
  echo "weights=$WEIGHTS"
  echo "adamw_job_id=$adamw_job"
  echo "sgd_job_id=$sgd_job"
} | tee "$record"

echo "monitor: squeue -j ${adamw_job},${sgd_job} -o '%.18i %.20j %.2t %.10M %.6D %R'"
echo "accounting: sacct -j ${adamw_job},${sgd_job} --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES"
