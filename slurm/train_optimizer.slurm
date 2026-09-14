#!/usr/bin/env bash
#SBATCH --job-name=resnet_mil
#SBATCH --partition=gpu_l48
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --chdir=/appsnew/home/jlpan/ResNet
#SBATCH --output=/appsnew/home/jlpan/ResNet/runs/slurm/resnet-%x-%j.out
#SBATCH --error=/appsnew/home/jlpan/ResNet/runs/slurm/resnet-%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/appsnew/home/jlpan/ResNet}"
ENV_PY="${ENV_PY:-/appsnew/home/jlpan/.conda/envs/usfm-thyroid/bin/python}"
OPTIMIZER="${OPTIMIZER:?Submit with OPTIMIZER=adamw or OPTIMIZER=sgd}"
BACKBONE="${BACKBONE:-resnet50}"
SEED="${SEED:-20260811}"
WEIGHTS="${WEIGHTS:-imagenet}"
DATA_ROOT="$PROJECT_ROOT/data/private_mil"
CODE="$PROJECT_ROOT/code/train_resnet_mil.py"
RUN_DIR="$PROJECT_ROOT/runs/${BACKBONE}_${OPTIMIZER}_${SLURM_JOB_ID}"

case "$OPTIMIZER" in
  adamw|sgd) ;;
  *) echo "Unsupported OPTIMIZER=$OPTIMIZER" >&2; exit 2 ;;
esac
case "$BACKBONE" in
  resnet18|resnet50) ;;
  *) echo "Unsupported BACKBONE=$BACKBONE" >&2; exit 2 ;;
esac

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"

mkdir -p "$PROJECT_ROOT/runs/slurm" "$PROJECT_ROOT/.cache/torch" "$MPLCONFIGDIR"
trap 'status=$?; echo "job_end=$(date --iso-8601=seconds) exit_status=$status"; exit $status' EXIT

echo "===== Patient-level ResNet-50 attention MIL ====="
echo "job_id=${SLURM_JOB_ID:-unknown} backbone=$BACKBONE optimizer=$OPTIMIZER seed=$SEED"
echo "project_root=$PROJECT_ROOT data_root=$DATA_ROOT run_dir=$RUN_DIR"
echo "training_source=private_only TN3K=not_used weights=$WEIGHTS"
echo "host=$(hostname) start=$(date --iso-8601=seconds)"

test -x "$ENV_PY"
test -f "$CODE"
test -f "$DATA_ROOT/patient_manifest.csv"
test -f "$DATA_ROOT/image_manifest.csv"
test ! -e "$RUN_DIR"

"$ENV_PY" -c 'import torch, torchvision, sklearn, PIL, numpy; print("env_ok", torch.__version__, torchvision.__version__, torch.cuda.is_available())'
"$ENV_PY" "$PROJECT_ROOT/code/validate_package.py" --data-root "$DATA_ROOT" --decode-sample 30

"$ENV_PY" "$CODE" \
  --data-root "$DATA_ROOT" \
  --output-dir "$RUN_DIR" \
  --optimizer "$OPTIMIZER" \
  --backbone "$BACKBONE" \
  --seed "$SEED" \
  --weights "$WEIGHTS" \
  --epochs 120 \
  --patience 15 \
  --freeze-epochs 3 \
  --max-train-images 16 \
  --max-eval-images 0 \
  --encode-chunk 8 \
  --grad-accum 4 \
  --num-workers 4

test -s "$RUN_DIR/best.pt"
test -s "$RUN_DIR/summary_metrics.json"
echo "summary_file=$RUN_DIR/summary_metrics.json"
cat "$RUN_DIR/summary_metrics.json"
