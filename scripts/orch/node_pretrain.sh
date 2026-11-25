#!/usr/bin/env bash
# Runs ON the 주경야독 GPU node (sovereign-llm-3). Submit with orchctl, one step at a time:
#
#   python scripts/orch/orchctl.py up --gpus 0            # CPU session for code + prep
#   python scripts/orch/orchctl.py runfile --raw scripts/orch/node_pretrain.sh   # STEP=code (default)
#   STEP=prep  ... (prepend `export STEP=prep` to the script, or set it in env.sh)
#   python scripts/orch/orchctl.py up --gpus 1            # GPU session for training
#   STEP=train ...
#   STEP=eval CKPTS="runs/a/best.pt runs/b/best.pt" SPLIT=all   # score on all of VitalDB
#   STEP=subset PROFILE=signal  # derive $ROOT/data/vitaldb_<profile> from $DATA (no re-parsing)
#   then train / eval with DATA=$ROOT/data/vitaldb_signal
#
# orchctl sends the file through stdin, so set variables by editing the first lines
# (or `printf 'export STEP=train\n' | cat - node_pretrain.sh > /tmp/x.sh` and runfile that).
#
# Since the 2026-11 move, $ORCH_HOME is not set inside sessions; the account's files are
# still under the old path, which is the default BASE below.
set -euo pipefail

STEP="${STEP:-code}"
BASE="${ORCH_HOME:-/NHNHOME/orch-data/users/fd0cc5e8e1e24c9da69dbc8c15330cdd}"
ROOT="${ROOT:-$BASE/carmen_numeric_unittab}"  # not carmen_numeric: another project lives there
CODE="${CODE:-$ROOT/code}"
BRANCH="${BRANCH:-main}"
PY="${PY:-$BASE/venvs/carmen/bin/python}"   # torch 2.11 + cu130, numpy, pandas, vitaldb
DATA="${DATA:-$ROOT/data/vitaldb}"
VITAL_DIR="${VITAL_DIR:-$BASE/vitaldb/raw}"
CASES_CSV="${CASES_CSV:-$BASE/aki/csv/clinical_data.csv}"
LABS_CSV="${LABS_CSV:-$BASE/aki/csv/lab_data.csv}"
PREP_WORKERS="${PREP_WORKERS:-48}"
PRESET="${PRESET:-base}"
STEPS="${STEPS:-30000}"
BATCH="${BATCH:-128}"
LR="${LR:-2e-4}"
RUN="${RUN:-unittab_${PRESET}_$(date +%Y%m%d_%H%M)}"
EXTRA="${EXTRA:-}"   # extra pretrain.py flags, e.g. "--span-p 0.2"
SPLIT="${SPLIT:-all}"  # eval: train | val | test | all | heldout
PROFILE="${PROFILE:-signal}"  # subset: field profile (carmen_numeric/data/subset.py)
CKPTS="${CKPTS:-runs/unittab_base_v2/best.pt runs/unittab_base_span_v1/best.pt}"  # under ROOT
# HOME is the per-session workspace; keep caches with the persistent files
export XDG_CACHE_HOME="$BASE/.cache" PIP_CACHE_DIR="$BASE/.cache/pip" TORCH_HOME="$BASE/.cache/torch"
# the node's 72 cores are shared and saturated: keep every BLAS/OpenMP pool small
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1

mkdir -p "$ROOT/runs" "$ROOT/logs"
cd "$ROOT"

case "$STEP" in
code)
  # no git on the node: fetch the branch tarball from codeload (as bin/update_code.sh does)
  curl -sSL -o repo.tar.gz \
    "https://codeload.github.com/dlcjfgmlnasa/carmen-numeric/tar.gz/refs/heads/$BRANCH"
  rm -rf .code_new && mkdir .code_new && tar xzf repo.tar.gz -C .code_new --strip-components=1
  rm -rf "$CODE" && mv .code_new "$CODE" && rm -f repo.tar.gz
  echo "code updated to $BRANCH:"; ls "$CODE"
  ;;
prep)
  LOG="$ROOT/logs/prep_$(date +%Y%m%d_%H%M).log"
  ( cd "$CODE" && setsid nohup "$PY" scripts/prepare_vitaldb.py --out "$DATA" \
      --vital-dir "$VITAL_DIR" --cases-csv "$CASES_CSV" --labs-csv "$LABS_CSV" \
      --workers "$PREP_WORKERS" > "$LOG" 2>&1 < /dev/null & )
  echo "prep started; log: $LOG"
  ;;
subset)
  [ -f "$DATA/normalizer.json" ] || { echo "no cache at $DATA — run STEP=prep first"; exit 1; }
  DST="$ROOT/data/vitaldb_$PROFILE"
  LOG="$ROOT/logs/subset_${PROFILE}_$(date +%Y%m%d_%H%M).log"
  ( cd "$CODE" && setsid nohup "$PY" scripts/derive_subset.py --src "$DATA" --dst "$DST" \
      --profile "$PROFILE" > "$LOG" 2>&1 < /dev/null & )
  echo "subset started -> $DST; log: $LOG"
  ;;
train)
  [ -f "$DATA/normalizer.json" ] || { echo "no cache at $DATA — run STEP=prep first"; exit 1; }
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  NGPU=$("$PY" -c "import torch; print(max(torch.cuda.device_count(), 1))")
  LOG="$ROOT/runs/$RUN.log"
  echo "[train] run=$RUN preset=$PRESET steps=$STEPS gpus=$NGPU batch/gpu=$BATCH extra=$EXTRA"
  if [ "$NGPU" -gt 1 ]; then
    LAUNCH=("$PY" -m torch.distributed.run --standalone --nproc_per_node "$NGPU")
  else
    LAUNCH=("$PY")
  fi
  ( cd "$CODE" && setsid nohup "${LAUNCH[@]}" scripts/pretrain.py --data "$DATA" \
      --out "$ROOT/runs/$RUN" --preset "$PRESET" --steps "$STEPS" --batch-size "$BATCH" \
      --lr "$LR" --amp bf16 --data-on-device --workers 2 --cpu-threads 4 \
      --eval-every 2000 --eval-batches 400 $EXTRA \
      > "$LOG" 2>&1 < /dev/null & )
  echo "started; log: $LOG"
  ;;
eval)
  [ -f "$DATA/normalizer.json" ] || { echo "no cache at $DATA — run STEP=prep first"; exit 1; }
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  paths=(); for c in $CKPTS; do paths+=("$ROOT/$c"); done
  LOG="$ROOT/logs/eval_${SPLIT}_$(date +%Y%m%d_%H%M).log"
  ( cd "$CODE" && setsid nohup "$PY" scripts/eval_ckpt.py --data "$DATA" --split "$SPLIT" \
      "${paths[@]}" > "$LOG" 2>&1 < /dev/null & )
  echo "eval started; log: $LOG"
  ;;
*)
  echo "unknown STEP=$STEP (code | prep | subset | train | eval)"; exit 1
  ;;
esac
