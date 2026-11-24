#!/usr/bin/env bash
# Runs ON the 주경야독 GPU node — submit it with
#
#   python scripts/orch/orchctl.py up --gpus 8
#   STEPS=30000 python scripts/orch/orchctl.py runfile scripts/orch/node_pretrain.sh
#
# (orchctl injects the script through stdin; variables below can be overridden by
# prefixing them inside the script or exporting them in $ORCH_HOME/env.sh.)
#
# Layout under $ORCH_HOME/carmen_numeric:
#   code/   repo checkout (git clone, or the tarball pushed with orchctl push)
#   .venv/  persistent python env
#   data/vitaldb/  numerical-row cache built from $ORCH_HOME/vitaldb/raw/*.vital
#   runs/<RUN>/    checkpoints + log.jsonl; runs/<RUN>.log is stdout
set -euo pipefail
: "${ORCH_HOME:?ORCH_HOME is not set — run this on the node}"

ROOT="${ROOT:-$ORCH_HOME/carmen_numeric}"
CODE="${CODE:-$ROOT/code}"
VENV="${VENV:-$ROOT/.venv}"
DATA="${DATA:-$ROOT/data/vitaldb}"
VITAL_DIR="${VITAL_DIR:-$ORCH_HOME/vitaldb/raw}"
CASES_CSV="${CASES_CSV:-$ORCH_HOME/aki/csv/clinical_data.csv}"
LABS_CSV="${LABS_CSV:-$ORCH_HOME/aki/csv/lab_data.csv}"
PRESET="${PRESET:-base}"
STEPS="${STEPS:-30000}"
BATCH="${BATCH:-64}"
LR="${LR:-2e-4}"
RUN="${RUN:-unittab_${PRESET}_$(date +%Y%m%d_%H%M)}"
PREP_WORKERS="${PREP_WORKERS:-32}"

mkdir -p "$ROOT" "$ROOT/runs"
[ -d "$CODE" ] || { echo "no code at $CODE — clone or push the repo first"; exit 1; }

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q torch numpy pandas vitaldb
fi
PY="$VENV/bin/python"
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
EOF

if [ ! -f "$DATA/normalizer.json" ]; then
  echo "[prep] building cache in $DATA"
  args=(--out "$DATA" --vital-dir "$VITAL_DIR" --workers "$PREP_WORKERS")
  [ -f "$CASES_CSV" ] && args+=(--cases-csv "$CASES_CSV")
  [ -f "$LABS_CSV" ] && args+=(--labs-csv "$LABS_CSV")
  "$PY" "$CODE/scripts/prepare_vitaldb.py" "${args[@]}"
fi

NGPU=$("$PY" -c "import torch; print(max(torch.cuda.device_count(), 1))")
echo "[train] run=$RUN preset=$PRESET steps=$STEPS gpus=$NGPU"
# detached: the job returns at once, poll runs/$RUN.log
( setsid nohup "$VENV/bin/torchrun" --standalone --nproc_per_node "$NGPU" \
    "$CODE/scripts/pretrain.py" --data "$DATA" --out "$ROOT/runs/$RUN" \
    --preset "$PRESET" --steps "$STEPS" --batch-size "$BATCH" --lr "$LR" \
    --amp bf16 --workers 8 --eval-every 2000 \
    > "$ROOT/runs/$RUN.log" 2>&1 < /dev/null & )
echo "started; log: $ROOT/runs/$RUN.log"
