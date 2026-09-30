#!/usr/bin/env bash
# Uso: bash run.sh <caminho_db> <pasta_txt> <arquivo_saida.csv>
# Sobe o vLLM local, enriquece o .db, roda as 4 etapas e grava o CSV de submissão.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "uso: bash run.sh <caminho_db> <pasta_txt> <arquivo_saida.csv>" >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")"
DB="$(realpath "$1")"
TXT="$(realpath "$2")"
SAIDA="$(realpath -m "$3")"

MODELO="${MODELO:-google/gemma-4-12B-it-qat-w4a16-ct}"
PORTA="${PORTA:-8000}"
WORKDIR="${WORKDIR:-$(pwd)/outputs/final}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$WORKDIR" "$(dirname "$SAIDA")"

vllm serve "$MODELO" \
  --host 127.0.0.1 --port "$PORTA" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --enable-chunked-prefill \
  --seed 0 \
  > "$WORKDIR/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

echo "aguardando o vLLM em 127.0.0.1:$PORTA"
for _ in $(seq 1 180); do
  if curl -fs "http://127.0.0.1:$PORTA/v1/models" > /dev/null; then
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM encerrou antes de ficar pronto; veja $WORKDIR/vllm.log" >&2
    exit 1
  fi
  sleep 5
done
curl -fs "http://127.0.0.1:$PORTA/v1/models" > /dev/null || {
  echo "vLLM não respondeu a tempo; veja $WORKDIR/vllm.log" >&2
  exit 1
}

PRE=(python -m desafio_jusbrasil.database_preprocessing
  --config configs/final_database_preprocessing.yaml
  --input "$DB" --output "$WORKDIR/enriched.db" --audit-dir "$WORKDIR/database-preprocessing")

# O run completo não materializa o banco se algum documento ficar em revisão;
# nesse caso materializa com o que existe (documentos sem resultado ficam com colunas nulas).
if ! "${PRE[@]}"; then
  echo "aviso: há documentos sem resultado; materializando o banco parcial" >&2
  "${PRE[@]}" --materializar
fi

python -m desafio_jusbrasil.orchestrator \
  --config configs/final_pipeline.yaml \
  --input-dir "$TXT" --workdir "$WORKDIR" --database "$WORKDIR/enriched.db"

python scripts/json_to_submission.py "$WORKDIR/predictions" "$SAIDA"
