#!/usr/bin/env bash
# Executado dentro da imagem Docker (ENTRYPOINT); no host, use bash run.sh.
# Uso: bash scripts/run_in_container.sh <caminho_db> <pasta_txt> <arquivo_saida.csv>
# Sobe o vLLM local, enriquece o .db, roda as 4 etapas e grava o CSV de submissão.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "uso: bash scripts/run_in_container.sh <caminho_db> <pasta_txt> <arquivo_saida.csv>" >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
DB="$(realpath "$1")"
TXT="$(realpath "$2")"
SAIDA="$(realpath -m "$3")"

MODELO="${MODELO:-google/gemma-4-12B-it-qat-w4a16-ct}"
REVISAO="${MODEL_REVISION:-1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee}"
PORTA="${PORTA:-8000}"
WORKDIR="${WORKDIR:-$(pwd)/outputs/final}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
ESPERA_MAXIMA=1800
# Tempo máximo do enriquecimento, em minutos; ao estourar, materializa o banco com o que já foi feito.
PRE_TEMPO_MAXIMO="${PRE_TEMPO_MAXIMO:-180}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$WORKDIR" "$(dirname "$SAIDA")"
# Tudo o que o script imprime também vai para $WORKDIR/run.log.
exec > >(tee -a "$WORKDIR/run.log") 2>&1

INICIO=$SECONDS
ETAPA="inicialização"
log() { printf '[%s] [%s] %s\n' "$(date +%H:%M:%S)" "$ETAPA" "$*"; }
duracao() { local s=$(( SECONDS - $1 )); printf '%dm%02ds' $((s / 60)) $((s % 60)); }

VLLM_PID=""
finalizar() {
  local codigo=$?
  if [ "$codigo" -ne 0 ]; then
    log "FALHOU (código $codigo) após $(duracao "$INICIO")"
    log "logs: $WORKDIR/run.log e $WORKDIR/vllm.log"
  fi
  [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true
  # Devolve a posse dos arquivos ao usuário do host quando o container roda como root.
  if [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
    chown -R "$HOST_UID:$HOST_GID" "$WORKDIR" "$SAIDA" 2>/dev/null || true
  fi
}
trap finalizar EXIT

[ -f "$DB" ] || { log "banco não encontrado: $DB"; exit 2; }
N_TXT=$(( $(find "$TXT" -maxdepth 1 -name '*.txt' | wc -l) ))
[ "$N_TXT" -gt 0 ] || { log "nenhum .txt em $TXT"; exit 2; }
log "banco: $DB ($(du -h "$DB" | cut -f1))"
log "documentos: $N_TXT .txt em $TXT"
log "saída: $SAIDA"
log "workdir: $WORKDIR"
log "modelo: $MODELO @ ${REVISAO:0:12}"
log "vLLM: max-model-len=$MAX_MODEL_LEN gpu-memory-utilization=$GPU_MEMORY_UTILIZATION max-num-seqs=$MAX_NUM_SEQS"

# Pré-processamento e entidades leem o dicionário de relatores ao lado do banco enriquecido.
cp data/relatores_padronizacao.json "$WORKDIR/"

ETAPA="vLLM"
T=$SECONDS
vllm serve "$MODELO" \
  --revision "$REVISAO" --tokenizer-revision "$REVISAO" \
  --host 127.0.0.1 --port "$PORTA" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --reasoning-parser gemma4 \
  --enable-chunked-prefill \
  --seed 0 \
  > "$WORKDIR/vllm.log" 2>&1 &
VLLM_PID=$!
log "subindo o servidor (pid $VLLM_PID, log em $WORKDIR/vllm.log)"

# A primeira subida compila kernels e pode demorar.
pronto=0
while [ $(( SECONDS - T )) -lt "$ESPERA_MAXIMA" ]; do
  if curl -fs "http://127.0.0.1:$PORTA/v1/models" > /dev/null; then
    pronto=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    log "o vLLM encerrou antes de ficar pronto; últimas linhas do log:"
    tail -n 40 "$WORKDIR/vllm.log" | sed 's/^/    /'
    exit 1
  fi
  if [ $(( (SECONDS - T) % 60 )) -lt 5 ] && [ $(( SECONDS - T )) -ge 60 ]; then
    log "ainda aguardando ($(duracao "$T")): $(tail -n 1 "$WORKDIR/vllm.log" | cut -c1-120)"
  fi
  sleep 5
done
if [ "$pronto" -ne 1 ]; then
  log "o vLLM não respondeu em ${ESPERA_MAXIMA}s; últimas linhas do log:"
  tail -n 40 "$WORKDIR/vllm.log" | sed 's/^/    /'
  exit 1
fi
log "pronto em $(duracao "$T")"

ETAPA="pré-processamento"
T=$SECONDS
log "enriquecendo o banco em $WORKDIR/enriched.db (limite de $PRE_TEMPO_MAXIMO min)"
PRE=(python3 -m desafio_jusbrasil.database_preprocessing
  --config configs/final_database_preprocessing.yaml
  --input "$DB" --output "$WORKDIR/enriched.db" --audit-dir "$WORKDIR/database-preprocessing")

# O run completo não materializa o banco se algum documento ficar em revisão;
# nesse caso materializa com o que existe (documentos sem resultado ficam com colunas nulas).
# Os checkpoints são gravados de forma atômica, então interromper no meio não corrompe nada.
if timeout "${PRE_TEMPO_MAXIMO}m" "${PRE[@]}"; then
  log "concluído em $(duracao "$T")"
else
  codigo=$?
  revisao="$WORKDIR/database-preprocessing/revisao.jsonl"
  if [ "$codigo" -eq 124 ]; then
    log "limite de $PRE_TEMPO_MAXIMO min atingido; materializando o banco com os documentos já processados"
  elif [ -s "$revisao" ]; then
    log "$(wc -l < "$revisao") documentos sem resultado (ver $revisao); materializando o banco parcial"
  else
    log "falhou com código $codigo sem documentos em revisão; tentando materializar o que existe"
  fi
  "${PRE[@]}" --materializar
  log "banco parcial materializado em $(duracao "$T")"
fi

ETAPA="pipeline"
T=$SECONDS
log "extração → completude → entidades → veracidade"
python3 -m desafio_jusbrasil.orchestrator \
  --config configs/final_pipeline.yaml \
  --input-dir "$TXT" --workdir "$WORKDIR" --database "$WORKDIR/enriched.db"
log "concluído em $(duracao "$T")"

ETAPA="submissão"
python3 scripts/json_to_submission.py "$WORKDIR/predictions" "$SAIDA"
python3 - "$WORKDIR/predictions" <<'EOF' | while IFS= read -r linha; do log "$linha"; done
import collections, json, pathlib, sys
docs = [json.loads(p.read_text(encoding="utf-8")) for p in pathlib.Path(sys.argv[1]).glob("*.json")]
classes = collections.Counter(c["classificacao"] for d in docs for c in d["citacoes"])
print(f"{len(docs)} documentos, {sum(classes.values())} citações: "
      + ", ".join(f"{k}={v}" for k, v in sorted(classes.items())))
EOF

ETAPA="fim"
log "CSV em $SAIDA; tempo total $(duracao "$INICIO")"
