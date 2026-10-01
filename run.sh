#!/usr/bin/env bash
# Ponto de entrada único: bash run.sh <caminho_db> <pasta_txt> <arquivo_saida>
# Roda a solução inteira no container Docker (vLLM + pipeline), sem rede.
# Requer Docker com acesso à GPU (NVIDIA Container Toolkit). Se a imagem não existir,
# ela é construída antes (único passo que usa internet: dependências e pesos).
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "uso: bash run.sh <caminho_db> <pasta_txt> <arquivo_saida>" >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")"
IMAGEM="${IMAGEM:-desafio-jusbrasil:latest}"
DB="$(realpath "$1")"
TXT="$(realpath "$2")"
SAIDA="$3"
[ -f "$DB" ] || { echo "banco não encontrado: $1" >&2; exit 2; }
[ -d "$TXT" ] || { echo "pasta de TXT não encontrada: $2" >&2; exit 2; }
# Uma pasta como saída recebe submission.csv.
if [ -d "$SAIDA" ] || [[ "$SAIDA" == */ ]]; then
  SAIDA="${SAIDA%/}/submission.csv"
fi
SAIDA="$(realpath -m "$SAIDA")"
mkdir -p "$(dirname "$SAIDA")" outputs

if ! docker image inspect "$IMAGEM" > /dev/null 2>&1; then
  echo "imagem $IMAGEM não encontrada; construindo"
  docker build -t "$IMAGEM" .
fi

docker run --rm --gpus "${GPUS:-all}" --network none --ipc=host \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -v "$DB:/entrada/$(basename "$DB")" \
  -v "$TXT:/entrada/txt:ro" \
  -v "$(dirname "$SAIDA"):/saida" \
  -v "$(pwd)/outputs:/app/outputs" \
  "$IMAGEM" "/entrada/$(basename "$DB")" /entrada/txt "/saida/$(basename "$SAIDA")"

echo "$SAIDA"
