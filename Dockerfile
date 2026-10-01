# Build (precisa de internet; baixa dependências e os pesos):
#   docker build -t desafio-jusbrasil .
# Execução (offline, 1 GPU de 24 GB): use `bash run.sh <db> <pasta_txt> <saida>` no host,
# que monta os caminhos e chama esta imagem; ou diretamente:
#   docker run --rm --gpus all --network none --ipc=host \
#     -v /caminho/dados:/dados -v /caminho/saida:/saida \
#     desafio-jusbrasil /dados/base.db /dados/txt /saida/submission.csv
#
# Imagem base fixada por tag e digest.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.29.0@sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1
FROM ${VLLM_IMAGE}

ENV HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=0 \
    PYTHONUNBUFFERED=1 \
    OPENAI_API_KEY=dummy

# Pesos em revisão fixa (google/gemma-4-12B-it-qat-w4a16-ct)
ARG MODEL_ID=google/gemma-4-12B-it-qat-w4a16-ct
ARG MODEL_REVISION=1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee
ENV MODEL_REVISION=${MODEL_REVISION}
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}', revision='${MODEL_REVISION}')"

# curl: run_in_container.sh espera /v1/models responder. Fica depois do download dos
# pesos para não invalidar essa camada no cache.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# Versões fixas, as mesmas da imagem validada na RTX 4090 (compatíveis com o vLLM da base).
RUN pip install --no-cache-dir openai==3.22.1 python-dotenv==1.2.3 pyyaml==6.0.3 \
      tqdm==4.70.0 pandas==3.0.6 pydantic==2.13.5 \
    && pip install --no-cache-dir --no-deps .

COPY configs ./configs
COPY data/relatores_padronizacao.json ./data/relatores_padronizacao.json
COPY scripts/json_to_submission.py ./scripts/json_to_submission.py
COPY scripts/run_in_container.sh ./scripts/run_in_container.sh

# No runtime a rede pode estar cortada
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENTRYPOINT ["bash", "/app/scripts/run_in_container.sh"]
