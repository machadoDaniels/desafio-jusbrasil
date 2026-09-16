"""Etapa 1: extrai candidatos de documentos jurídicos."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from tqdm import tqdm

from .contracts import (
    CandidatoCitacao,
    CandidatoCitacaoRequest,
    DocumentoExtraido,
    ExtratorCandidatos,
    LoteCandidatosRequest,
    PipelineConfig,
)

_SYSTEM_PROMPT = """Você extrai citações de documentos jurídicos brasileiros.
Retorne todas as citações a jurisprudência, súmulas e dispositivos legais.
Retorne somente o trecho literal, o tipo e, opcionalmente, a confiança.
Não calcule nem retorne posições, índices, offsets, início ou fim.
Não extraia números do processo do próprio cabeçalho, números de OAB,
protocolos, valores monetários ou referências sem natureza jurídica.
Classifique como jurisprudencia ou lei. Não avalie veracidade nesta etapa."""


class AgenteExtrator:
    """Extrai candidatos com saída estruturada e corrige seus spans localmente."""

    def __init__(
        self,
        cliente: OpenAI,
        modelo: str,
        temperature: float = 0,
        reasoning_effort: str | None = None,
    ) -> None:
        self._cliente = cliente
        self._modelo = modelo
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort

    def extrair(self, texto: str) -> list[CandidatoCitacao]:
        candidatos = self._consultar_modelo(texto)
        candidatos_com_spans = self._adicionar_spans(texto, candidatos)
        return self._remover_duplicatas(candidatos_com_spans)

    def _consultar_modelo(self, texto: str) -> list[CandidatoCitacaoRequest]:
        parametros = {}
        if self._reasoning_effort is not None:
            parametros["reasoning_effort"] = self._reasoning_effort
        resposta = self._cliente.chat.completions.parse(
            model=self._modelo,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Extraia as citações deste documento:\n\n{texto}",
                },
            ],
            response_format=LoteCandidatosRequest,
            **parametros,
        )
        resultado = resposta.choices[0].message.parsed
        if resultado is None:
            raise RuntimeError("o modelo não retornou uma extração estruturada")
        return resultado.candidatos

    @staticmethod
    def _adicionar_spans(
        texto: str,
        candidatos: list[CandidatoCitacaoRequest],
    ) -> list[CandidatoCitacao]:
        encontrados = []
        for candidato in candidatos:
            inicio = texto.find(candidato.trecho)
            while inicio >= 0:
                encontrados.append(
                    CandidatoCitacao(
                        **candidato.model_dump(),
                        inicio=inicio,
                        fim=inicio + len(candidato.trecho),
                    )
                )
                inicio = texto.find(candidato.trecho, inicio + 1)
        return encontrados

    def _remover_duplicatas(
        self,
        candidatos: list[CandidatoCitacao],
    ) -> list[CandidatoCitacao]:
        priorizados = sorted(
            candidatos,
            key=lambda candidato: (
                -(candidato.confianca_extracao or 0.0),
                candidato.inicio,
                candidato.fim,
            ),
        )
        unicos: list[CandidatoCitacao] = []
        for candidato in priorizados:
            if not any(_iou(candidato, existente) >= 0.5 for existente in unicos):
                unicos.append(candidato)
        return sorted(unicos, key=lambda candidato: (candidato.inicio, candidato.fim))


def _iou(a: CandidatoCitacao, b: CandidatoCitacao) -> float:
    intersecao = max(0, min(a.fim, b.fim) - max(a.inicio, b.inicio))
    if intersecao == 0:
        return 0.0
    uniao = (a.fim - a.inicio) + (b.fim - b.inicio) - intersecao
    return intersecao / uniao


def _escrever_jsonl(documentos: list[DocumentoExtraido], destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=destino.parent, delete=False
    ) as temporario:
        caminho_temporario = Path(temporario.name)
        for documento in documentos:
            temporario.write(documento.model_dump_json(exclude_none=True) + "\n")
    caminho_temporario.replace(destino)


def executar_extracao(
    input_dir: Path,
    output_file: Path,
    extrator: ExtratorCandidatos,
) -> None:
    arquivos = sorted(input_dir.glob("*.txt"))
    if not arquivos:
        raise ValueError(f"nenhum arquivo .txt encontrado em {input_dir}")

    documentos = []
    for arquivo in tqdm(arquivos, desc="Extraindo candidatos"):
        texto = arquivo.read_text(encoding="utf-8")
        documentos.append(
            DocumentoExtraido(
                documento_id=arquivo.stem,
                texto=texto,
                candidatos=extrator.extrair(texto),
            )
        )
    _escrever_jsonl(documentos, output_file)


def main() -> None:
    config = PipelineConfig.from_yaml(Path("pipeline.yaml"))
    destino = config.workdir / "01-extraction.jsonl"
    executar_extracao(
        config.input_dir,
        destino,
        AgenteExtrator(
            OpenAI(base_url=config.base_url),
            config.model,
            config.temperature,
            config.reasoning_effort,
        ),
    )
    print(f"{destino}: extração concluída")


if __name__ == "__main__":
    main()
