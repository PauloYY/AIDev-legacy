import json
import re

from json_repair import repair_json


_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def extract_json_object(content: str) -> str:
    """Torna o parsing de JSON tolerante a variações comuns de saída de LLM.

    Modelos (especialmente os menores/gratuitos) frequentemente:
    - envolvem o JSON num bloco de código markdown (```json... ```);
    - adicionam texto explicativo antes ou depois do JSON, mesmo quando
      instruídos a retornar "somente JSON".

    Isso normaliza esses casos antes do json.loads. Para uma resposta já
    limpa (só o objeto JSON), o comportamento não muda.
    """

    text = content.strip()

    fence_match = _CODE_FENCE_PATTERN.match(text)

    if fence_match:
        text = fence_match.group(1).strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return text


def parse_json_object(content: str) -> dict:
    """Faz o parsing tolerante de uma resposta da LLM que deveria ser JSON.

    Ordem de tentativas:
    1. json.loads direto, no texto já sanitizado por extract_json_object
       (sem markdown fences / texto ao redor).
    2. Se isso falhar — JSON estruturalmente malformado, por exemplo
       aspas ou chaves não escapadas quando o modelo tenta citar/embutir
       um trecho de código — tenta reparar com json_repair antes de
       desistir. Isso é especialmente comum em linguagens com muita
       pontuação (chaves, aspas), como Java e C++.

    Levanta json.JSONDecodeError se nada funcionar, para o chamador
    converter numa mensagem de erro clara (e o Runner tentar de novo).
    """

    candidate = extract_json_object(content)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    repaired = repair_json(candidate, return_objects=True)

    if isinstance(repaired, dict) and repaired:
        return repaired

    raise json.JSONDecodeError(
        "Não foi possível interpretar como JSON, mesmo após tentativa de reparo.",
        candidate,
        0,
    )
