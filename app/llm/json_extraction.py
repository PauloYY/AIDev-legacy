import re


_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def extract_json_object(content: str) -> str:
    """Torna o parsing de JSON tolerante a variações comuns de saída de LLM.

    Modelos (especialmente os menores/gratuitos) frequentemente:
    - envolvem o JSON num bloco de código markdown (```json ... ```);
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