"""vLLM(OpenAI 호환)으로 로그 스니펫을 보안 관점에서 분석한다."""

from __future__ import annotations

from openai import APIError, OpenAI

from core.config import TEAM_VLLM_BASE_URL, TEAM_VLLM_MODEL, team_vllm_api_key

SECURITY_SYSTEM = (
    "너는 숙련된 보안 엔지니어다. 다음 로그가 보안 위협(무단 접근, 권한 오용 등)인지 "
    "판단하고 위험도를 1-5로 분류한 뒤 한 줄 대응책을 제시하라. 한국어로 간결하게 답하라."
)

# 모델 컨텍스트·응답 길이 고려해 사용자 메시지 상한(문자)
_MAX_USER_CHARS = 12_000


def analyze_log_snippet(log_text: str, *, timeout: float = 120.0) -> str:
    """로그 텍스트 일부를 LLM에 보내 분석 문장을 반환한다."""
    text = (log_text or "").strip()
    if not text:
        return "(빈 로그)"

    payload = text[-_MAX_USER_CHARS:]
    client = OpenAI(
        base_url=TEAM_VLLM_BASE_URL.rstrip("/"),
        api_key=team_vllm_api_key(),
        timeout=timeout,
    )
    try:
        resp = client.chat.completions.create(
            model=TEAM_VLLM_MODEL,
            messages=[
                {"role": "system", "content": SECURITY_SYSTEM},
                {"role": "user", "content": payload},
            ],
            temperature=0.2,
            max_tokens=512,
        )
    except APIError as e:
        raw = str(e)
        hint = ""
        if (
            "Connection error" in raw
            or "InternalServerError" in raw
            or "litellm" in raw.lower()
            or "500" in raw
        ):
            hint = (
                "\n\n[안내] 게이트웨이(LiteLLM 등)는 살아 있지만 **실제 추론 백엔드(Ollama·vLLM 등)** 에 "
                "연결하지 못한 경우가 많습니다. `/v1/models` 에 모델이 보여도 inference 경로가 다를 수 있습니다.\n"
                "→ `10.226.50.2` 서버에서 백엔드 프로세스·포트·LiteLLM 라우팅을 확인하세요.\n"
                "→ 로그만 보려면 `.env` 에 `DISABLE_LLM=1` 로 분석 호출을 끌 수 있습니다."
            )
        return f"LLM API 오류: {e}{hint}"
    except Exception as e:
        return f"LLM 오류: {e}"

    msg = resp.choices[0].message
    out = (msg.content or "").strip()
    return out if out else "(모델 응답 없음)"
