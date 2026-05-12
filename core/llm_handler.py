"""vLLM(OpenAI 호환)으로 로그 스니펫을 보안 관점에서 분석한다."""

from __future__ import annotations

import time

from openai import APIError, OpenAI

from core.config import TEAM_VLLM_BASE_URL, TEAM_VLLM_MODEL, team_vllm_api_key

QA_SYSTEM = (
    "너는 보안·시스템 엔지니어다. 사용자 질문에 **한국어**로 간결히 답하라.\n"
    "- 아래에 '참고 로그'가 붙어 있으면 그 내용을 **근거**로 활용하고, 없는 내용은 지어내지 말 것.\n"
    "- 로그와 무관한 일반 질문이면 일반 지식 범위에서 답해도 된다.\n"
    "- 답은 **10문장 이내**로 정리한다."
)

SECURITY_SYSTEM = (
    "너는 숙련된 보안 엔지니어다. 사용자가 붙여 넣은 **로그 스니펫**만 근거로 분석하라. 한국어로 답하라.\n\n"
    "## 출력 형식(반드시 준수)\n"
    "1) **분석(본문)**: 먼저, 근거가 되는 로그를 **짧게 인용**할 때는 반드시 **날짜·시간(타임스탬프)이 보이도록** "
    "원문 그대로 또는 최소 축약만 하여 인용한다.\n"
    "2) 그 다음 문장은 **'~ 로그 부분(타임스탬프 포함)은 ~ 이슈로 보이며, ~ 확인이 필요하다'** 를 "
    "변형한 자연스러운 문장으로 **분석 위주**로 쓴다. (예: '2025-05-11 10:23:01 …' 형태가 인용에 포함되게)\n"
    "3) **위협 여부·맥락**을 한 줄로 정리한다(정상/의심/악성 등).\n"
    "4) **위험도**는 1~5 숫자 한 번만 명시한다.\n"
    "5) **대응·해결**은 부가 정보다. **분석이 끝난 뒤** 필요할 때만, "
    "'참고(조치):' 로 시작해 **한 문장 이하**로 적는다. 없으면 생략한다.\n"
    "6) '대응책:'만 단독으로 크게 쓰거나, 대응만 반복하는 형식은 **금지**한다.\n\n"
    "## 작성 원칙\n"
    "- 인용하지 않은 사실을 지어내지 말 것. 로그에 없는 IP·사용자·파일명을 새로 만들지 말 것.\n"
    "- 동일한 문장을 복붙하듯 반복하지 말 것.\n"
    "- 전체 길이는 **8문장 이내**로 압축한다."
)

# 모델 컨텍스트·응답 길이 고려해 사용자 메시지 상한(문자)
_MAX_USER_CHARS = 12_000

_LLM_RETRY_MAX = 2
_LLM_RETRY_SLEEP_SEC = 2.5


def _llm_error_is_transient(exc: APIError) -> bool:
    s = str(exc).lower()
    code = getattr(exc, "status_code", None)
    return bool(
        code in (500, 502, 503, 504)
        or "500" in s
        or "502" in s
        or "503" in s
        or "connection error" in s
        or "internalservererror" in s
        or "litellm" in s
        or "timeout" in s
    )


def _llm_error_hint(model_id: str, exc: APIError) -> str:
    raw = str(exc)
    parts = [
        "\n\n[안내] 게이트웨이(LiteLLM 등)는 살아 있지만 **실제 추론 백엔드**에 연결하지 못한 경우가 많습니다. "
        "`/v1/models` 에 보여도 **모델 그룹(라우팅)** 이 비어 있으면 500이 납니다.\n"
        "→ `10.226.50.2` 에서 LiteLLM·Ollama·vLLM 프로세스와 라우팅 설정을 확인하세요.\n"
        "→ 로그만 보려면 `.env` 에 `DISABLE_LLM=1`.\n"
    ]
    mid = (model_id or "").strip()
    if mid.startswith(("openai/", "ollama/")):
        parts.append(
            f"→ 지금 모델 `{mid}` 는 접두사 라우팅이 자주 비어 있습니다. "
            "사이드바에서 **Qwen3-VL (Base)** (`qwen3-vl`) 로 바꿔 보세요.\n"
        )
    if "Connection error" in raw or "InternalServerError" in raw:
        parts.append(
            "→ **일시적** 장애일 수 있습니다. 잠시 후 다시 시도하거나, 위와 같이 **Base** 모델로 바꿔 보세요."
        )
    return "".join(parts)


def analyze_log_snippet(
    log_text: str,
    *,
    model: str | None = None,
    timeout: float = 120.0,
) -> str:
    """로그 텍스트 일부를 LLM에 보내 분석 문장을 반환한다.

    Args:
        log_text: 분석할 로그 본문.
        model: OpenAI 호환 ``model`` id. None이면 ``TEAM_VLLM_MODEL``(환경 기본값).
        timeout: API 타임아웃(초).
    """
    text = (log_text or "").strip()
    if not text:
        return "(빈 로그)"

    model_id = (model or "").strip() or TEAM_VLLM_MODEL
    payload = text[-_MAX_USER_CHARS:]
    client = OpenAI(
        base_url=TEAM_VLLM_BASE_URL.rstrip("/"),
        api_key=team_vllm_api_key(),
        timeout=timeout,
    )
    for attempt in range(_LLM_RETRY_MAX):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SECURITY_SYSTEM},
                    {"role": "user", "content": payload},
                ],
                temperature=0.2,
                max_tokens=900,
            )
            msg = resp.choices[0].message
            out = (msg.content or "").strip()
            return out if out else "(모델 응답 없음)"
        except APIError as e:
            if (
                attempt + 1 < _LLM_RETRY_MAX
                and _llm_error_is_transient(e)
            ):
                time.sleep(_LLM_RETRY_SLEEP_SEC)
                continue
            hint = ""
            if _llm_error_is_transient(e):
                hint = _llm_error_hint(model_id, e)
            return f"LLM API 오류: {e}{hint}"
        except Exception as e:
            return f"LLM 오류: {e}"


def answer_user_question(
    question: str,
    context_logs: str = "",
    *,
    model: str | None = None,
    timeout: float = 120.0,
) -> str:
    """페이지에서 사용자가 입력한 질문에 LLM이 답한다. 선택적으로 현재 로그 맥락을 붙인다."""
    q = (question or "").strip()
    if not q:
        return "(질문이 비어 있습니다.)"

    ctx = (context_logs or "").strip()
    if ctx:
        payload = (
            f"## 사용자 질문\n{q}\n\n"
            f"## 참고 로그 (일부)\n{ctx[-_MAX_USER_CHARS:]}"
        )
    else:
        payload = f"## 사용자 질문\n{q}"

    model_id = (model or "").strip() or TEAM_VLLM_MODEL
    client = OpenAI(
        base_url=TEAM_VLLM_BASE_URL.rstrip("/"),
        api_key=team_vllm_api_key(),
        timeout=timeout,
    )
    for attempt in range(_LLM_RETRY_MAX):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": QA_SYSTEM},
                    {"role": "user", "content": payload},
                ],
                temperature=0.3,
                max_tokens=1200,
            )
            msg = resp.choices[0].message
            out = (msg.content or "").strip()
            return out if out else "(모델 응답 없음)"
        except APIError as e:
            if (
                attempt + 1 < _LLM_RETRY_MAX
                and _llm_error_is_transient(e)
            ):
                time.sleep(_LLM_RETRY_SLEEP_SEC)
                continue
            hint = ""
            if _llm_error_is_transient(e):
                hint = _llm_error_hint(model_id, e)
            return f"LLM API 오류: {e}{hint}"
        except Exception as e:
            return f"LLM 오류: {e}"
