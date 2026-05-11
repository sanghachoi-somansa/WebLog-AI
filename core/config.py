"""팀 공용 Local LLM(vLLM, OpenAI 호환) 고정 설정.

모델 목록 확인: http://10.226.50.2/v1/models
"""

from __future__ import annotations

import os

# 공유 엔드포인트는 고정. 모델 id는 팀 로컬 설정·`/v1/models`와 일치해야 함.
# 팀 기본(로컬 config 권장): qwen3-vl — 대안 예: qwen3-vl:latest, ollama/qwen3-vl, openai/qwen3-vl
TEAM_VLLM_BASE_URL = "http://10.226.50.2/v1"
TEAM_VLLM_MODEL = os.getenv("VLLM_MODEL", "qwen3-vl")


def team_vllm_models_url() -> str:
    return f"{TEAM_VLLM_BASE_URL.rstrip('/')}/models"


def team_vllm_api_key() -> str:
    """OpenAI 호환 클라이언트용. 서버가 키를 요구하지 않으면 dummy로 둬도 됨."""
    return os.getenv("VLLM_API_KEY", "dummy")


def llm_enabled() -> bool:
    """`.env` 등에서 LLM 호출을 끄려면 ``DISABLE_LLM=1``."""
    v = os.getenv("DISABLE_LLM", "").strip().lower()
    return v not in ("1", "true", "yes", "on")


def llm_error_cooldown_seconds() -> float:
    """연속 500 등으로 분석 피드가 도배될 때 재시도 간격(초)."""
    try:
        return float(os.getenv("LLM_COOLDOWN_AFTER_ERROR", "120"))
    except ValueError:
        return 120.0
