"""LogSentinel AI — Streamlit 대시보드 (SSH 스트림 + LLM 분석)."""

from __future__ import annotations

import html
import os
import queue
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

from core.config import (
    TEAM_VLLM_BASE_URL,
    TEAM_VLLM_MODEL,
    llm_enabled,
    llm_error_cooldown_seconds,
    team_vllm_models_url,
)
from core.llm_handler import analyze_log_snippet, answer_user_question
from core.ssh_manager import RemoteTailError, iter_remote_tail_lines

PAGE_TITLE = "LogSentinel AI"

LOG_QUEUE_MAX = 10_000
MAX_LOG_BUFFER_CHARS = 400_000
MAX_ANALYSIS_BUFFER_CHARS = 300_000
LLM_BATCH_MIN_LINES = 25
LLM_SNIPPET_CHARS = 8_000
MAX_PENDING_LLM = 2

# Somansa 제품 로그 프리셋
# 팀 로컬 LLM 설정과 동일한 id (사이드바에서 선택)
LLM_MODEL_PRESETS: tuple[tuple[str, str], ...] = (
    ("Qwen3-VL (Base)", "qwen3-vl"),
    ("Qwen3-VL (Latest)", "qwen3-vl:latest"),
    ("Qwen3-VL (Ollama 경로)", "ollama/qwen3-vl"),
    ("Qwen3-VL (OpenAI 경로)", "openai/qwen3-vl"),
)

LOG_PATH_PRESETS: tuple[tuple[str, str], ...] = (
    ("1. dlpcenter.log", "/somansa/dlpcenter/tomcat/logs/dlpcenter.log"),
    ("2. queryserver.log", "/somansa/common/tomcat_queryserver/logs/queryserver.log"),
    ("3. edrserver.log", "/somansa/edr/log/edrserver.log"),
    ("4. analyzerd.log", "/somansa/common/log/analyzerd.log"),
    ("5. summaryd.log", "/somansa/common/log/summaryd.log"),
    ("6. indexerd.log", "/somansa/common/log/indexerd.log"),
    ("7. checkerd.log", "/somansa/common/log/checkerd.log"),
)


def _ssh_port() -> int:
    try:
        return int(os.getenv("SSH_PORT", "22"))
    except ValueError:
        return 22


def _ssh_tail_worker(
    *,
    host: str,
    username: str,
    password: str,
    remote_path: str,
    port: int,
    stop_ev: threading.Event,
    log_q: queue.Queue[str],
    err_holder: list[str | None],
    gen_holder: list[Any],
) -> None:
    err_holder[0] = None
    gen: Any = None
    try:
        gen = iter_remote_tail_lines(
            host,
            username,
            password,
            remote_path,
            port=port,
            initial_lines=100,
            follow_rotated=True,
            use_pty=True,
        )
        gen_holder[0] = gen
        for line in gen:
            if stop_ev.is_set():
                break
            while True:
                try:
                    log_q.put_nowait(line)
                    break
                except queue.Full:
                    try:
                        log_q.get_nowait()
                    except queue.Empty:
                        pass
    except RemoteTailError as e:
        err_holder[0] = str(e)
    except Exception as e:
        err_holder[0] = f"SSH 스트림 오류: {e}"
    finally:
        gen_holder[0] = None
        if gen is not None:
            try:
                gen.close()
            except Exception:
                pass


def _ensure_session_defaults() -> None:
    if st.session_state.get("_logsentinel_inited"):
        return
    defaults: dict[str, Any] = {
        "ssh_host": os.getenv("SSH_HOST", ""),
        "ssh_user": os.getenv("SSH_USER", ""),
        "ssh_password": os.getenv("SSH_PASSWORD", ""),
        "log_path": os.getenv("LOG_PATH", "/var/log/auth.log"),
        "running": False,
        "log_buffer": "",
        "analysis_buffer": "",
        "worker_error": None,
        "line_count_since_llm": 0,
        "llm_futures": [],
        "llm_cooldown_until": 0.0,
        "vllm_model_id": TEAM_VLLM_MODEL,
        "stream_generation": 0,
        "manual_llm_future": None,
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)
    st.session_state.setdefault("llm_executor", ThreadPoolExecutor(max_workers=2))
    st.session_state.setdefault("log_autoscroll", True)
    st.session_state["_logsentinel_inited"] = True


def _validate_for_start() -> str | None:
    if not str(st.session_state.get("ssh_host", "")).strip():
        return "서버 IP/호스트를 입력하세요."
    if not str(st.session_state.get("ssh_user", "")).strip():
        return "사용자 ID를 입력하세요."
    if not str(st.session_state.get("log_path", "")).strip():
        return "로그 파일 경로를 입력하세요."
    return None


def _stop_ssh_worker() -> None:
    stop_ev: threading.Event | None = st.session_state.get("stop_ev")
    if stop_ev is not None:
        stop_ev.set()
    gh = st.session_state.get("ssh_gen_holder")
    if isinstance(gh, list) and gh and gh[0] is not None:
        try:
            gh[0].close()
        except Exception:
            pass
        gh[0] = None
    th: threading.Thread | None = st.session_state.get("ssh_thread")
    if th is not None and th.is_alive():
        th.join(timeout=5.0)
    st.session_state["ssh_thread"] = None
    st.session_state["stop_ev"] = None
    st.session_state["log_q"] = None
    st.session_state["ssh_err_holder"] = None
    st.session_state["ssh_gen_holder"] = None


def _ensure_ssh_thread_started() -> None:
    if not st.session_state.get("running"):
        return
    th: threading.Thread | None = st.session_state.get("ssh_thread")
    if th is not None and th.is_alive():
        return

    err = _validate_for_start()
    if err:
        st.session_state["worker_error"] = err
        st.session_state["running"] = False
        return

    st.session_state["worker_error"] = None
    st.session_state["log_q"] = queue.Queue(maxsize=LOG_QUEUE_MAX)
    st.session_state["stop_ev"] = threading.Event()
    st.session_state["ssh_err_holder"] = [None]
    st.session_state["ssh_gen_holder"] = [None]

    host = str(st.session_state["ssh_host"]).strip()
    user = str(st.session_state["ssh_user"]).strip()
    password = str(st.session_state["ssh_password"])
    path = str(st.session_state["log_path"]).strip()
    port = _ssh_port()

    t = threading.Thread(
        target=_ssh_tail_worker,
        kwargs={
            "host": host,
            "username": user,
            "password": password,
            "remote_path": path,
            "port": port,
            "stop_ev": st.session_state["stop_ev"],
            "log_q": st.session_state["log_q"],
            "err_holder": st.session_state["ssh_err_holder"],
            "gen_holder": st.session_state["ssh_gen_holder"],
        },
        daemon=True,
        name="logsentinel-ssh-tail",
    )
    st.session_state["ssh_thread"] = t
    t.start()


def _sync_ssh_worker() -> None:
    if st.session_state.get("running"):
        _ensure_ssh_thread_started()
    else:
        _stop_ssh_worker()


def _drain_log_queue() -> None:
    log_q: queue.Queue[str] | None = st.session_state.get("log_q")
    if log_q is None:
        return
    drained = 0
    while drained < 800:
        try:
            line = log_q.get_nowait()
        except queue.Empty:
            break
        buf = str(st.session_state.get("log_buffer", ""))
        buf += line + "\n"
        if len(buf) > MAX_LOG_BUFFER_CHARS:
            buf = buf[-MAX_LOG_BUFFER_CHARS:]
        st.session_state["log_buffer"] = buf
        st.session_state["line_count_since_llm"] = (
            int(st.session_state.get("line_count_since_llm", 0)) + 1
        )
        drained += 1


def _poll_ssh_errors() -> None:
    holder = st.session_state.get("ssh_err_holder")
    if isinstance(holder, list) and holder and holder[0]:
        st.session_state["worker_error"] = holder[0]
        holder[0] = None
        st.session_state["running"] = False
        _stop_ssh_worker()


def _schedule_llm_if_needed() -> None:
    if not st.session_state.get("running"):
        return
    if not llm_enabled():
        return
    if time.monotonic() < float(st.session_state.get("llm_cooldown_until") or 0):
        return
    futures: list[Future[str]] = list(st.session_state.get("llm_futures") or [])
    if len(futures) >= MAX_PENDING_LLM:
        return
    if int(st.session_state.get("line_count_since_llm", 0)) < LLM_BATCH_MIN_LINES:
        return

    buf = str(st.session_state.get("log_buffer", "")).strip()
    if not buf:
        return

    snippet = buf[-LLM_SNIPPET_CHARS:]
    st.session_state["line_count_since_llm"] = 0

    ex: ThreadPoolExecutor = st.session_state["llm_executor"]
    model_id = str(st.session_state.get("vllm_model_id") or TEAM_VLLM_MODEL).strip()
    gen = int(st.session_state.get("stream_generation", 0))

    def _run_batch(snip: str, mid: str, g: int) -> tuple[int, str]:
        return g, analyze_log_snippet(snip, model=mid)

    fut = ex.submit(_run_batch, snippet, model_id, gen)
    futures.append(fut)
    st.session_state["llm_futures"] = futures


def _poll_llm_futures() -> None:
    futures: list[Future[tuple[int, str]]] = list(
        st.session_state.get("llm_futures") or []
    )
    if not futures:
        return
    cur_gen = int(st.session_state.get("stream_generation", 0))
    kept: list[Future[tuple[int, str]]] = []
    for fut in futures:
        if not fut.done():
            kept.append(fut)
            continue
        try:
            gen, text = fut.result()
        except Exception as e:
            gen, text = cur_gen, f"분석 스레드 오류: {e}"
        if int(gen) != cur_gen:
            continue
        stamp = datetime.now().strftime("%H:%M:%S")
        ab = str(st.session_state.get("analysis_buffer", ""))
        ab += f"\n--- [{stamp}]\n{text}\n"
        if len(ab) > MAX_ANALYSIS_BUFFER_CHARS:
            ab = ab[-MAX_ANALYSIS_BUFFER_CHARS:]
        st.session_state["analysis_buffer"] = ab
        if (
            text.startswith("LLM API 오류")
            or text.startswith("LLM 오류")
            or text.startswith("분석 스레드 오류")
        ):
            st.session_state["llm_cooldown_until"] = (
                time.monotonic() + llm_error_cooldown_seconds()
            )
        else:
            st.session_state["llm_cooldown_until"] = 0.0
    st.session_state["llm_futures"] = kept


def _poll_manual_llm_future() -> None:
    fut: Future[str] | None = st.session_state.get("manual_llm_future")
    if fut is None or not isinstance(fut, Future):
        return
    if not fut.done():
        return
    st.session_state["manual_llm_future"] = None
    try:
        text = fut.result()
    except Exception as e:
        text = f"수동 질문 처리 오류: {e}"
    stamp = datetime.now().strftime("%H:%M:%S")
    ab = str(st.session_state.get("analysis_buffer", ""))
    ab += f"\n--- [수동 질문 {stamp}]\n{text}\n"
    if len(ab) > MAX_ANALYSIS_BUFFER_CHARS:
        ab = ab[-MAX_ANALYSIS_BUFFER_CHARS:]
    st.session_state["analysis_buffer"] = ab
    if (
        text.startswith("LLM API 오류")
        or text.startswith("LLM 오류")
        or text.startswith("수동 질문 처리 오류")
    ):
        st.session_state["llm_cooldown_until"] = (
            time.monotonic() + llm_error_cooldown_seconds()
        )


def _detect_log_path_change_and_reset() -> None:
    cur = str(st.session_state.get("log_path", "")).strip()
    if "_prev_log_path" not in st.session_state:
        st.session_state["_prev_log_path"] = cur
        return
    prev = str(st.session_state.get("_prev_log_path", "")).strip()
    if cur == prev:
        return
    st.session_state["log_buffer"] = ""
    st.session_state["analysis_buffer"] = ""
    st.session_state["line_count_since_llm"] = 0
    st.session_state["llm_futures"] = []
    st.session_state["llm_cooldown_until"] = 0.0
    st.session_state["worker_error"] = None
    st.session_state["stream_generation"] = int(
        st.session_state.get("stream_generation", 0)
    ) + 1
    st.session_state["manual_llm_future"] = None
    if st.session_state.get("running"):
        st.session_state["running"] = False
        _stop_ssh_worker()
    st.session_state["_prev_log_path"] = cur


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SQL_KW_RE = re.compile(
    r"\b(SELECT|FROM|WHERE|AND|OR|JOIN|INNER|LEFT|RIGHT|ON|"
    r"ORDER\s+BY|GROUP\s+BY|INSERT|INTO|VALUES|UPDATE|SET|DELETE|"
    r"LIMIT|HAVING|UNION|CASE|WHEN|THEN|ELSE|END)\b",
    re.IGNORECASE,
)


def _sql_keyword_spans(escaped_line: str) -> str:
    def repl(m: re.Match[str]) -> str:
        return (
            f'<span style="color:#67e8f9;font-weight:600">{m.group(0)}</span>'
        )

    return _SQL_KW_RE.sub(repl, escaped_line)


def _format_log_line_html(line: str) -> str:
    raw = _ANSI_ESCAPE_RE.sub("", line).rstrip("\r\n")
    esc = html.escape(raw)
    body = _sql_keyword_spans(esc)
    low = raw.lower()
    if any(
        x in low
        for x in ("error", "fatal", "exception", "critical", "denied", "failure")
    ):
        return f'<span style="color:#fca5a5">{body}</span>'
    if "warn" in low:
        return f'<span style="color:#fbbf24">{body}</span>'
    if " info" in low or low.startswith("info") or "[info]" in low:
        return f'<span style="color:#93c5fd">{body}</span>'
    return f'<span style="color:#e4e4e7">{body}</span>'


def _log_body_to_html(body: str) -> str:
    lines = (body or "").splitlines()
    if not lines:
        return '<span style="color:#71717a">(로그 없음)</span>'
    parts: list[str] = []
    for ln in lines:
        parts.append(_format_log_line_html(ln))
    return "<br>\n".join(parts)


def _maybe_scroll_log_bottom_into_view(enabled: bool) -> None:
    if not enabled:
        return
    components.html(
        """
        <script>
        const d = window.parent.document;
        const el = d.getElementById("logsentinel-log-bottom");
        if (el) {
          el.scrollIntoView({behavior: "auto", block: "end"});
        } else {
          const main = d.querySelector("section.main");
          if (main) { main.scrollTop = main.scrollHeight; }
        }
        </script>
        """,
        height=0,
        width=0,
    )


def _render_log_stream_block(body: str, *, height: int, autoscroll: bool) -> None:
    inner = _log_body_to_html(body)
    st.markdown(
        (
            f'<div id="logsentinel-log-wrap" style="max-height:{height}px;overflow:auto;'
            "font-family:ui-monospace,Consolas,monospace;font-size:0.82rem;"
            "white-space:pre-wrap;word-break:break-word;padding:0.6rem 0.75rem;"
            "border-radius:6px;border:1px solid rgba(128,128,128,0.28);"
            "background:rgba(15,23,42,0.35);"
            f'">{inner}<div id="logsentinel-log-bottom" style="height:1px;width:100%"></div></div>'
        ),
        unsafe_allow_html=True,
    )
    _maybe_scroll_log_bottom_into_view(autoscroll)


def _render_sidebar() -> None:
    st.sidebar.header("연결 설정")
    st.sidebar.subheader("SSH (원격 로그)")
    st.sidebar.text_input("서버 IP / 호스트", key="ssh_host")
    st.sidebar.text_input("사용자 ID", key="ssh_user")
    st.sidebar.text_input("비밀번호", type="password", key="ssh_password")

    st.sidebar.subheader("로그 파일 경로")
    st.sidebar.caption("프리셋 버튼을 누르면 아래 입력란에 경로가 채워집니다.")
    for idx, (label, path) in enumerate(LOG_PATH_PRESETS):
        if st.sidebar.button(
            label, key=f"log_preset_{idx}", use_container_width=True
        ):
            st.session_state["log_path"] = path
    st.sidebar.text_input(
        "직접 입력 (필요 시 수정)",
        key="log_path",
        placeholder="/var/log/messages",
    )

    st.sidebar.subheader("LLM (팀 공용)")
    st.sidebar.markdown(
        f"- **Base URL:** `{TEAM_VLLM_BASE_URL}`\n"
        f"- **등록 모델 목록:** [{team_vllm_models_url()}]({team_vllm_models_url()})"
    )
    _presets = list(LLM_MODEL_PRESETS)
    _labels = [p[0] for p in _presets]
    _ids = [p[1] for p in _presets]
    if TEAM_VLLM_MODEL not in _ids:
        _labels.insert(0, f"환경 기본 ({TEAM_VLLM_MODEL})")
        _ids.insert(0, TEAM_VLLM_MODEL)

    _cur = str(st.session_state.get("vllm_model_id") or TEAM_VLLM_MODEL).strip()
    if _cur not in _ids:
        st.session_state["vllm_model_id"] = TEAM_VLLM_MODEL
        _cur = TEAM_VLLM_MODEL
    _idx = _ids.index(_cur)

    _choice = st.sidebar.selectbox(
        "분석 모델",
        options=_labels,
        index=_idx,
        key="llm_model_select_label",
        disabled=not llm_enabled(),
    )
    st.session_state["vllm_model_id"] = _ids[_labels.index(_choice)]

    if not llm_enabled():
        st.sidebar.warning(
            "`DISABLE_LLM=1` 이 설정되어 **분석 API 호출이 꺼져** 있습니다. "
            "SSH 로그만 수집합니다."
        )
    _mid = str(st.session_state.get("vllm_model_id", ""))
    if _mid.startswith(("openai/", "ollama/")):
        st.sidebar.info(
            "`openai/`·`ollama/` 접두 모델은 LiteLLM **라우팅이 비어** 500이 나는 경우가 많습니다. "
            "불안하면 **Qwen3-VL (Base)** (`qwen3-vl`)을 쓰세요."
        )
    st.sidebar.caption(
        f"API `model`: `{st.session_state['vllm_model_id']}` — 첫 기본값은 `.env`의 "
        "`VLLM_MODEL`(없으면 코드 기본). `500`·`Connection error` 등은 백엔드·라우팅 이슈일 수 있음. "
        "로그만: `.env`에 `DISABLE_LLM=1`, 반복 간격: `LLM_COOLDOWN_AFTER_ERROR`(초)."
    )


def _render_manual_llm_section() -> None:
    st.subheader("LLM 질문")
    with st.expander("현재 화면에서 모델에게 직접 질문 (자동 배치 분석과 별도)", expanded=False):
        if not llm_enabled():
            st.warning("`DISABLE_LLM=1` 이면 API를 호출할 수 없습니다.")
            return
        fut = st.session_state.get("manual_llm_future")
        if isinstance(fut, Future) and not fut.done():
            st.info("이전 질문 처리 중입니다. 잠시만 기다려 주세요.")
        st.text_area(
            "질문 내용",
            key="manual_llm_question",
            height=110,
            placeholder="예: 방금 로그에서 반복되는 오류의 의미는?",
        )
        st.checkbox(
            "현재 로그 버퍼 끝부분을 맥락으로 함께 보내기",
            value=True,
            key="manual_llm_attach_logs",
        )
        if st.button("질문 보내기", key="btn_manual_llm_send"):
            q = str(st.session_state.get("manual_llm_question", "")).strip()
            if not q:
                st.warning("질문을 입력하세요.")
            else:
                pending = st.session_state.get("manual_llm_future")
                if isinstance(pending, Future) and not pending.done():
                    st.warning("처리 중인 질문이 있습니다.")
                else:
                    ctx = ""
                    if st.session_state.get("manual_llm_attach_logs", True):
                        buf = str(st.session_state.get("log_buffer", "")).strip()
                        if buf:
                            ctx = buf[-LLM_SNIPPET_CHARS:]
                    model_id = str(
                        st.session_state.get("vllm_model_id") or TEAM_VLLM_MODEL
                    ).strip()
                    ex: ThreadPoolExecutor = st.session_state["llm_executor"]
                    st.session_state["manual_llm_future"] = ex.submit(
                        answer_user_question,
                        q,
                        ctx,
                        model=model_id,
                    )


def _render_controls() -> None:
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("분석 시작", type="primary", key="btn_start"):
            v = _validate_for_start()
            if v:
                st.session_state["worker_error"] = v
            else:
                st.session_state["worker_error"] = None
                st.session_state["line_count_since_llm"] = 0
                st.session_state["llm_futures"] = []
                st.session_state["running"] = True
    with c2:
        if st.button("분석 중지", key="btn_stop"):
            st.session_state["running"] = False
    with c3:
        status_label = "실행 중" if st.session_state["running"] else "대기"
        st.metric("실시간 상태", status_label)


def _render_monospace_block(body: str, *, height: int = 420) -> None:
    """분석 피드 등 단색 monospace 블록."""
    safe = html.escape(body or " ")
    st.markdown(
        (
            f'<div style="max-height:{height}px;overflow:auto;'
            "font-family:ui-monospace,Consolas,monospace;font-size:0.82rem;"
            "white-space:pre-wrap;word-break:break-word;padding:0.6rem 0.75rem;"
            "border-radius:6px;border:1px solid rgba(128,128,128,0.28);"
            f'">{safe}</div>'
        ),
        unsafe_allow_html=True,
    )


def _render_log_analysis_panels(*, live: bool) -> None:
    if st.session_state.get("worker_error"):
        st.error(str(st.session_state["worker_error"]))

    _poll_manual_llm_future()

    if live:
        _drain_log_queue()
        _poll_ssh_errors()
        _schedule_llm_if_needed()
        _poll_llm_futures()

    log_text = str(st.session_state.get("log_buffer", ""))
    analysis_text = str(st.session_state.get("analysis_buffer", ""))

    left, right = st.columns(2, gap="medium")
    with left:
        st.subheader("실시간 로그 스트림")
        if live and st.session_state.get("ssh_thread") and st.session_state["ssh_thread"].is_alive():
            st.caption("SSH tail 수신 중… (버퍼는 0.35초마다 갱신)")
        st.checkbox(
            "맨 아래 자동 스크롤 (끄면 이전 줄을 스크롤로 확인)",
            key="log_autoscroll",
        )
        _render_log_stream_block(
            log_text or " ",
            height=420,
            autoscroll=bool(st.session_state.get("log_autoscroll", True)),
        )
    with right:
        st.subheader("AI 보안 분석 피드")
        n_pending = len(st.session_state.get("llm_futures") or [])
        n_lines = int(st.session_state.get("line_count_since_llm", 0))
        if llm_enabled():
            st.caption(
                f"자동 배치: `{n_lines}` / `{LLM_BATCH_MIN_LINES}` 줄마다 요청 · "
                f"대기 중 분석 `{n_pending}`건"
            )
        else:
            st.caption("LLM 비활성화(`DISABLE_LLM`) — 피드는 수동 질문만 표시됩니다.")
        _render_monospace_block(analysis_text or " ")


@st.fragment(run_every=0.35)
def _live_panels_fragment() -> None:
    _render_log_analysis_panels(live=True)


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    _ensure_session_defaults()

    st.title(PAGE_TITLE)
    st.caption(
        "원격 서버 로그를 SSH로 스트리밍하고, 팀 vLLM으로 보안 분석을 수행하는 대시보드입니다."
    )

    _render_sidebar()
    _detect_log_path_change_and_reset()
    _render_controls()
    _render_manual_llm_section()
    _sync_ssh_worker()

    if st.session_state.get("running") and hasattr(st, "fragment"):
        # fragment 첫 틱 전에도 한 번 비워 두어 첫 화면이 바로 채워지게 함
        _drain_log_queue()
        _poll_ssh_errors()
        _schedule_llm_if_needed()
        _poll_llm_futures()
        _live_panels_fragment()
    else:
        if st.session_state.get("running") and not hasattr(st, "fragment"):
            _drain_log_queue()
            _poll_ssh_errors()
            _schedule_llm_if_needed()
            _poll_llm_futures()
        _render_log_analysis_panels(live=False)
        if st.session_state.get("running") and not hasattr(st, "fragment"):
            time.sleep(0.5)
            st.rerun()


main()
