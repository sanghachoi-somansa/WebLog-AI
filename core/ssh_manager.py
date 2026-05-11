"""SSH로 원격 로그 tail 스트림을 줄 단위로 받는 모듈."""

from __future__ import annotations

import shlex
import socket
import time
from collections.abc import Iterator

import paramiko


class RemoteTailError(RuntimeError):
    """원격 tail 실행 또는 스트림 수신 중 오류."""


def iter_remote_tail_lines(
    host: str,
    username: str,
    password: str,
    remote_path: str,
    *,
    port: int = 22,
    connect_timeout: float = 30.0,
    initial_lines: int = 100,
    follow_rotated: bool = True,
    use_pty: bool = True,
) -> Iterator[str]:
    """Paramiko로 SSH 접속 후 `tail`로 로그를 따라가며 한 줄씩 반환한다.

    - ``initial_lines``: 마지막 N줄을 먼저 보낸 뒤 새 줄을 계속 받는다. ``0``이면
      이미 쌓인 내용 없이 새로 추가되는 줄만 받는다.
    - ``follow_rotated``가 True이면 ``-F``(로테이션 시 재시도), False이면 ``-f``.

    제너레이터를 닫거나 소진하면 SSH 세션을 정리한다.

    Args:
        host: 서버 호스트명 또는 IP.
        username: SSH 사용자.
        password: SSH 비밀번호.
        remote_path: 원격 로그 파일 절대 경로.
        port: SSH 포트 (기본 22).
        connect_timeout: TCP·SSH 핸드셰이크 제한(초).
        initial_lines: tail 시작 시 출력할 마지막 줄 수.
        follow_rotated: ``tail -F`` 사용 여부.
        use_pty: True이면 의사 터미널에서 실행해 ``tail`` 출력이 줄 단위로
            바로 흘러오기 쉽다(기본). 일부 환경에서만 False로 끈다.

    Yields:
        개행을 제거한 UTF-8 디코딩 한 줄씩.

    Raises:
        RemoteTailError: 연결·인증 실패, tail 비정상 종료 등.
        paramiko.SSHException: 하위 Paramiko 오류를 그대로 전달할 수 있음.
    """
    path = remote_path.strip()
    if not path:
        raise RemoteTailError("로그 경로가 비어 있습니다.")

    flag = "-F" if follow_rotated else "-f"
    n = max(0, int(initial_lines))
    # POSIX 셸 한 줄 명령; 경로는 인용해 주입 방지
    cmd = f"tail -n {n} {flag} {shlex.quote(path)}"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        try:
            client.connect(
                hostname=host,
                port=port,
                username=username,
                password=password,
                timeout=connect_timeout,
                banner_timeout=connect_timeout,
                auth_timeout=connect_timeout,
                allow_agent=False,
                look_for_keys=False,
            )
        except (socket.error, paramiko.SSHException, EOFError) as e:
            raise RemoteTailError(f"SSH 연결 실패: {e}") from e

        try:
            _stdin, stdout, stderr = client.exec_command(cmd, get_pty=use_pty)
        except paramiko.SSHException as e:
            raise RemoteTailError(f"원격 명령 실행 실패: {e}") from e

        ch = stdout.channel
        try:
            yield from _read_tail_stdout(stdout, stderr, ch)
        finally:
            try:
                ch.close()
            except Exception:
                pass
    finally:
        client.close()


def _read_tail_stdout(
    stdout: paramiko.ChannelFile,
    stderr: paramiko.ChannelFile,
    channel: paramiko.Channel,
) -> Iterator[str]:
    """stdout에서 줄 단위로 읽고, 채널 종료 시 stderr를 모아 예외를 만든다."""
    while True:
        try:
            line = stdout.readline()
        except socket.timeout as e:
            raise RemoteTailError("소켓 읽기 시간 초과") from e

        if line:
            if isinstance(line, bytes):
                text = line.decode("utf-8", errors="replace")
            else:
                text = str(line)
            yield text.rstrip("\r\n")
            continue

        # 빈 줄: EOF 또는 아직 데이터 없음
        if channel.exit_status_ready():
            code = channel.recv_exit_status()
            err_bytes = stderr.read() if stderr else b""
            err = (
                err_bytes.decode("utf-8", errors="replace").strip()
                if err_bytes
                else ""
            )
            if code != 0:
                raise RemoteTailError(
                    f"tail 종료(exit {code})" + (f": {err}" if err else "")
                )
            if err:
                raise RemoteTailError(f"tail 종료: {err}")
            return

        # exit 준비 안 됨인데 빈 readline → 잠시 대기 없이 루프; tail -f는 블로킹이 정상
        # 채널이 닫혔는데 exit_status가 아직 안 올라온 경우
        if channel.closed:
            err_bytes = stderr.read() if stderr else b""
            err = (
                err_bytes.decode("utf-8", errors="replace").strip()
                if err_bytes
                else ""
            )
            raise RemoteTailError(
                "스트림이 예기치 않게 닫혔습니다."
                + (f" stderr: {err}" if err else "")
            )

        # readline이 빈 값을 돌려줬는데 아직 종료 신호가 없는 비정상 경우
        time.sleep(0.05)
