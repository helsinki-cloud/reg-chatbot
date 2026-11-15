"""공개 배포용 사용량 제한. 누군가 URL을 퍼뜨려 무료 API 한도를 하루치 다 써버리는 것을 막는다.

- DAILY_LIMIT  : 하루(한국시간 기준) 전체 질문 수 상한
- SESSION_LIMIT: 접속 1회(브라우저 탭)당 질문 수 상한
주의: 앱이 재시작(절전 후 깨어남, 재배포)되면 하루 카운터는 0으로 돌아갑니다. 완벽한 방어가 아니라 '과열 방지'용입니다.
"""
import threading
from datetime import datetime, timedelta, timezone

import config

_KST = timezone(timedelta(hours=9))
_lock = threading.Lock()
_state = {"day": "", "count": 0}


def _today() -> str:
    return datetime.now(_KST).strftime("%Y-%m-%d")


def today_count() -> int:
    with _lock:
        return _state["count"] if _state["day"] == _today() else 0


def try_consume(session_count: int) -> tuple[bool, str]:
    """질문 1건을 쓸 수 있으면 (True, ''), 아니면 (False, 사유). 허용될 때만 하루 카운터를 올린다."""
    if config.SESSION_LIMIT and session_count >= config.SESSION_LIMIT:
        return False, f"한 번 접속에서 할 수 있는 질문은 {config.SESSION_LIMIT}개까지입니다. 페이지를 새로고침하면 다시 이용할 수 있습니다."
    with _lock:
        if _state["day"] != _today():
            _state["day"], _state["count"] = _today(), 0
        if config.DAILY_LIMIT and _state["count"] >= config.DAILY_LIMIT:
            return False, "오늘 이용 가능한 질문 수를 모두 사용했습니다. 내일 다시 이용해 주세요."
        _state["count"] += 1
    return True, ""
