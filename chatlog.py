"""대화 기록 저장 (선택). .env 에 LOG_CHATS=1 을 넣으면 logs/chat_log.csv 에 질문·답변·근거가 쌓인다.

시범 운영에서 '직원들이 뭘 묻는지', '어떤 질문에 답을 못 했는지'를 보고 개선하는 용도.
※ 기록되는 것은 직원이 입력한 질문 전체이므로, 사용 전에 직원들에게 알려야 한다.
"""
import csv
import json
import threading
import time

import config

_lock = threading.Lock()
HEADER = ["시각", "질문", "답변", "근거(상위5)", "검색어", "소요시간"]


def log(question: str, answer: str, sources: list[dict], timing: dict | None, queries: list[str] | None = None) -> None:
    """기록 실패(엑셀로 파일을 열어 둔 경우 등)가 챗봇 동작을 막지 않도록 모든 오류를 삼킨다."""
    if not config.LOG_CHATS:
        return
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = config.LOG_DIR / "chat_log.csv"
        with _lock:
            new = not path.exists()
            with path.open("a", newline="", encoding="utf-8-sig") as f:  # 엑셀에서 한글이 안 깨지도록 BOM
                w = csv.writer(f)
                if new:
                    w.writerow(HEADER)
                w.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    question,
                    answer,
                    " | ".join(s["ref"] for s in sources[:5]),
                    " | ".join(queries or []),
                    json.dumps({k: round(v, 1) for k, v in (timing or {}).items()}, ensure_ascii=False),
                ])
    except Exception:
        pass
