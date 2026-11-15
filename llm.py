"""답변 생성 (OpenAI 호환 API). Qwen 로컬(vLLM 등)·클라우드 제공사 모두 base_url만 바꾸면 된다."""
import json
import re
from dataclasses import dataclass
from typing import Iterable, Iterator

import requests

from openai import OpenAI

import config
from retriever import Hit

_client = None
THINKING = object()  # "AI가 생각 중" 신호 (화면에 진행 상태를 보여주기 위한 표시용)


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY, timeout=120, max_retries=1)
    return _client


SYSTEM = f"""당신은 {config.ORG_NAME}의 규정 안내 챗봇입니다. 직원의 질문에 [참고 규정]을 읽고 정확하고 구체적으로 답합니다.

답하는 방법:
1. 질문에 물음이 여러 개면 모두 답하세요. 예를 들어 "상여금 있나? 추가로 나오는 거 있어?"는 '상여금 여부'와 '다른 지급 항목' 두 가지를 모두 답해야 합니다.
2. "있나요?", "뭐 나와요?"처럼 범위를 묻는 질문은 한 항목에 "없다"고만 답하지 말고, [참고 규정]에서 관련된 항목을 모두 찾아 알려주세요. 특정 월·시기를 묻는다면 그 시기에 해당하는 지급 항목을 빠짐없이 찾으세요. 설날은 1월 하순~2월, 추석은 9~10월에 있으므로, 해당 월 질문에서 명절 관련 지급 항목이 [참고 규정]에 있으면 함께 안내하세요.
3. 첫 문장은 결론을 자연스러운 완결 문장으로 쓰세요. 예: "2월에는 상여금은 없지만, 명절휴가비는 지급됩니다." 이어서 항목별 금액·비율·조건을 설명하세요. 항목이 3개 이상일 때만 "- "로 목록을 쓰세요.
4. 날짜·일수·금액·비율은 규정에 적힌 그대로 쓰고, "다만" 같은 예외·조건도 함께 알려주세요.
5. 문장이 "별표 N"이나 "제N조"를 가리키기만 하면, 그 별표·조문의 내용이 [참고 규정]에 있는지 찾아서 읽고 구체적인 내용(수당 이름, 금액, 비율, 기간 등)을 직접 풀어서 쓰세요. "별표 N에 따른다"라고만 쓰고 끝내면 안 됩니다. 그 내용이 [참고 규정]에 없으면 "별표 N의 구체적인 내용은 이번 검색에서 찾지 못했습니다"라고 분명히 밝히세요.
6. 문장 끝에 근거 번호를 [1]처럼 붙이세요.
7. 표는 구분(예: 질병휴직/육아휴직)에 따라 답이 달라질 수 있으니 구분별로 나누어 설명하세요.
8. [참고 규정] 어디에도 관련 내용이 전혀 없을 때만 "제공된 규정에서는 해당 내용을 확인할 수 없습니다."라고 답하세요. 규정에 없는 내용을 지어내지 마세요. 한국어로 답하세요.
9. 규정에 지급 조건이나 시기가 적혀 있는데 질문이 그 조건에 해당하지 않으면, "확인할 수 없다"가 아니라 "해당하지 않는다/지급되지 않는다"고 근거와 함께 분명히 답하세요. (예: 명절휴가비가 설날·추석이 속한 달에만 지급된다고 적혀 있으면, 11월에는 지급되지 않는다고 답합니다.)
10. 이전 대화는 "그럼", "그건" 같은 생략된 말을 이해하는 데만 참고하세요. 이전에 답하지 못했거나 이미 끝난 질문, 현재 질문과 상관없는 주제는 답변에서 언급하지 마세요. 오직 [질문]에만 답하세요.
11. 근거 번호는 그 문장의 실제 근거가 된 것만 붙이세요. 모든 번호를 한꺼번에 나열하지 마세요.

예시 1)
[참고 규정]
[1] 복무규정 제4조(경조사 휴가)
본인 결혼은 별표 2에 따른다.
[2] 복무규정 별표 2(경조사 휴가 일수)
| 구분 | 일수 |
| 본인 결혼 | 5일 |
[질문]
결혼하면 휴가 며칠이야?
[답변]
본인 결혼 시 경조사 휴가는 5일입니다. 제4조가 일수를 별표 2에 따르도록 하고 있고 [1], 별표 2에 본인 결혼 5일이 규정되어 있습니다 [2].

예시 2)
[참고 규정]
[1] 샘플규정 제10조(상여금)
3월, 9월에 봉급의 40%를 상여금으로 지급한다.
[2] 샘플규정 제11조(명절휴가비)
설날과 추석에 봉급의 60%를 명절휴가비로 지급한다.
[질문]
2월에 상여금 나와? 추가로 나오는 건 없어?
[답변]
2월에는 상여금이 지급되지 않습니다. 상여금은 3월과 9월에 봉급의 40%가 지급됩니다 [1]. 다만 설날이 있는 달(보통 1월 하순~2월)에는 명절휴가비로 봉급의 60%가 지급됩니다 [2]."""


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] {h.chunk.ref}" + (" (본문이 인용한 관련 조문)" if h.via == "ref" else "") + f"\n{h.chunk.text}"
        for i, h in enumerate(hits, 1)
    )


def fit_hits(hits: list[Hit], max_chars: int | None = None) -> list[Hit]:
    """AI에 보내는 근거 분량을 제한한다. 무료 외부 AI는 분당 처리량 한도가 있어서, 넘치면 순위가 낮은 것부터 뺀다."""
    max_chars = max_chars or config.LLM_CONTEXT_CHARS
    out, used = [], 0
    for h in hits:
        n = len(h.chunk.text) + len(h.chunk.ref) + 12
        if out and used + n > max_chars:
            continue
        out.append(h)
        used += n
    return out


def _strip_think(s: str) -> str:
    return re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()


def strip_think_stream(deltas: Iterable[str]) -> Iterator[str]:
    """Qwen3가 내보내는 <think>…</think> 사고 과정을 스트림에서 걸러낸다."""
    buf, state = "", "probe"
    for d in deltas:
        if not isinstance(d, str):
            yield d
            continue
        if not d:
            continue
        if state == "pass":
            yield d
            continue
        buf += d
        if state == "probe":
            s = buf.lstrip()
            if not s or "<think>".startswith(s):
                continue  # 아직 판단 불가
            if s.startswith("<think>"):
                state = "think"
            else:
                state = "pass"
                yield buf
                buf = ""
                continue
        if state == "think" and "</think>" in buf:
            rest = buf.split("</think>", 1)[1].lstrip("\n")
            state, buf = "pass", ""
            if rest:
                yield rest
    if state == "probe" and buf:
        yield buf


def _no_think(text: str) -> str:
    """Qwen3 하이브리드 모델은 메시지 끝의 /no_think 로 '생각' 단계를 건너뛴다."""
    return f"{text}\n\n/no_think" if config.LLM_NO_THINK else text


def use_ollama_native() -> bool:
    b = config.LLM_BACKEND
    if b in ("ollama", "openai"):
        return b == "ollama"
    return ":11434" in config.LLM_BASE_URL


def _ollama_root() -> str:
    u = config.LLM_BASE_URL.rstrip("/")
    return u[:-3] if u.endswith("/v1") else u


def _ollama_post(messages: list[dict], stream: bool, max_tokens: int, temperature: float):
    """Ollama 자체 API(/api/chat). think=False로 '생각' 단계를 공식적으로 끄고, num_ctx로 읽는 길이를 늘린다."""
    body = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "stream": stream,
        "think": False,
        "options": {"num_ctx": config.OLLAMA_NUM_CTX, "temperature": temperature, "num_predict": max_tokens},
    }
    r = requests.post(f"{_ollama_root()}/api/chat", json=body, stream=stream, timeout=(10, 600))
    if r.status_code != 200 and "think" in r.text.lower():
        # 생각 기능이 없는 모델(qwen2.5 등)이 think 옵션을 거부하면 옵션을 빼고 다시 보낸다
        r.close()
        body.pop("think")
        r = requests.post(f"{_ollama_root()}/api/chat", json=body, stream=stream, timeout=(10, 600))
    if r.status_code != 200:
        msg = r.text[:300]
        r.close()
        raise RuntimeError(f"Ollama 오류 {r.status_code}: {msg}")
    return r


def _ollama_stream(messages: list[dict], max_tokens: int, temperature: float) -> Iterator:
    r = _ollama_post(messages, True, max_tokens, temperature)
    try:
        for line in r.iter_lines():
            if not line:
                continue
            d = json.loads(line)
            if "error" in d:
                raise RuntimeError(f"Ollama 오류: {d['error']}")
            msg = d.get("message") or {}
            if msg.get("thinking"):
                yield THINKING
            if msg.get("content"):
                yield msg["content"]
            if d.get("done"):
                break
    finally:
        r.close()


def _openai_stream(resp) -> Iterator:
    for c in resp:
        if not c.choices:
            continue
        delta = c.choices[0].delta
        extra = getattr(delta, "model_extra", None) or {}
        if extra.get("reasoning") or extra.get("reasoning_content"):  # 생각 내용이 별도 칸으로 오는 서버
            yield THINKING
        if delta.content:
            yield delta.content


def complete(system: str, user: str, max_tokens: int = 300, temperature: float = 0.0) -> str:
    if use_ollama_native():
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        r = _ollama_post(msgs, False, max_tokens, temperature)
        return _strip_think((r.json().get("message") or {}).get("content", ""))
    r = client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": _no_think(user)}],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=config.LLM_EXTRA_BODY,
    )
    return _strip_think(r.choices[0].message.content or "")


def condense(history: list[dict], question: str) -> str:
    """후속 질문('그럼 그건 며칠이야?')을 단독으로 검색 가능한 문장으로 다시 쓴다. 실패하면 원문 그대로.

    기본값(CONDENSE_WITH_LLM=0)은 LLM을 부르지 않는다: 짧은 질문이면 직전 질문을 앞에 붙여 검색한다.
    """
    if not history:
        return question
    if not config.CONDENSE_WITH_LLM:
        last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        return f"{last_user} {question}" if last_user and len(question) <= 25 else question
    convo = "\n".join(f"{'직원' if m['role'] == 'user' else '챗봇'}: {m['content'][:300]}" for m in history[-4:])
    try:
        out = complete(
            "대화 맥락을 참고해, 마지막 질문을 맥락 없이도 이해되는 한 문장짜리 검색용 질문으로 다시 쓰세요. "
            "답변하지 말고 다시 쓴 질문 한 문장만 출력하세요. 이미 단독으로 이해되면 그대로 출력하세요.",
            f"[대화]\n{convo}\n\n[마지막 질문]\n{question}",
            max_tokens=150,
        )
        lines = [ln for ln in out.splitlines() if ln.strip()]
        return lines[0].strip() if lines else question
    except Exception:
        return question


def clean_history(messages: list[dict]) -> list[dict]:
    """AI에게 넘길 대화 이력에서 '답이 없거나 실패한 턴'을 질문까지 한 쌍으로 뺀다.

    답변만 지우고 질문을 남기면, AI가 '답을 못 한 질문'을 다음 답변에서 다시 짚는다.
    """
    out: list[dict] = []
    for m in messages:
        c = m.get("content", "")
        bad = m["role"] == "assistant" and ("확인할 수 없습니다" in c or "답변이 중지되었습니다" in c or c.startswith("⚠"))
        if bad:
            if out and out[-1]["role"] == "user":
                out.pop()
            continue
        out.append(m)
    return out


@dataclass
class Plan:
    standalone: str       # 혼자서도 뜻이 통하는 검색 질문
    extras: list[str]     # 보충 검색어
    followup: bool        # 이전 대화를 봐야만 뜻이 통하는 후속 질문인가


def use_expansion() -> bool:
    v = config.QUERY_EXPANSION
    if v in ("0", "1"):
        return v == "1"
    return not use_ollama_native()  # auto: 느린 PC 모델(Ollama)에서는 끄고, 빠른 외부 AI에서는 켠다


PLAN_SYSTEM = """직원이 기관 규정집에 대해 질문했습니다. 규정집에서 답을 찾기 위한 검색어를 만드세요.

출력: 정확히 5줄. 번호·기호·설명 없이 한 줄에 하나.
1줄: "후속: 예" 또는 "후속: 아니오". 질문이 이전 대화를 봐야만 뜻이 통하면(예: "그럼 2월은?", "그건 며칠이야?") 예, 혼자서도 뜻이 통하는 새 질문이면 아니오.
2줄: 대화 맥락을 반영해 혼자서도 뜻이 통하는 완전한 질문 한 문장.
3~5줄: 답에 함께 필요할 수 있는 규정 용어 검색어(각 25자 이내). 질문에 직접 나온 단어뿐 아니라, 답에 같이 들어갈 관련 항목·동의어도 쓰세요.

참고:
- 월·시기를 묻는 질문은 그때 지급될 수 있는 모든 보수 항목(상여금, 명절휴가비, 정근수당, 가족수당, 학비보조, 급식보조 등)을 각각 검색어로 쓰세요. 설날은 1~2월, 추석은 9~10월입니다.
- 휴직·휴가·출장·퇴직 같은 상황 질문은 '지급기준', '별표', '정산' 같은 표현을 곁들이세요.
- 질문에 없는 사실을 지어내지 마세요.

예)
[질문] 육아휴직 중에 월급 나와?
후속: 아니오
육아휴직 중 보수와 수당이 지급되는가?
휴직자 보수 지급기준 별표
육아휴직 봉급 수당 지급 여부
복직 후 보수 정산"""


def plan(history: list[dict], question: str) -> Plan:
    """질문 -> (검색 질문, 보충 검색어 최대 3개, 후속 질문 여부). AI 호출 1회(빠른 AI에서 약 1초).

    '2월에 뭐 나와?'처럼 답이 여러 조문에 흩어진 질문도, '설날 명절휴가비' 같은 보충 검색어 덕분에 찾을 수 있다.
    '후속 질문 여부'는 답변 단계에서 이전 대화를 AI에게 보여줄지 정하는 데 쓴다. (새 주제 질문이면 이력을 보내지 않는다)
    실패하거나 꺼져 있으면 기존 방식(질문 그대로 / 짧은 후속질문은 직전 질문과 합침)으로 돌아간다.
    """
    base = condense(history, question)
    rule_followup = bool(history) and base != question  # 규칙 방식: 짧은 질문이면 직전 질문과 합쳐서 달라진다
    if not use_expansion():
        return Plan(base, [], rule_followup)
    convo = "\n".join(f"{'직원' if m['role'] == 'user' else '챗봇'}: {m['content'][:200]}" for m in history[-4:])
    user = (f"[대화]\n{convo}\n\n" if convo else "") + f"[질문] {question}"
    try:
        out = complete(PLAN_SYSTEM, user, max_tokens=220, temperature=0.0)
    except Exception:
        return Plan(base, [], rule_followup)
    followup = None
    lines = []
    for ln in out.splitlines():
        ln = re.sub(r"^\s*(?:[-*•·]|\d+[\.\)])\s*", "", ln).strip()
        m = re.match(r"^후속\s*[:：]\s*(예|아니오|아니요)", ln)
        if m:
            followup = m.group(1) == "예"
            continue
        if 2 <= len(ln) <= 90 and not ln.startswith(("[", "출력", "검색어")):
            lines.append(ln)
    if not lines:
        return Plan(base, [], rule_followup)
    if followup is None:
        followup = rule_followup  # AI가 표시를 빠뜨리면 규칙으로 판단
    standalone, extras = lines[0], lines[1:4]
    if question.strip() and question.strip() != standalone and not history and len(question) <= 90:
        extras.append(question.strip())  # 직원이 쓴 원래 표현도 검색어로 함께 쓴다
    return Plan(standalone, extras, bool(history) and followup)


def plan_queries(history: list[dict], question: str) -> tuple[str, list[str]]:
    """(호환용) plan() 의 앞의 두 값만."""
    p = plan(history, question)
    return p.standalone, p.extras


def _extra_body(deep: bool):
    extra = dict(config.LLM_EXTRA_BODY or {})
    if deep:
        extra["reasoning_effort"] = config.DEEP_EFFORT
    return extra or None


def _too_large(e: Exception) -> bool:
    low = str(e).lower()
    return "request too large" in low or "too large" in low or "413" in low


def _limit_numbers(e: Exception):
    """오류문의 'Limit 8000, Requested 9100' 에서 (한도, 요청량)을 꺼낸다. 없으면 (None, None)."""
    m = re.search(r"limit\s*(\d+).*?requested\s*(\d+)", str(e), flags=re.I | re.S)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _hits_chars(hits: list[Hit]) -> int:
    return sum(len(h.chunk.text) + len(h.chunk.ref) + 12 for h in hits)


def _shrink_hits(hits: list[Hit], e: Exception) -> list[Hit]:
    """요청이 너무 크다는 오류를 받으면, 서버가 알려준 한도/요청량 비율만큼 근거를 줄인다(최소 1개 유지)."""
    limit, requested = _limit_numbers(e)
    factor = min(0.8, max(0.3, (limit / requested) * 0.8)) if limit and requested else 0.6
    shrunk = fit_hits(hits, max_chars=int(_hits_chars(hits) * factor))
    if len(shrunk) >= len(hits) and len(hits) > 1:  # 줄지 않았으면 마지막 것을 뺀다
        shrunk = hits[:-1]
    return shrunk


def _build_messages(question: str, hits: list[Hit], history: list[dict], no_think: bool,
                    context_question: str | None = None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM}]
    # 이전 대화는 맥락용이라 길게 필요 없다 (토큰 절약). 새 주제 질문이면 호출하는 쪽에서 빈 이력을 넘긴다.
    messages += [{"role": m["role"], "content": m["content"][:400]} for m in history[-4:]]
    q = question
    if context_question and context_question.strip() != question.strip():
        q = f"{question}\n(앞선 대화를 반영하면 이 질문의 뜻: {context_question})"
    user_msg = f"[참고 규정]\n{format_context(hits)}\n\n[질문]\n{q}"
    messages.append({"role": "user", "content": _no_think(user_msg) if no_think else user_msg})
    return messages


LAST: dict = {"hits": [], "deep": False}  # 마지막 답변에 실제로 쓴 근거/모드 (화면 안내용)


def stream_answer(question: str, hits: list[Hit], history: list[dict], deep: bool = False,
                  context_question: str | None = None) -> Iterator[str]:
    """답변 스트리밍.

    deep=True(정밀 모드): AI가 답하기 전에 더 생각하게 한다(OpenAI 호환 서버만).
    - 서버가 정밀 모드 옵션을 거부하면 일반 모드로 자동 재시도
    - '요청이 너무 크다(분당 한도 초과)' 오류면, 정밀 모드를 끄고 근거를 줄여서 자동 재시도 (최대 2번)
    """
    if use_ollama_native():
        messages = _build_messages(question, hits, history, no_think=False, context_question=context_question)
        LAST.update(hits=hits, deep=False)
        yield from strip_think_stream(_ollama_stream(messages, config.LLM_MAX_TOKENS, 0.1))
        return

    cur_hits, cur_deep = hits, deep
    resp = None
    for attempt in range(3):
        messages = _build_messages(question, cur_hits, history, no_think=True, context_question=context_question)
        try:
            resp = client().chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=int(config.LLM_MAX_TOKENS * (1.5 if cur_deep else 1)),
                stream=True,
                extra_body=_extra_body(cur_deep),
            )
            break
        except Exception as e:
            low = str(e).lower()
            if _too_large(e) and attempt < 2:
                cur_deep, cur_hits = False, _shrink_hits(cur_hits, e)
                continue
            if cur_deep and not _too_large(e) and ("400" in low or "reasoning" in low):
                cur_deep = False  # 이 모델이 정밀 모드 옵션을 못 받으면 일반 모드로
                continue
            raise
    LAST.update(hits=cur_hits, deep=cur_deep)
    try:
        yield from strip_think_stream(_openai_stream(resp))
    finally:
        resp.close()  # 중지했을 때 서버 쪽 생성도 끊는다


def explain_error(e: Exception) -> str:
    """AI 서버 오류를 직원/시연용으로 알아듣기 쉬운 한국어 안내로 바꾼다. (원문은 뒤에 짧게 덧붙임)"""
    raw = f"{type(e).__name__}: {e}"
    low = raw.lower()
    limit, requested = _limit_numbers(e)
    if _too_large(e):  # 주의: 서버가 429로 알려줘도 '요청 크기' 문제면 기다려도 소용없다
        nums = f" (한도 {limit:,} / 이번 요청 {requested:,})" if limit and requested else ""
        msg = (f"한 번에 보내는 내용이 너무 커서 무료 한도를 넘었습니다{nums}. 기다려도 해결되지 않으니, "
               "왼쪽에서 '정밀 모드'를 끄거나 '근거 조문 수'를 줄여서 다시 질문해 주세요.")
    elif ("429" in low or "quota" in low or "rate limit" in low or "ratelimit" in low or "resource_exhausted" in low) and (
            "perday" in low or "per day" in low or "daily" in low or "per_day" in low):
        msg = "오늘 무료 사용량(하루 한도)을 모두 썼습니다. 내일 다시 이용하거나, 다른 AI 설정으로 바꿔 주세요."
    elif "429" in low or "quota" in low or "rate limit" in low or "ratelimit" in low or "resource_exhausted" in low:
        msg = "무료 사용 한도(분당 처리량)에 잠시 걸렸습니다. 30~60초 뒤에 다시 질문해 주세요."
    elif "401" in low or "invalid api key" in low or "authentication" in low:
        msg = "AI 서버 인증에 실패했습니다. .env 의 LLM_API_KEY 를 확인해 주세요."
    elif "404" in low or "does not exist" in low:
        msg = "AI 모델 이름을 찾을 수 없습니다. .env 의 LLM_MODEL 을 확인해 주세요. (python check_llm.py 로 목록 확인)"
    elif "connection" in low or "timeout" in low or "timed out" in low:
        msg = "AI 서버에 연결하지 못했습니다. 인터넷 연결과 .env 의 LLM_BASE_URL 을 확인해 주세요."
    else:
        msg = "답변 생성 중 오류가 발생했습니다."
    return f"{msg}\n\n(상세: {raw[:300]})"
