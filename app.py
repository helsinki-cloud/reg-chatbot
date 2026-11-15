"""Streamlit 챗봇 화면.  실행: streamlit run app.py"""
import hmac
import time

import streamlit as st

import chatlog
import config
import llm
import usage
from retriever import Retriever

st.set_page_config(page_title=config.APP_TITLE, page_icon="📘", layout="wide")

# ── 접속 암호 (APP_PASSWORD 가 설정된 경우만). 검색 모델을 불러오기 전에 먼저 확인한다 ──
if config.APP_PASSWORD and not st.session_state.get("authed"):
    st.title(f"📘 {config.APP_TITLE}")
    pw = st.text_input("접속 암호를 입력하세요", type="password")
    if pw:
        if hmac.compare_digest(pw.encode(), config.APP_PASSWORD.encode()):
            st.session_state["authed"] = True
            st.rerun()
        st.error("암호가 맞지 않습니다.")
    st.stop()

NOT_FOUND = "제공된 규정에서는 해당 내용을 확인할 수 없습니다. 담당 부서에 문의해 주세요."
STOPPED = "⏹ *답변이 중지되었습니다.*"


@st.cache_resource(show_spinner="검색 모델을 불러오는 중입니다 (최초 1회, 시간이 조금 걸려요)...")
def load_retriever():
    r = Retriever()
    r.search("워밍업", top_k=1)  # 첫 질문이 느리지 않도록 미리 한 번 돌려둔다
    return r


def render_sources(sources: list[dict], timing: dict | None = None, queries: list[str] | None = None):
    if not sources:
        return
    with st.expander(f"📎 근거 조문 {len(sources)}건 보기"):
        if queries and len(queries) > 1:
            st.caption("🔎 검색어: " + "  |  ".join(queries))
        for i, s in enumerate(sources, 1):
            tag = "  · 본문이 인용한 관련 조문" if s["via"] == "ref" else (f"  · 관련도 {s['score']:.2f}" if s["via"] == "rerank" else "")
            st.markdown(f"**[{i}] {s['ref']}**{tag}")
            st.text(s["text"])
    if timing:
        st.caption(" · ".join(f"{k} {v:.1f}초" for k, v in timing.items()))


# ── 준비 ──
try:
    retriever = load_retriever()
except FileNotFoundError:
    st.error("검색 인덱스가 없습니다. 먼저 터미널에서 `python ingest.py` 를 실행하세요.")
    st.stop()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 직전 실행이 중간에 끊겼으면(중지 버튼 / 답변 중 새 질문 입력) 그 질문을 '중지됨'으로 기록해 둔다
if "running" in st.session_state:
    run = st.session_state.pop("running")
    partial = run.get("partial", "").strip()
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"{partial}\n\n{STOPPED}" if partial else STOPPED,
        "sources": run.get("sources", []),
        "timing": None,
    })

# ── 사이드바 ──
with st.sidebar:
    st.header("설정")
    top_k = st.slider(
        "근거 조문 수", 3, 10, config.TOP_K,
        help="AI에게 읽혀서 답변 근거로 쓰게 할 조문 개수입니다. "
             "늘리면 여러 조문에 걸친 질문을 놓칠 확률이 줄지만 답변이 느려지고 엉뚱한 조문이 섞일 수 있습니다. "
             "줄이면 빨라지지만 필요한 조문이 빠질 수 있습니다.",
    )
    deep = st.toggle(
        "🧠 정밀 모드", value=config.DEEP_DEFAULT,
        help="켜면 AI가 답하기 전에 더 깊이 생각합니다. 답이 더 정확하고 빠짐없지만 몇 초 더 걸리고, 무료 사용 한도를 더 씁니다.",
    )
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()
    if config.EXAMPLE_QUESTIONS:
        st.subheader("예시 질문")
        for i, q in enumerate(config.EXAMPLE_QUESTIONS):
            if st.button(q, key=f"ex{i}", use_container_width=True):
                st.session_state.pending = q
    if config.APP_NOTICE:
        st.info(config.APP_NOTICE)
    st.caption(f"문서 조각 {len(retriever.chunks)}개 색인됨")
    st.caption("답변은 등록된 규정만 근거로 합니다. 중요한 사안은 담당 부서에 최종 확인하세요.")

# ── 본문 ──
st.title(f"📘 {config.APP_TITLE}")
st.caption("규정·지침에 대해 궁금한 점을 물어보세요. 답변마다 근거 조문을 함께 보여드립니다.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources", []), m.get("timing"), m.get("queries"))

prompt = st.chat_input("질문을 입력하세요") or st.session_state.pop("pending", None)
if prompt:
    ok, why = usage.try_consume(st.session_state.get("n_asked", 0))
    if not ok:
        st.warning(why)
        st.stop()
    st.session_state["n_asked"] = st.session_state.get("n_asked", 0) + 1
    # 답이 없거나 실패한 턴은 (질문까지 한 쌍으로) AI에게 넘기는 대화 이력에서 뺀다
    history = llm.clean_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    run = st.session_state["running"] = {"partial": "", "sources": []}  # 중지됐을 때 복구용
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st.button("⏹ 중지", key=f"stop{len(st.session_state.messages)}", help="누르면 현재 답변을 멈춥니다")
        timing: dict[str, float] = {}

        t0 = time.time()
        with st.spinner("관련 규정을 찾는 중..."):
            plan = llm.plan(history, prompt)
            standalone, extras = plan.standalone, plan.extras
            if extras:
                timing["검색어 생성"] = time.time() - t0
            hits = llm.fit_hits(retriever.search(standalone, top_k=top_k, extra_queries=extras))
            timing.update(retriever.last_timing)

        queries = [standalone, *extras]
        sources = [{"ref": h.chunk.ref, "score": h.score, "via": h.via, "text": h.chunk.text} for h in hits]
        run["sources"] = sources

        t1 = time.time()
        status = st.empty()

        def tee(gen):
            """첫 글자 전까지 진행 상태를 보여주고, 받은 글자는 저장해 두어 중지해도 남긴다."""
            first, last_update = True, 0.0
            for tok in gen:
                if tok is llm.THINKING:
                    if time.time() - last_update > 1:
                        status.caption(f"🤔 AI가 생각하는 중… {time.time() - t1:.0f}초")
                        last_update = time.time()
                    continue
                if first:
                    timing["첫 글자까지"] = time.time() - t1
                    status.empty()
                    first = False
                run["partial"] += tok
                yield tok

        if not hits:
            answer = NOT_FOUND
            st.markdown(answer)
        else:
            status.caption("🧠 AI가 근거를 꼼꼼히 읽고 답변을 작성하는 중…" if deep else "✍️ AI가 답변을 작성하는 중…")
            try:
                # 새 주제 질문이면 이전 대화를 AI에게 보여주지 않는다 (이전 내용이 답변에 섞여 나오는 것을 막는다)
                answer_history = history[-2:] if plan.followup else []
                context_q = standalone if plan.followup else None
                answer = st.write_stream(tee(llm.stream_answer(prompt, hits, answer_history, deep=deep, context_question=context_q)))
                answer = answer if isinstance(answer, str) else ""
                if not answer.strip():
                    answer = "⚠ AI가 빈 답변을 보냈습니다. 터미널에서 `python check_llm.py` 로 AI 연결 상태를 확인해 주세요."
                    st.markdown(answer)
            except Exception as e:  # LLM 서버 문제여도 검색 결과는 보여준다
                answer = f"⚠ {llm.explain_error(e)}\n\n아래 근거 조문은 지금도 확인하실 수 있습니다."
                st.markdown(answer)
            status.empty()
            if len(llm.LAST["hits"]) < len(hits) or (deep and not llm.LAST["deep"]):
                used = len(llm.LAST["hits"])
                st.caption(f"ℹ️ 무료 사용 한도 때문에 근거 {used}건만 읽고 일반 모드로 답했습니다. (아래 근거 목록에는 검색된 전체가 표시됩니다)")
            # AI가 '확인 불가'라고 했지만 관련도가 매우 높은 조문이 있으면, 그 조문을 직접 확인하도록 안내
            top = hits[0]
            if "확인할 수 없습니다" in answer and top.via == "rerank" and top.score >= 0.7:
                note = f"\n\n💡 다만 질문과 관련도가 높은 조문이 있습니다: **{top.chunk.ref}** — 아래 근거 조문 [1]의 원문을 확인해 주세요."
                answer += note
                st.markdown(note)
        timing["답변 생성"] = time.time() - t1
        render_sources(sources, timing, queries)

    st.session_state.pop("running", None)
    chatlog.log(prompt, answer, sources, timing, queries)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources, "timing": timing, "queries": queries})
    st.rerun()  # 중지 버튼을 화면에서 치우기 위해 한 번 다시 그린다
