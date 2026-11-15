"""AI(LLM) 연결·속도 진단.  실행: python check_llm.py

챗봇 화면 없이 AI만 따로 불러서, 첫 글자까지 몇 초 걸리는지 / 생각 단계가 켜져 있는지 /
(Ollama라면) GPU를 쓰고 있는지를 보여준다. 결과를 그대로 복사해서 공유하면 원인을 찾을 수 있다.
"""
import json
import sys
import time

import requests

import config
import llm
from chunker import Chunk
from retriever import Hit


def measure(title, make_gen):
    print(f"\n=== {title} ===")
    t0 = time.time()
    first = None
    thinking = 0
    text = ""
    try:
        for tok in make_gen():
            if tok is llm.THINKING:
                thinking += 1
                if thinking == 1:
                    print(f"  (AI가 '생각' 단계를 진행 중... {time.time() - t0:.1f}초)")
                continue
            if first is None:
                first = time.time() - t0
                print(f"  첫 글자까지: {first:.1f}초")
            text += tok
    except Exception as e:
        print(f"  ❌ 실패: {type(e).__name__}: {str(e)[:300]}")
        if "does not exist" in str(e) or "not found" in str(e).lower() or "404" in str(e):
            print(f"\n  → LLM_MODEL 에 적은 '{config.LLM_MODEL}' 는 이 서버에 없는 이름입니다. 쓸 수 있는 이름은 아래와 같습니다.")
            list_models()
        return False
    total = time.time() - t0
    print(f"  전체: {total:.1f}초 | 답변 {len(text)}자 | 생각 신호 {thinking}회")
    print(f"  답변 앞부분: {text[:200]!r}")
    if not text.strip():
        print("  ⚠ 답변이 비어 있습니다.")
    return True


def list_models():
    """이 주소(LLM_BASE_URL)에서 실제로 쓸 수 있는 모델 이름 목록. (Groq, Ollama 등 OpenAI 호환 서버)"""
    url = config.LLM_BASE_URL.rstrip("/") + "/models"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {config.LLM_API_KEY}"}, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"❌ 목록을 가져오지 못했습니다 ({url}): {type(e).__name__}: {str(e)[:200]}")
        print("   LLM_BASE_URL 과 LLM_API_KEY 가 맞는지 확인하세요.")
        return
    ids = sorted(m.get("id", "").removeprefix("models/") for m in r.json().get("data", []))  # Gemini는 'models/' 접두어를 붙여 줌
    print(f"{url}\n사용 가능한 모델 {len(ids)}개 (LLM_MODEL 에는 아래 이름을 그대로 적으세요):")
    for i in ids:
        print("  ", i)


def main():
    if "--models" in sys.argv:
        return list_models()
    print(f"LLM 주소: {config.LLM_BASE_URL}")
    print(f"모델: {config.LLM_MODEL}")
    native = llm.use_ollama_native()
    print(f"호출 방식: {'Ollama 전용(생각 끔, num_ctx=' + str(config.OLLAMA_NUM_CTX) + ')' if native else 'OpenAI 호환'}")

    if native:
        try:
            ps = requests.get(f"{llm._ollama_root()}/api/ps", timeout=5).json()
            models = ps.get("models", [])
            if not models:
                print("메모리에 올라간 모델: 없음 (첫 호출 때 올리느라 더 느릴 수 있음)")
            for m in models:
                size, vram = m.get("size", 0), m.get("size_vram", 0)
                where = "GPU" if vram and vram >= size * 0.9 else ("CPU+GPU 혼합" if vram else "CPU만")
                print(f"메모리에 올라간 모델: {m.get('name')} | {size / 1e9:.1f}GB | {where}")
        except Exception as e:
            print(f"❌ Ollama에 연결할 수 없습니다: {e}\n   Ollama가 켜져 있는지(작업표시줄 라마 아이콘) 확인하세요.")
            return

    # 1) 짧은 질문
    if measure("1) 짧은 인사", lambda: llm.strip_think_stream(_short())) is False:
        return

    # 2) 실제 챗봇과 같은 크기의 질문 (색인된 조문 6개를 근거로 넣음)
    try:
        meta = json.loads((config.INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in meta["chunks"]][:6]
        hits = [Hit(c, 1.0, "rerank") for c in chunks]
        ctx_chars = sum(len(c.text) for c in chunks)
        measure(f"2) 실제 크기 질문 (근거 {len(hits)}개, 약 {ctx_chars:,}자)",
                lambda: llm.stream_answer("이 규정들의 핵심 내용을 세 문장으로 요약해 주세요.", hits, []))
    except FileNotFoundError:
        print("\n(인덱스가 없어 2번 테스트는 건너뜀. python ingest.py 먼저 실행)")

    print("\n[해석]")
    print("- 1번부터 실패: 주소/모델명 오류 또는 Ollama 꺼짐")
    print("- 1번은 빠른데 2번 '첫 글자까지'가 수십 초 이상: PC가 긴 글을 읽는 속도가 느린 것 (CPU 실행일 가능성)")
    print("- '생각 신호'가 0보다 크면: 생각 단계가 아직 켜져 있음")


def _short():
    if llm.use_ollama_native():
        return llm._ollama_stream([{"role": "user", "content": "'안녕하세요'라고만 답하세요."}], 50, 0.0)
    resp = llm.client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": llm._no_think("'안녕하세요'라고만 답하세요.")}],
        max_tokens=50, stream=True, extra_body=config.LLM_EXTRA_BODY,
    )
    return llm._openai_stream(resp)


if __name__ == "__main__":
    main()
