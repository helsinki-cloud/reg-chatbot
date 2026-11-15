"""하이브리드 검색: 임베딩 + BM25 -> RRF 결합 -> 리랭커 -> 참조조문 보강."""
import json
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

import config
from chunker import Chunk
from ko_tokenizer import APPX_RE, ART_RE, norm_appendix, norm_article, tokenize


@dataclass
class Hit:
    chunk: Chunk
    score: float
    via: str  # "rerank" | "rrf" | "ref"(본문이 참조한 조문을 추가로 붙인 것)


def rrf(rank_lists: list[list[int]], k: int = 60, weights: list[float] | None = None) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: 여러 검색 결과의 '순위'만 보고 합친다 (점수 스케일이 달라도 안전). weights로 목록별 비중 조절."""
    s: dict[int, float] = defaultdict(float)
    for j, lst in enumerate(rank_lists):
        w = 1.0 if weights is None else weights[j]
        for r, idx in enumerate(lst):
            s[idx] += w / (k + r + 1)
    return sorted(s.items(), key=lambda x: -x[1])


class Retriever:
    def __init__(self, embedder=None, reranker="auto", index_dir=None):
        index_dir = index_dir or config.INDEX_DIR
        meta = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        # 임베딩 없이(키워드 검색만) 쓰는 경우: 설정이 none 이거나, 색인에 임베딩이 없다고 적혀 있거나, 벡터 파일이 없을 때
        no_vectors = config.EMBED_BACKEND == "none" or meta.get("embed_model") == "none" or not (index_dir / "embeddings.npy").exists()
        self.dense = embedder is not None or not no_vectors
        if self.dense and meta.get("embed_model") != config.EMBED_MODEL and embedder is None:
            raise RuntimeError(
                f"인덱스는 {meta.get('embed_model')} 로 만들어졌는데 현재 설정은 {config.EMBED_MODEL} 입니다. "
                "python ingest.py 를 다시 실행하세요."
            )
        self.chunks = [Chunk(**c) for c in meta["chunks"]]
        self.emb = np.load(index_dir / "embeddings.npy") if self.dense else None
        self.bm25 = BM25Okapi([tokenize(c.embed_text) for c in self.chunks])
        self.by_article: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, c in enumerate(self.chunks):
            if c.article:
                self.by_article[(c.doc, c.article)].append(i)

        if embedder is None and self.dense:
            from models import Embedder
            embedder = Embedder()
        self.embedder = embedder  # None이면 키워드(BM25) 검색만 사용
        if reranker == "auto":
            if config.RERANK_BACKEND == "none" or self.embedder is None and config.RERANK_BACKEND == "local":
                reranker = None  # 리랭커 없음(또는 가벼운 배포: 로컬 모델 라이브러리가 없으므로 쓰지 않음)
            else:
                from models import make_reranker
                reranker = make_reranker()
        self.reranker = reranker
        self.last_timing: dict[str, float] = {}

    # ── 단계별 순위 (eval.py에서도 사용) ──
    def _query_vectors(self, queries: list[str]):
        if self.embedder is None:
            return [None] * len(queries)
        if hasattr(self.embedder, "encode_queries"):
            return list(self.embedder.encode_queries(queries))  # 검색어 여러 개를 모델에 한 번만 통과시킨다
        return [self.embedder.encode_query(q) for q in queries]

    def rank_dense(self, query: str, n: int, vec=None) -> list[int]:
        if self.embedder is None:
            return []
        sims = self.emb @ (vec if vec is not None else self.embedder.encode_query(query))
        return [int(i) for i in np.argsort(-sims)[:n]]

    def rank_sparse(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores)[:n] if scores[i] > 0]

    def rank_multi(self, queries: list[str], n: int) -> list[tuple[int, float]]:
        """검색어가 하나면 임베딩+키워드를 RRF로 합친 순위. 여럿이면 '검색어마다 1등을 보장'하는 번갈아 뽑기.

        점수를 합산하면 원래 질문과 비슷한 조문이 계속 이겨서, 보충 검색어가 찾아낸 다른 항목(예: '2월 뭐 나와?' 에서
        '설날 명절휴가비')이 밀려난다. 그래서 원래 질문은 상위 2개, 보충 검색어는 각각 1개씩 번갈아 뽑는다.
        """
        per_query = []
        for q, vec in zip(queries, self._query_vectors(queries)):
            fused = rrf([self.rank_dense(q, config.CANDIDATES, vec), self.rank_sparse(q, config.CANDIDATES)])
            per_query.append(fused)
        if len(queries) == 1:
            return per_query[0][:n]
        orders = [[i for i, _ in lst] for lst in per_query]
        merged: list[int] = []
        seen: set[int] = set()
        pos = [0] * len(orders)
        while len(merged) < n and any(pos[j] < len(orders[j]) for j in range(len(orders))):
            for j, order in enumerate(orders):
                quota, taken = (2 if j == 0 else 1), 0
                while taken < quota and pos[j] < len(order):
                    idx = order[pos[j]]
                    pos[j] += 1
                    if idx not in seen:
                        seen.add(idx)
                        merged.append(idx)
                        taken += 1
        return [(i, 1.0 / (r + 1)) for r, i in enumerate(merged[:n])]

    def rank_hybrid(self, query: str, n: int) -> list[tuple[int, float]]:
        return self.rank_multi([query], n)

    # ── 최종 검색 ──
    def search(self, query: str, top_k: int | None = None, expand: bool = True, apply_min: bool = True,
               extra_queries: list[str] | None = None) -> list[Hit]:
        top_k = top_k or config.TOP_K
        t0 = time.time()
        fused = self.rank_multi([query, *(extra_queries or [])], max(config.RERANK_POOL, top_k))
        self.last_timing = {"검색": time.time() - t0}
        if self.reranker is not None and fused:
            t1 = time.time()
            scores = self.reranker.score(query, [self.chunks[i].embed_text for i, _ in fused])
            self.last_timing["리랭킹"] = time.time() - t1
            ranked = sorted(zip((i for i, _ in fused), scores), key=lambda x: -x[1])
            hits = [Hit(self.chunks[i], float(s), "rerank") for i, s in ranked[:top_k]]
            if apply_min and config.MIN_RERANK_SCORE > 0:
                hits = [h for h in hits if h.score >= config.MIN_RERANK_SCORE]
        else:
            hits = [Hit(self.chunks[i], s, "rrf") for i, s in fused[:top_k]]
        return self._expand_refs(hits) if expand else hits

    def _expand_refs(self, hits: list[Hit], max_extra: int | None = None) -> list[Hit]:
        """상위 조문 본문이 '별표 12' 나 '제N조'를 가리키면, 그 내용도 근거로 붙여준다.

        별표를 먼저 붙인다: 수당 목록·금액 같은 구체적인 내용이 표에 들어 있는 경우가 많기 때문.
        """
        max_extra = config.REF_EXPAND if max_extra is None else max_extra
        present = {h.chunk.id for h in hits}
        extra: list[Hit] = []

        def add(doc: str, label: str, own: str):
            if label == own:
                return
            for idx in self.by_article.get((doc, label), []):
                c = self.chunks[idx]
                if c.id not in present and len(extra) < max_extra:
                    present.add(c.id)
                    extra.append(Hit(c, 0.0, "ref"))

        top = [h.chunk for h in hits[:3]]
        scan = {c.id: f"{c.title}\n{c.text}" for c in top}  # 제목도 본다: "[별표 12] …(제2조 관련)" 의 '제2조'
        for c in top:  # 1순위: 별표
            for m in APPX_RE.finditer(scan[c.id]):
                add(c.doc, norm_appendix(m), c.article)
        for c in top:  # 2순위: 다른 조문
            for m in ART_RE.finditer(scan[c.id]):
                add(c.doc, norm_article(m), c.article)
        return hits + extra
