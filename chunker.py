"""규정 문서 청킹.

핵심 아이디어: 글자 수로 자르지 않고 '제N조' 단위로 자른다.
각 조각에 규정명·장·조문번호·조문제목을 붙여서, 검색할 때 "인사규정 제16조(경력인정)" 같은
정보까지 함께 임베딩/키워드 검색 대상이 되게 한다.
"""
import re
from dataclasses import dataclass

# 줄 맨 앞의 '제N조' (뒤에 (제목) 또는 공백+본문이 오는 경우만).
# 문장 중간의 "제3조에 따라" 같은 참조는 줄 맨 앞이 아니거나 바로 붙어 있어서 걸리지 않는다.
ARTICLE_RE = re.compile(
    r"^[ \t]*제[ \t]*(\d+)[ \t]*조(?:의[ \t]*(\d+))?(?=[ \t]*[\(（]|[ \t]+\S)"
    r"[ \t]*(?:[\(（]([^)）\n]{0,40})[)）])?",
    re.M,
)
CHAPTER_RE = re.compile(r"^[ \t]*(제[ \t]*\d+[ \t]*[장절편][^\n]*)\n?", re.M)
CLAUSE_RE = re.compile(r"^\s*(?:[①-⑳]|\(\d+\)|\d+\.)")
HEADING_RE = re.compile(
    r"^\s*(?:#{1,4}\s+|\d{1,2}[\.\)]\s+|[가-힣][\.\)]\s+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ][\.\s]|[■▶◆●]\s*|\[[^\]]{2,30}\])"
)
# 별표: "[별표 12] 제목" 처럼 대괄호가 있으면 확실한 시작. 대괄호 없이 "별표 12 제목" 만 있는 줄은
# 짧고 문장(…다.)이 아닐 때만 인정한다. ("별표 12의 기준에 따라…" 같은 본문 줄바꿈은 제외)
APPX_BRACKET_RE = re.compile(
    r"^[ \t]*[\[\(<【〈][ \t]*별[ \t]*표[ \t]*(\d+)(?:[ \t]*의[ \t]*(\d+))?[ \t]*[\]\)>】〉][ \t]*([^\n]*)$", re.M
)
APPX_BARE_RE = re.compile(r"^[ \t]*별[ \t]*표[ \t]*(\d+)(?:의(\d+))?(?=[ \t]|$)[ \t]*([^\n]*)$", re.M)
_PARTICLE_START = re.compile(r"^(?:의|에서|에는|에도|에|을|를|은|는|이|가|과|와|로|으로)(?:\s|$)")
# 부칙: "부칙", "부칙(2017.1.1)", "부칙 <제15호, 2017.1.1>" 한 줄
SUPP_RE = re.compile(r"^[ \t]*부[ \t]*칙[ \t]*(?:[\(<（]([^)>）\n]{0,40})[)>）])?[ \t]*$", re.M)
NOISE_RE = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$|^\s*[Pp]age\s*\d+.*$")

MIN_BODY = 20       # 이보다 짧은 조문(목차 줄 등)은 버린다
MIN_SECTION = 200   # 일반 문서: 이보다 짧으면 앞 섹션에 합친다


@dataclass
class Chunk:
    id: str
    doc: str        # 규정명 (파일명)
    article: str    # "제16조", "제16조의2", 일반 문서는 ""
    title: str      # 조문 제목 / 섹션 제목
    chapter: str    # "제2장 인사"
    part: str       # 긴 조문을 나눴을 때 "2/3", 아니면 ""
    text: str
    source: str     # 원본 파일 경로

    @property
    def key(self) -> str:
        """평가용 정답 키. 예: '인사규정 제16조'"""
        return f"{self.doc} {self.article or self.title}".strip()

    @property
    def ref(self) -> str:
        r = f"{self.doc} {self.article}".strip()
        if self.title:
            r += f"({self.title})"
        if self.part:
            r += f" [{self.part}]"
        return r

    @property
    def embed_text(self) -> str:
        head = f"{self.ref} | {self.chapter}" if self.chapter else self.ref
        return f"[{head}]\n{self.text}"


# 조각난 글을 이어 붙인 뒤, 다시 줄을 나눌 위치 (조문·장·항·부칙 시작)
_BREAK_BEFORE = re.compile(
    r"\s*(?=제\s*\d+\s*조(?:\s*의\s*\d+)?\s*[\(（])"   # 제16조( / 제16조의2(
    r"|\s*(?=제\s*\d+\s*[장절편]\s)"                       # 제2장 
    r"|\s*(?=[①-⑳])"                                          # ① ② …
    r"|\s*(?=부\s*칙(?:\s|[\(<]|$))"                         # 부칙
)


def _is_fragmented(lines: list[str]) -> bool:
    """PDF 추출이 단어/글자 단위로 줄을 끊어놓은 상태인지 (줄 길이 중앙값이 아주 짧으면)."""
    ls = sorted(len(ln.strip()) for ln in lines if ln.strip())
    return len(ls) >= 20 and ls[len(ls) // 2] <= 6


def _defragment(lines: list[str]) -> str:
    flat = re.sub(r"\s+", " ", " ".join(ln.strip() for ln in lines if ln.strip()))
    flat = re.sub(r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?", lambda m: f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else ""), flat)
    flat = re.sub(r"제\s*(\d+)\s*([장절편])", r"제\1\2", flat)
    return re.sub(r"\n{2,}", "\n", _BREAK_BEFORE.sub("\n", flat)).strip()


_MD_PREFIX = re.compile(r"^\s*(?:#{1,6}\s+|>\s*|[-*+]\s+(?=\**\s*제\s*\d))")


def _strip_md(line: str) -> str:
    """마크다운 기호 제거: '## 제3조(…)', '**제3조(…)**', '- 제3조 …' 도 조문으로 인식되게."""
    line = _MD_PREFIX.sub("", line)
    return line.replace("**", "").replace("__", "")


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    lines = [_strip_md(ln.rstrip()) for ln in text.split("\n")]
    if _is_fragmented(lines):  # 조각난 상태에선 숫자만 있는 줄도 내용('제/3/조')이라 먼저 이어 붙인다
        flat = _defragment(lines)
        return re.sub(r"(?:^|\s)[-–—]\s?\d{1,4}\s?[-–—](?=\s|$)", " ", flat)  # '- 12 -' 페이지 번호 제거
    lines = [ln for ln in lines if not NOISE_RE.match(ln)]  # 페이지 번호 줄 제거
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _pack(lines: list[str], max_chars: int) -> list[str]:
    """줄 단위로 max_chars 안에서 묶는다. 가능하면 항(①②…) 경계에서 끊는다."""
    pieces, cur, size = [], [], 0
    for ln in lines:
        too_big = size + len(ln) + 1 > max_chars
        clause_break = size > max_chars * 0.6 and CLAUSE_RE.match(ln)
        if cur and (too_big or clause_break):
            pieces.append("\n".join(cur))
            cur, size = [], 0
        cur.append(ln)
        size += len(ln) + 1
    if cur:
        pieces.append("\n".join(cur))
    return pieces


_TABLE_SEP = re.compile(r"^\s*\|?[\s:\-|]*-[\s:\-|]*\|[\s:\-|]*$")


def _table_header(lines: list[str]):
    """마크다운 표의 머리 두 줄(제목행 + |---|)을 찾는다."""
    for i in range(len(lines) - 1):
        if lines[i].lstrip().startswith("|") and _TABLE_SEP.match(lines[i + 1]):
            return [lines[i], lines[i + 1]]
    return None


def _emit(out, doc, article, title, chapter, body, source, max_chars):
    lines = body.split("\n")
    pieces = _pack(lines, max_chars) if len(body) > max_chars else [body]
    header = _table_header(lines) if len(pieces) > 1 else None
    n = len(pieces)
    for i, p in enumerate(pieces, 1):
        # 표가 쪼개져 둘째 조각부터 머리 행이 없으면, 무슨 칸인지 알 수 있게 머리 행을 붙여준다
        if header and i > 1 and p.lstrip().startswith("|") and header[0] not in p:
            p = "\n".join(header) + "\n" + p
        out.append(Chunk("", doc, article, title, chapter, f"{i}/{n}" if n > 1 else "", p.strip(), source))


def _appendix_label(m) -> str:
    return f"별표 {m.group(1)}" + (f"의{m.group(2)}" if m.group(2) else "")


def _boundaries(text: str, arts) -> list[tuple]:
    """조문 / 별표 / 부칙이 시작되는 위치를 모두 모아 순서대로 정렬. (시작, 머리글 끝, 종류, 번호, 제목)"""
    bs = []
    for m in arts:
        art = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
        bs.append((m.start(), m.end(), "article", art, (m.group(3) or "").strip()))
    for rx, bracketed in ((APPX_BRACKET_RE, True), (APPX_BARE_RE, False)):
        for m in rx.finditer(text):
            rest = (m.group(3) or "").strip()
            if _PARTICLE_START.match(rest):                       # "[별표 12]의 기준에…" 는 본문 속 인용
                continue
            if not bracketed and (len(m.group(0).strip()) > 50 or rest.endswith(("다", "."))):
                continue                                          # 문장이 줄바꿈된 것
            if bracketed and rest.endswith(("다.", "한다", "된다")):
                continue
            bs.append((m.start(), m.end(), "appendix", _appendix_label(m), rest))
    for m in SUPP_RE.finditer(text):
        bs.append((m.start(), m.end(), "supp", "부칙", (m.group(1) or "").strip()))
    return sorted(bs, key=lambda b: b[0])


def _meaningful(body: str) -> int:
    return len(re.sub(r"[\s\.·…ㆍ\-\d|:]", "", body))


def _by_article(doc, text, source, max_chars, arts):
    chapters = [(m.start(), m.group(1).strip()) for m in CHAPTER_RE.finditer(text)]

    def chapter_at(pos):
        cur = ""
        for p, t in chapters:
            if p <= pos:
                cur = t
            else:
                break
        return cur

    bs = _boundaries(text, arts)
    out = []
    pre = CHAPTER_RE.sub("", text[: bs[0][0]]).strip()
    if len(pre) >= 40:  # 규정 제목/머리말
        _emit(out, doc, "", "머리말", "", pre, source, max_chars)

    supp_label = ""  # 부칙 안에서는 조문 번호가 다시 1부터 시작하므로 '부칙 제2조'처럼 본칙과 구분한다
    for i, (start, hend, kind, article, title) in enumerate(bs):
        end = bs[i + 1][0] if i + 1 < len(bs) else len(text)
        body = CHAPTER_RE.sub("", text[hend:end]).strip()  # 다음 장 제목이 딸려오는 것 제거
        if kind == "supp":
            supp_label = f"부칙({title})" if title else "부칙"
            if _meaningful(body) >= MIN_BODY:  # 조문 없이 바로 본문이 오는 부칙
                _emit(out, doc, "부칙", title, "", body, source, max_chars)
            continue
        if _meaningful(body) < MIN_BODY // 2:  # 목차 줄('…… 12') 등 내용 없는 것
            continue
        chapter = chapter_at(start)
        if kind == "appendix":
            supp_label = ""
        elif supp_label:
            article, chapter = f"부칙 {article}", supp_label
        _emit(out, doc, article, title, chapter, body, source, max_chars)
    return out


def _by_section(doc, text, source, max_chars):
    out, n_sec = [], 0
    cur_title, cur_lines, size = "", [], 0

    def flush():
        nonlocal n_sec, cur_title, cur_lines, size
        body = "\n".join(cur_lines).strip()
        if body:
            n_sec += 1
            _emit(out, doc, "", cur_title or f"섹션{n_sec}", "", body, source, max_chars)
        cur_title, cur_lines, size = "", [], 0

    for ln in text.split("\n"):
        is_head = bool(HEADING_RE.match(ln)) and 0 < len(ln.strip()) <= 50
        if is_head and size >= MIN_SECTION:
            flush()
        if is_head and not cur_title:
            cur_title = ln.strip().lstrip("#").strip()
        cur_lines.append(ln)
        size += len(ln) + 1
    flush()
    return out


def chunk_document(doc: str, text: str, source: str = "", max_chars: int = 1100) -> list[Chunk]:
    text = clean(text)
    arts = list(ARTICLE_RE.finditer(text))
    chunks = _by_article(doc, text, source, max_chars, arts) if len(arts) >= 2 else _by_section(doc, text, source, max_chars)
    if chunks:
        from ko_tokenizer import is_spaceless, respace

        if is_spaceless(text):  # 조문 머리('제24조의2(…)')는 건드리지 않고 본문만 띄어쓰기 복원
            for c in chunks:
                c.text = respace(c.text)
    for i, c in enumerate(chunks):
        c.id = f"{doc}#{i}"
    return chunks
