"""BM25용 한국어 토크나이저 (Kiwi 형태소 분석).

'제16조' 같은 조문 번호는 형태소 분석에 맡기지 않고 정규식으로 따로 토큰화해서,
"제16조가 뭐야?" 같은 조문 번호 직접 검색이 정확히 걸리게 한다.
"""
import re
from functools import lru_cache

_KEEP = {"NNG", "NNP", "NR", "SL", "SN", "VV", "VA", "XR"}  # 명사·수사·외국어·숫자·동사/형용사 어근
ART_RE = re.compile(r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?")
APPX_RE = re.compile(r"별\s*표\s*(\d+)(?:\s*의\s*(\d+))?")


def norm_article(m: "re.Match") -> str:
    return f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")


def norm_appendix(m: "re.Match") -> str:
    """'별표12', '별표 12', '[별표 12의2]' -> '별표 12' / '별표 12의2' (청크의 article 값과 같은 형식)"""
    return f"별표 {m.group(1)}" + (f"의{m.group(2)}" if m.group(2) else "")


@lru_cache(maxsize=1)
def _kiwi():
    """검색(BM25)용. 사용자 단어를 넣지 않는다: '연차휴가'가 한 덩어리가 되면 '휴가'로 검색할 때 안 걸리기 때문."""
    from kiwipiepy import Kiwi

    return Kiwi()


@lru_cache(maxsize=1)
def _kiwi_spacer():
    """띄어쓰기 복원용. 업무 용어(호봉, 가봉 등)가 쪼개지지 않게 사전에 추가한다."""
    from kiwipiepy import Kiwi

    import config

    k = Kiwi()
    for w in config.KIWI_USER_WORDS:
        k.add_user_word(w, "NNG")
    return k


def respace(text: str) -> str:
    """띄어쓰기가 사라진 한국어(PDF 추출 결과 등)의 띄어쓰기를 줄 단위로 복원한다."""
    return "\n".join(_kiwi_spacer().space(ln, reset_whitespace=True) if ln.strip() else ln for ln in text.split("\n"))


def is_spaceless(text: str) -> bool:
    hangul = sum(1 for ch in text if "가" <= ch <= "힣")
    return hangul >= 200 and text.count(" ") / hangul < 0.15


def tokenize(text: str) -> list[str]:
    toks = [norm_article(m) for m in ART_RE.finditer(text)]
    toks += [norm_appendix(m).replace(" ", "") for m in APPX_RE.finditer(text)]  # '별표12' (표 직접 검색용)
    toks += [t.form.lower() for t in _kiwi().tokenize(text) if t.tag in _KEEP]
    return toks
