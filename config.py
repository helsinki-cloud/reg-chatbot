"""환경설정. 모든 값은 .env 또는 환경변수로 바꿀 수 있다."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DOCS_DIR = Path(os.getenv("DOCS_DIR", ROOT / "docs"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "index"))

APP_TITLE = os.getenv("APP_TITLE", "사내 규정 안내 챗봇")
ORG_NAME = os.getenv("ORG_NAME", "우리 기관")
EXAMPLE_QUESTIONS = [q.strip() for q in os.getenv("EXAMPLE_QUESTIONS", "").split("|") if q.strip()]

EMBED_BACKEND = os.getenv("EMBED_BACKEND", "local")  # local | api | none(임베딩 없이 키워드 검색만: 무료 클라우드용, 가벼움)
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")

RERANK_BACKEND = os.getenv("RERANK_BACKEND", "local")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
RERANK_BASE_URL = os.getenv("RERANK_BASE_URL", "")
RERANK_API_KEY = os.getenv("RERANK_API_KEY", "")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-8B")
_extra = os.getenv("LLM_EXTRA_BODY", "").strip()
LLM_EXTRA_BODY = json.loads(_extra) if _extra else None

CANDIDATES = int(os.getenv("CANDIDATES", "40"))
RERANK_POOL = int(os.getenv("RERANK_POOL", "15"))
TOP_K = int(os.getenv("TOP_K", "6"))
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0"))

# 속도 관련
RERANK_MAX_LEN = int(os.getenv("RERANK_MAX_LEN", "1024"))      # 리랭커가 읽는 최대 길이(토큰)
LLM_NO_THINK = os.getenv("LLM_NO_THINK", "1") == "1"           # Qwen3의 '생각' 단계를 끈다(/no_think)
CONDENSE_WITH_LLM = os.getenv("CONDENSE_WITH_LLM", "0") == "1"  # 후속질문 정리에 LLM 사용(느림). 0이면 간단 규칙으로 처리
# LLM 호출 방식: auto(주소에 :11434가 있으면 Ollama 전용 방식) | ollama | openai
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))  # Ollama가 한 번에 읽는 길이. 작으면 규정이 잘려서 들어감

# 띄어쓰기 복원/키워드 검색용 사전에 추가할 업무 용어 (쉼표로 구분)
KIWI_USER_WORDS = [w.strip() for w in os.getenv(
    "KIWI_USER_WORDS", "호봉,가봉,근속가봉,상여금,봉급,수당,연차휴가,경조사,퇴직급여금,임금피크제,복리후생"
).split(",") if w.strip()]

# 본문이 가리키는 별표·조문을 근거에 자동으로 덧붙이는 최대 조각 수 (많을수록 정확하지만 AI가 읽을 양이 늘어남)
REF_EXPAND = int(os.getenv("REF_EXPAND", "3"))

# ── 답변 품질 관련 ──
# 질문마다 AI가 보충 검색어를 만들어 여러 각도로 검색: auto(외부/빠른 AI일 때만 켬) | 1(항상) | 0(끔)
QUERY_EXPANSION = os.getenv("QUERY_EXPANSION", "auto")
DEEP_DEFAULT = os.getenv("DEEP_DEFAULT", "1") == "1"   # 정밀 모드(AI가 답하기 전에 더 생각함) 기본값
DEEP_EFFORT = os.getenv("DEEP_EFFORT", "low")           # low | medium | high (높을수록 정확하지만 느리고 토큰을 더 씀)
LLM_CONTEXT_CHARS = int(os.getenv("LLM_CONTEXT_CHARS", "3500"))  # AI에 보내는 근거 글자 수 상한 (무료 한도 보호)
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))         # 답변 최대 길이(정밀 모드는 1.5배). 요청 한도 계산에 이 값이 미리 포함됨

# ── 배포(사내 서버) 관련 ──
APP_PASSWORD = os.getenv("APP_PASSWORD", "")          # 값이 있으면 접속 암호를 물어봄 (비우면 암호 없음)
LOG_CHATS = os.getenv("LOG_CHATS", "0") == "1"        # 1이면 logs/chat_log.csv 에 대화 기록
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT / "logs"))

# ── 검색 모델 교체용 ──
# 질문/문서 앞에 붙이는 말. 비워 두면(설정 안 하면) 모델 이름을 보고 자동으로 정한다.
#   Qwen3-Embedding: 질문에만 지시문 / e5 계열: 질문 'query: ', 문서 'passage: ' / 그 외(bge-m3 등): 없음
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX")   # None = 자동
EMBED_DOC_PREFIX = os.getenv("EMBED_DOC_PREFIX")       # None = 자동
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1100"))  # 조각 최대 글자 수 (작은 모델은 한 번에 읽는 길이가 짧아 700 권장)

# ── 공개 배포(Streamlit Community Cloud 등) ──
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "0"))        # 하루 전체 질문 상한 (0=무제한). 무료 API 한도를 누가 다 써버리는 것을 막음
SESSION_LIMIT = int(os.getenv("SESSION_LIMIT", "0"))    # 접속 1회(브라우저 탭)당 질문 상한 (0=무제한)
APP_NOTICE = os.getenv("APP_NOTICE", "")                # 사이드바에 항상 보이는 안내문 (예: 비공식 참고용, 개인정보 입력 금지)
