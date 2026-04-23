import os
import re
import requests

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.callbacks.base import BaseCallbackHandler

# ── API 키 설정 (로컬: .env / Streamlit Cloud: st.secrets) ───────────────────
load_dotenv()
try:
    for k in ("OPENAI_API_KEY", "TAVILY_API_KEY"):
        if k in st.secrets:
            os.environ[k] = st.secrets[k]
except Exception:
    pass

# ── 경로 ─────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_BASE, "..", ".."))
DATA_DIR   = os.path.join(_ROOT, "artifacts", "api-server", "data")
SLIDES_DIR = os.path.join(_ROOT, "artifacts", "api-server", "public", "gmap-slides")
COVER_PATH = os.path.join(_ROOT, "artifacts", "pdf-chatbot", "public", "book-cover.png")
BOOK_URL   = "https://ebook-product.kyobobook.co.kr/dig/epd/ebook/E000012350958"

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="똑똑한 유럽여행 도우미",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif !important;
}

/* ── 배경 ── */
.stApp {
    background: #f8fafc !important;
}
/* 헤더는 투명하게 유지 (사이드바 토글 버튼 살림) */
header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
}
/* 헤더 안 불필요한 요소만 숨김 */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ── 메인 컨테이너 여백 ── */
.main .block-container {
    padding-top: 1rem !important;
    padding-bottom: 6rem !important;
    max-width: 780px !important;
}

/* ── 사이드바 ── */
section[data-testid="stSidebar"] {
    background: white !important;
    border-right: 1px solid #e2e8f0 !important;
}
/* 사이드바 접기 버튼 숨김 */
[data-testid="stSidebarCollapseButton"] { display: none !important; }


/* ── 구분선 ── */
hr { border-color: #e2e8f0 !important; margin: 20px 0 !important; }

/* ── 추천 질문 버튼: 640px 이하에서 1열로 전환 → 700px 이하로 확대 ── */
@media (max-width: 800px) {
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width: 100% !important;
        flex: none !important;
        min-width: 100% !important;
    }
}

/* ── 추천 질문 버튼 ── */
.stButton > button {
    border-radius: 12px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: all 0.18s ease !important;
    border: 1.5px solid #e2e8f0 !important;
    background: white !important;
    color: #334155 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
    padding: 12px 16px !important;
    white-space: normal !important;
    text-align: left !important;
    line-height: 1.5 !important;
    height: auto !important;
    min-height: 56px !important;
}
.stButton > button:hover {
    border-color: #93c5fd !important;
    background: #f0f7ff !important;
    color: #1d4ed8 !important;
    box-shadow: 0 4px 12px rgba(59,130,246,0.12) !important;
    transform: translateY(-1px) !important;
}

/* ── 채팅 입력창 ── */
.stChatInput textarea {
    border: none !important;
    font-size: 15px !important;
    background: transparent !important;
    box-shadow: none !important;
    color: #1e293b !important;
    padding: 12px 14px !important;
}
.stChatInput textarea::placeholder { color: #94a3b8 !important; }

div[data-testid="stChatInput"] button {
    background: #2563eb !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: none !important;
    color: white !important;
    margin: 4px !important;
}
div[data-testid="stChatInput"] button:hover {
    background: #1d4ed8 !important;
}

/* ── 채팅 버블 ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 4px 0 !important;
    gap: 10px !important;
}

/* 사용자 버블 */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {
    background: #2563eb !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 12px 18px !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.2) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] li {
    color: white !important;
    font-size: 15px !important;
    line-height: 1.65 !important;
    margin-bottom: 0 !important;
}

/* AI 버블 */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] {
    background: white !important;
    border-radius: 4px 18px 18px 18px !important;
    border: 1.5px solid #e2e8f0 !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] li {
    color: #1e293b !important;
    font-size: 15px !important;
    line-height: 1.75 !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] a {
    color: #2563eb !important;
}

/* ── 칩 스타일 ── */
.chip-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin: 8px 0 4px;
}
.chip {
    display: inline-block;
    padding: 6px 14px;
    background: #eff6ff;
    border: 1.5px solid #bfdbfe;
    border-radius: 999px;
    color: #1d4ed8;
    font-size: 13px;
    font-weight: 500;
    text-decoration: none !important;
    transition: all 0.15s ease;
    cursor: pointer;
    white-space: nowrap;
}
.chip:hover {
    background: #dbeafe;
    border-color: #60a5fa;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(59,130,246,0.15);
}

/* ── 하단 고정 칩 바 ── */
.chip-bar-fixed {
    position: fixed;
    bottom: 72px;
    left: 50%;
    transform: translateX(-50%);
    width: calc(100% - 48px);
    max-width: 780px;
    background: white;
    border-top: 1.5px solid #e2e8f0;
    padding: 10px 16px 8px;
    z-index: 100;
}
.chip-bar-fixed .chip-wrap { margin: 0; justify-content: flex-start; gap: 7px; }

/* ── 입력창 고정 배경 ── */
[data-testid="stBottom"] {
    background: white !important;
    border-top: none !important;
    padding-top: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# ── 콘텐츠 ───────────────────────────────────────────────────────────────────
SUGGESTIONS = [
    "동유럽에서 숙소간 이동하는 택시가 있어?",
    "잘츠부르크에서 할슈타트까지 한번에?",
    "유럽여행 필수준비물이 뭐야?",
    "빈 미술사 박물관, 인생 사진 명소?",
    "유럽 크리스마스 마켓 필수 먹거리는?",
    "현지 투어 예약 전, 주의사항!",
]

TRAVEL_TIPS = [
    ("🗺️ 길찾기",   "구글맵으로 길찾기 하는 법?"),
    ("⭐ 리뷰 보기", "구글맵 리뷰 보는 법?"),
    ("👁️ 라이브뷰",  "라이브뷰 기능이 뭐야?"),
    ("📍 위치공유",  "현재 위치 공유하는 법?"),
    ("🚂 기차예약",  "유럽기차 예약방법은?"),
    ("🗣️ 번역기",   "번역기로 실시간 대화하는 법?"),
]


# ── PDF → FAISS ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📚 여행 자료를 불러오는 중...")
def load_vector_stores() -> dict:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    stores = {}
    for name, fname in [("travel", "travel.pdf"), ("googlemap", "googlemap.pdf")]:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        text = "".join(p.extract_text() or "" for p in PdfReader(path).pages)
        docs = CharacterTextSplitter(separator="\n\n", chunk_size=800, chunk_overlap=100).create_documents([text])
        if docs:
            stores[name] = FAISS.from_documents(docs, embeddings)
    return stores


# ── 질문 분류 ─────────────────────────────────────────────────────────────────
_GMAP_RE = re.compile(
    r"구글\s*맵|구글\s*지도|google\s*map|길찾기|내비|나침반|스트리트뷰|위성\s*지도|"
    r"장소\s*검색|즐겨찾기\s*저장|오프라인\s*지도|라이브\s*뷰|live\s*view|위치\s*공유|"
    r"리뷰|평점|후기|기차\s*예약|유럽\s*기차|열차\s*예약|train|rail|버스\s*예약|"
    r"비행기\s*예약|번역기|실시간\s*대화|구글\s*번역|준비물|짐\s*싸기|챙길\s*것|packing",
    re.IGNORECASE,
)
_TOUR_RE = re.compile(r"투어|tour|주의사항|현지.*예약|예약.*주의|당일치기|야경.*투어|투어.*예약", re.IGNORECASE)

def is_gmap(q): return bool(_GMAP_RE.search(q))
def is_tour(q): return bool(_TOUR_RE.search(q))


# ── 슬라이드 이미지 ───────────────────────────────────────────────────────────
TOPIC_IMAGES = [
    {"pat": r"설치|다운로드|플레이\s*스토어|install",                        "pages": [3,4,5,6]},
    {"pat": r"화면\s*구성|인터페이스|검색창|나침반|현재\s*위치|기본\s*화면", "pages": [15,16,17,18]},
    {"pat": r"지도\s*유형|위성|지형|3d|스트리트뷰",                          "pages": [19,20,21,22,24,26]},
    {"pat": r"맛집|음식점|레스토랑|카페|브런치",                              "pages": [31,32,33,34,35]},
    {"pat": r"리뷰|평점|후기|별점",                                           "pages": [43,44,45]},
    {"pat": r"길찾기|경로|내비|navigation|도보|대중교통|자전거",              "pages": [48,51,52,53,55,56]},
    {"pat": r"라이브\s*뷰|live\s*view|증강현실|ar",                           "pages": [83,84,85,86,87,88,89]},
    {"pat": r"즐겨찾기|저장|목록|나만의\s*여행",                              "pages": [93,94,95,96]},
    {"pat": r"위치\s*공유|현재\s*위치\s*보내|동선|공유.*위치",               "files": [f"mylocation-{i}.png" for i in range(1,6)]},
    {"pat": r"기차\s*예약|유럽\s*기차|열차\s*예약|train|rail|버스\s*예약|비행기\s*예약", "files": [f"train-{i}.png" for i in range(1,9)]},
    {"pat": r"번역기|실시간\s*대화|구글\s*번역|google\s*translate",           "files": [f"translate-{i}.png" for i in range(1,7)]},
    {"pat": r"준비물|짐\s*싸기|챙길\s*것|packing",                           "files": [f"packing-{i}.png" for i in range(1,6)]},
]

def get_images(q: str, max_imgs: int = 6) -> list:
    files, pages = [], []
    for t in TOPIC_IMAGES:
        if re.search(t["pat"], q, re.IGNORECASE):
            files.extend(t.get("files", []))
            pages.extend(t.get("pages", []))
    if files:
        return [p for f in files if os.path.exists(p := os.path.join(SLIDES_DIR, f))][:max_imgs]
    if pages:
        return [p for n in pages if os.path.exists(p := os.path.join(SLIDES_DIR, f"page-{n:03d}.png"))][:max_imgs]
    return [p for n in [1,7,15] if os.path.exists(p := os.path.join(SLIDES_DIR, f"page-{n:03d}.png"))]


# ── Tavily 웹 검색 ────────────────────────────────────────────────────────────
def tavily_search(query: str) -> list:
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return []
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "search_depth": "basic", "max_results": 5},
            timeout=10,
        )
        return [{"title": x.get("title",""), "url": x.get("url",""), "content": x.get("content","")[:600]}
                for x in r.json().get("results", [])] if r.ok else []
    except Exception:
        return []


# ── 시스템 프롬프트 ───────────────────────────────────────────────────────────
def build_prompt(pdf_ctx: str, web_ctx: str, gmap: bool, tour: bool) -> str:
    p = """당신은 친절하고 유용한 동유럽 여행 및 구글맵 사용법 전문 챗봇입니다.
질문에 자세하고 실용적으로 답해주세요. 질문과 같은 언어로 답변하세요 (한국어 질문 → 한국어 답변).

## 관련 사이트 안내
- 투어·액티비티: GetYourGuide (https://www.getyourguide.com)
- 체코·슬로바키아 셔틀: CK Shuttle (https://www.ckshuttle.cz)
- 동유럽 버스: FlixBus (https://www.flixbus.com), RegioJet (https://www.regiojet.com)
- 기차 통합 검색: Omio (https://www.omio.com), Trainline (https://www.thetrainline.com)
- 숙소: Booking.com (https://www.booking.com), Hostelworld (https://www.hostelworld.com)
- 빈·잘츠부르크 교통: ÖBB (https://www.oebb.at)
- 체코 철도: České dráhy (https://www.cd.cz) / 헝가리 철도: MÁV (https://www.mavcsoport.hu)

## 도시 간 이동 질문 시 반드시 안내
- CK Shuttle (https://www.ckshuttle.cz): 프라하·빈·부다페스트·잘츠부르크 연결
- Bean Shuttle (https://www.beanshuttle.com): 숙소 픽업·드롭오프 포함 프라이빗 셔틀

## 잘츠부르크 → 할슈타트
직행버스 없음. 기차+페리 조합 또는 GetYourGuide 당일치기 투어: https://www.getyourguide.com

## 빈 미술사 박물관 사진 명소
알베르티나 미술관 2층 야외 테라스 — 영화 《비포 선셋》 촬영지. 무료 입장.

## 현지 투어 주의사항
투어는 도착 당일이나 다음날 최대한 빨리! 날씨 변수에 대비해 재도전 기회 확보.
작가 실제 경험: 부다페스트 야경 투어 당일 안개 → 다음날 어부의 요새 재도전 성공."""

    if gmap:
        p += "\n\n## 구글맵 질문\n추천 영상: https://youtu.be/oZ57SmPTh9s\n단계별로 쉽게, 해외 여행자에게 유용한 기능 강조."

    if tour:
        p += """

## ⚠️ 현지 투어 예약 — 아래 내용을 반드시 포함하세요
핵심 팁: 투어는 일정 초반에! 날씨 실패해도 재도전 가능.

작가 경험 투어 사례:
1. 리스본 → 신트라 당일치기 (동화 같은 궁전 마을, 차로 40분)
2. 런던 → 옥스퍼드 당일치기 (해리포터 촬영지, 버스 1시간 30분)
3. 런던 야경 투어 (템스강·타워브리지·빅벤)
4. 로마 → 폼페이·포지타노 남부 투어
5. 에딘버러 → 하이랜드 투어 (네스 호수·글렌코, 시니어 추천)
6. 타이페이 → 지우펀 투어 (《센과 치히로》 배경, 저녁 야경 최고)

예약: GetYourGuide (https://www.getyourguide.com), 클룩(Klook) — Free cancellation 필수 확인"""

    if pdf_ctx:
        p += f"\n\n## 참고 자료\n{pdf_ctx}"
    if web_ctx:
        p += f"\n\n## 최신 인터넷 정보\n{web_ctx}"
    if pdf_ctx or web_ctx:
        p += "\n\n위 정보를 종합하여 출처(여행기 / 구글맵 가이드 / 웹 정보)를 구분해서 답변하세요."
    return p


# ── RAG 검색 ──────────────────────────────────────────────────────────────────
def search_pdf(q: str, stores: dict, k: int = 4) -> str:
    results = []
    for label, name in [("여행기", "travel"), ("구글맵 가이드", "googlemap")]:
        store = stores.get(name)
        if store:
            for doc in store.similarity_search(q, k=k):
                results.append(f"[{label}] {doc.page_content}")
    return "\n\n".join(results[:6])


# ── 스트리밍 핸들러 ───────────────────────────────────────────────────────────
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **_):
        self.text += token
        self.container.markdown(self.text + "▌")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def tip_chips_html() -> str:
    items = "".join(
        f'<a href="?tip={label}" class="chip">{label}</a>'
        for label, _ in TRAVEL_TIPS
    )
    return f'<div class="chip-wrap">{items}</div>'


def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending" not in st.session_state:
        st.session_state.pending = None

    # 사이드바 토글 버튼 — iframe에서 부모 DOM에 직접 버튼 삽입
    import streamlit.components.v1 as components
    components.html("""
<script>
(function() {
    var pd = window.parent.document;
    if (pd.getElementById('sidebar-toggle-btn')) return;

    var style = pd.createElement('style');
    style.textContent = [
        '#sidebar-toggle-btn {',
        '  position:fixed; top:12px; left:12px; z-index:2147483647;',
        '  background:white; border:1.5px solid #e2e8f0; border-radius:8px;',
        '  padding:6px 12px; font-size:18px; cursor:pointer;',
        '  box-shadow:0 2px 8px rgba(0,0,0,0.12); line-height:1;',
        '  font-family:sans-serif;',
        '}',
        '#sidebar-toggle-btn:hover { background:#f0f7ff; border-color:#93c5fd; }',
    ].join('');
    pd.head.appendChild(style);

    var btn = pd.createElement('button');
    btn.id = 'sidebar-toggle-btn';
    btn.textContent = '☰';
    btn.addEventListener('click', function() {
        var selectors = [
            '[data-testid="collapsedControl"] button',
            '[data-testid="stSidebarCollapseButton"] button',
            'section[data-testid="stSidebar"] button',
            'header button',
        ];
        for (var i = 0; i < selectors.length; i++) {
            var t = pd.querySelector(selectors[i]);
            if (t) { t.click(); return; }
        }
    });
    pd.body.appendChild(btn);
})();
</script>
""", height=0)

    # 칩 클릭 처리 (query_params)
    if "tip" in st.query_params:
        clicked = st.query_params["tip"]
        for label, question in TRAVEL_TIPS:
            if label == clicked:
                st.session_state.pending = question
                break
        st.query_params.clear()
        st.rerun()

    stores = load_vector_stores()

    # 사이드바
    with st.sidebar:
        st.markdown("### ⚙️ 설정")
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.caption("동유럽 여행기 + 구글맵 가이드 + 웹 검색을 결합한 AI 여행 챗봇")
        st.divider()
        if os.path.exists(COVER_PATH):
            st.image(COVER_PATH, width="stretch")
        st.markdown(f"[📖 교보문고에서 책 보기]({BOOK_URL})")

    # ── 앱 타이틀 (항상 표시) ─────────────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;margin-bottom:0.5rem">'
        '<span style="font-size:2rem">✈️</span><br>'
        '<span style="font-size:1.7rem;font-weight:800;color:#1e293b;letter-spacing:-0.5px">'
        '똑똑한 유럽여행 도우미</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── 첫 화면 ───────────────────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown(
            '<p style="text-align:center;color:#64748b;font-size:0.95rem;margin:4px 0 28px">'
            "작가의 생생한 여행기와 구글맵 가이드를 기반으로 답해드립니다</p>",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2, gap="small")
        for i, sug in enumerate(SUGGESTIONS):
            with (col1 if i % 2 == 0 else col2):
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state.pending = sug
                    st.rerun()

        st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
        chips_inner = tip_chips_html().replace('<div class="chip-wrap">', '').replace('</div>', '')
        st.markdown(
            f'''<div style="border:1.5px solid #e2e8f0;border-radius:16px;padding:18px 22px;
                background:white;box-shadow:0 2px 10px rgba(0,0,0,0.04);">
              <p style="color:#475569;font-size:0.85rem;font-weight:600;margin:0 0 12px 0">
                📱 유럽 여행 필수 앱 활용법</p>
              <div class="chip-wrap" style="margin:0">{chips_inner}</div>
            </div>''',
            unsafe_allow_html=True,
        )

    # ── 채팅 화면 ─────────────────────────────────────────────────────────────
    else:
        st.markdown(
            '<p style="text-align:center;color:#94a3b8;font-size:0.85rem;margin:2px 0 16px">'
            "여행기 · 구글맵 가이드 · 최신 인터넷 정보를 함께 검색해서 답변드려요</p>",
            unsafe_allow_html=True,
        )

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("images"):
                    st.caption("🗺️ 구글맵 가이드 슬라이드")
                    cols = st.columns(min(len(msg["images"]), 2))
                    for i, p in enumerate(msg["images"]):
                        if os.path.exists(p):
                            with cols[i % 2]:
                                st.image(p, width="stretch")
                if msg.get("web_results"):
                    with st.expander("🌐 웹 검색 결과 보기"):
                        for r in msg["web_results"]:
                            st.markdown(f"- [{r['title']}]({r['url']})")

        if st.session_state.messages and st.session_state.messages[-1].get("is_gmap"):
            st.markdown(
                '<p style="font-size:0.85rem;font-weight:600;color:#2563eb;margin-bottom:2px">📱 유럽여행 앱 활용 질문을 더 해보세요.</p>',
                unsafe_allow_html=True,
            )
            st.markdown(tip_chips_html(), unsafe_allow_html=True)

    # ── 채팅 입력 ─────────────────────────────────────────────────────────────
    user_input = st.chat_input("유럽 여행, 구글맵 사용법 등 무엇이든 물어보세요...")

    question = st.session_state.pending or user_input
    if st.session_state.pending:
        st.session_state.pending = None

    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        box = st.empty()
        gmap = is_gmap(question)
        tour = is_tour(question)
        images = get_images(question) if gmap else []

        with st.spinner("📖 여행 자료 검색 중..."):
            pdf_ctx = search_pdf(question, stores)

        web_results = []
        if not gmap:
            with st.spinner("🌐 최신 정보 검색 중..."):
                web_results = tavily_search(question)

        web_ctx = "\n\n".join(
            f"[웹 {i+1}] {r['title']}\n{r['content']}\n출처: {r['url']}"
            for i, r in enumerate(web_results)
        )

        history = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in st.session_state.messages[:-1]
        ]

        handler = StreamHandler(box)
        llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.7, streaming=True, callbacks=[handler])
        llm.invoke([SystemMessage(content=build_prompt(pdf_ctx, web_ctx, gmap, tour))] + history + [HumanMessage(content=question)])
        box.markdown(handler.text)

        if images:
            st.caption("🗺️ 구글맵 가이드 슬라이드")
            cols = st.columns(min(len(images), 2))
            for i, p in enumerate(images):
                if os.path.exists(p):
                    with cols[i % 2]:
                        st.image(p, width="stretch")

        if web_results:
            with st.expander("🌐 웹 검색 결과 보기"):
                for r in web_results:
                    st.markdown(f"- [{r['title']}]({r['url']})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": handler.text,
        "images": images,
        "web_results": web_results,
        "is_gmap": gmap,
    })

    # 답변 후 채팅 영역으로 스크롤
    import streamlit.components.v1 as components
    components.html("""
<script>
    var container = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
    if (container) container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
</script>
""", height=0)

    st.rerun()


if __name__ == "__main__":
    main()
