import os
import sys
import re
import requests

sys.path.insert(0, "/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages")

# ── OpenAI 키 y-prefix 보정 ─────────────────────────────────────────────────
_key = os.environ.get("OPENAI_API_KEY", "")
if _key.startswith("y") and _key[1:].startswith("sk-"):
    os.environ["OPENAI_API_KEY"] = _key[1:]

import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "artifacts", "api-server", "data")
SLIDES_DIR = os.path.join(ROOT_DIR, "artifacts", "api-server", "public", "gmap-slides")
COVER_PATH = os.path.join(ROOT_DIR, "artifacts", "pdf-chatbot", "public", "book-cover.png")

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="똑똑한 유럽여행 도우미",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ── 밝은 배경 강제 (다크모드 방지) ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stMain"],
section[data-testid="stSidebar"],
.main .block-container {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}
[data-testid="stChatInput"] textarea { background: #f8f8f8 !important; color: #1a1a1a !important; }
[data-testid="stChatMessageContent"] { color: #1a1a1a !important; }

/* ── 기본 폰트 크기 ── */
html, body, [class*="css"] { font-size: 18px !important; }

/* ── 노란 제안 버튼 (suggestion) ── */
.sug-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin: 8px 0 20px 0;
}
.sug-btn {
    display: block;
    background: #FFD600;
    color: #1a1a1a;
    font-size: 17px;
    font-weight: 700;
    padding: 20px 14px;
    border-radius: 14px;
    text-decoration: none;
    text-align: center;
    border: 2px solid #e6c000;
    word-break: keep-all;
    line-height: 1.4;
    transition: background 0.15s;
}
.sug-btn:hover { background: #ffc400; color: #1a1a1a; text-decoration: none; }

/* ── 하단 도구 상자 ── */
.toolbox-wrap {
    background: #f0f2f6;
    border-radius: 16px;
    padding: 18px 20px 20px 20px;
    margin: 4px 0 12px 0;
}
.toolbox-title {
    font-size: 17px;
    font-weight: 800;
    color: #1a1a2e;
    margin: 0 0 14px 0;
}
.toolbox-scroll {
    display: flex;
    overflow-x: auto;
    gap: 10px;
    padding-bottom: 6px;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: thin;
}
.pill-btn {
    display: inline-block;
    background: #FF6B6B;
    color: white;
    font-size: 16px;
    font-weight: 700;
    padding: 14px 22px;
    border-radius: 28px;
    text-decoration: none;
    white-space: nowrap;
    flex-shrink: 0;
    transition: background 0.15s;
}
.pill-btn:hover { background: #e05555; color: white; text-decoration: none; }

/* ── 헤더 텍스트 ── */
.hero-title {
    font-size: 2.0rem;
    font-weight: 900;
    color: #1a1a2e;
    line-height: 1.25;
    word-break: keep-all;
    margin: 0 0 4px 0;
}
.hero-sub {
    font-size: 1.1rem;
    font-weight: 500;
    color: #444;
    line-height: 1.25;
    word-break: keep-all;
    margin: 0 0 18px 0;
}

/* ── 채팅 메시지 ── */
.stChatMessage { font-size: 17px !important; }

/* ── 기본 st.button 없애지 않음 (사이드바 초기화 버튼 등) ── */
.stButton > button {
    font-size: 16px !important;
    padding: 10px 18px !important;
    border-radius: 10px !important;
}

/* ── 책 표지 가운데 ── */
.cover-center { display:flex; justify-content:center; margin-bottom:18px; }
.cover-center img { border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.18); }
.book-link {
    display:block;
    text-align:center;
    font-size:14px;
    color:#1a73e8;
    margin-top:6px;
    text-decoration:none;
}
.book-link:hover { text-decoration:underline; }
</style>
""", unsafe_allow_html=True)

# ── 콘텐츠 정의 ───────────────────────────────────────────────────────────────
SUGGESTIONS = [
    ("💡 빈 미술사 박물관 인생 사진 명소?",   "빈 미술사 박물관 인생 사진 명소?"),
    ("💡 현지 투어 예약 전 주의사항!",         "현지 투어 예약 전 주의사항!"),
    ("💡 유럽여행 필수 준비물이 뭐야?",        "유럽여행 필수준비물이 뭐야?"),
    ("💡 구글맵 길찾기 어떻게 써?",            "구글맵으로 길찾기 하는 방법을 알려줘"),
    ("💡 프라하에서 빈 이동 방법?",            "프라하에서 빈 이동 방법이 뭐야?"),
    ("💡 기차 예약은 어디서 해?",              "유럽 기차 예약 방법을 알려줘"),
]

TRAVEL_TIPS = [
    ("🗺️ 길찾기",   "구글맵으로 길찾기 하는 방법을 알려줘"),
    ("⭐ 리뷰 보기", "구글맵에서 맛집 리뷰 보는 방법을 알려줘"),
    ("👁️ 라이브뷰",  "구글맵 라이브뷰 사용법을 알려줘"),
    ("📍 위치공유",  "구글맵으로 위치 공유하는 방법을 알려줘"),
    ("🚂 기차예약",  "유럽 기차 예약 방법을 알려줘"),
    ("🗣️ 번역기",   "구글 번역기 실시간 대화 기능 알려줘"),
]


# ── PDF 로딩 & FAISS (캐시) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="📚 여행 자료를 불러오는 중...")
def load_vector_stores():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    stores = {}
    for name, fname in [("travel", "travel.pdf"), ("googlemap", "googlemap.pdf")]:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            continue
        reader = PdfReader(fpath)
        raw = "".join(p.extract_text() or "" for p in reader.pages)
        splitter = CharacterTextSplitter(separator="\n\n", chunk_size=800, chunk_overlap=100)
        docs = splitter.create_documents([raw])
        if docs:
            stores[name] = FAISS.from_documents(docs, embeddings)
    return stores


# ── 질문 분류 ─────────────────────────────────────────────────────────────────
def is_gmap_question(q: str) -> bool:
    pat = (r"구글\s*맵|구글\s*지도|google\s*map|길찾기|내비|나침반|스트리트뷰|위성\s*지도|"
           r"장소\s*검색|즐겨찾기\s*저장|오프라인\s*지도|라이브\s*뷰|live\s*view|"
           r"위치\s*공유|리뷰|평점|후기|기차\s*예약|유럽\s*기차|열차\s*예약|train|rail|"
           r"버스\s*예약|비행기\s*예약|번역기|실시간\s*대화|구글\s*번역|준비물|짐\s*싸기|챙길\s*것|packing")
    return bool(re.search(pat, q, re.IGNORECASE))


def is_tour_question(q: str) -> bool:
    return bool(re.search(r"투어|tour|주의사항|현지.*예약|예약.*주의|당일치기|야경.*투어|투어.*예약", q, re.IGNORECASE))


# ── 슬라이드 이미지 ───────────────────────────────────────────────────────────
def page_path(n: int) -> str:
    return os.path.join(SLIDES_DIR, f"page-{str(n).zfill(3)}.png")

TOPIC_IMAGES = [
    {"pat": r"설치|다운로드|플레이\s*스토어|install",                             "pages": [3, 4, 5, 6]},
    {"pat": r"화면\s*구성|인터페이스|검색창|나침반|현재\s*위치|기본\s*화면",      "pages": [15, 16, 17, 18]},
    {"pat": r"지도\s*유형|위성|지형|3d|스트리트뷰",                               "pages": [19, 20, 21, 22, 24, 26]},
    {"pat": r"맛집|음식점|레스토랑|카페|브런치",                                   "pages": [31, 32, 33, 34, 35]},
    {"pat": r"리뷰|평점|후기|별점",                                                "pages": [43, 44, 45]},
    {"pat": r"길찾기|경로|내비|navigation|도보|대중교통|자전거",                   "pages": [48, 51, 52, 53, 55, 56]},
    {"pat": r"라이브\s*뷰|live\s*view|증강현실|ar",                                "pages": [83, 84, 85, 86, 87, 88, 89]},
    {"pat": r"즐겨찾기|저장|목록|나만의\s*여행",                                   "pages": [93, 94, 95, 96]},
    {"pat": r"위치\s*공유|현재\s*위치\s*보내|동선|공유.*위치",
     "files": [f"mylocation-{i}.png" for i in range(1, 6)]},
    {"pat": r"기차\s*예약|유럽\s*기차|열차\s*예약|train|rail|버스\s*예약|비행기\s*예약",
     "files": [f"train-{i}.png" for i in range(1, 9)]},
    {"pat": r"번역기|실시간\s*대화|구글\s*번역|google\s*translate",
     "files": [f"translate-{i}.png" for i in range(1, 7)]},
    {"pat": r"준비물|짐\s*싸기|챙길\s*것|packing",
     "files": [f"packing-{i}.png" for i in range(1, 6)]},
]

def get_relevant_images(q: str, max_imgs: int = 6) -> list:
    matched_files, matched_pages = [], []
    for topic in TOPIC_IMAGES:
        if re.search(topic["pat"], q, re.IGNORECASE):
            matched_files.extend(topic.get("files", []))
            matched_pages.extend(topic.get("pages", []))
    if matched_files:
        return [p for f in matched_files if os.path.exists(p := os.path.join(SLIDES_DIR, f))][:max_imgs]
    if matched_pages:
        return [p for n in matched_pages if os.path.exists(p := page_path(n))][:max_imgs]
    return [p for n in [1, 7, 15] if os.path.exists(p := page_path(n))]


# ── Tavily 웹 검색 ────────────────────────────────────────────────────────────
def tavily_search(query: str) -> list:
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "search_depth": "basic",
                  "max_results": 5, "include_answer": False},
            timeout=10,
        )
        if not resp.ok:
            return []
        return [{"title": r.get("title", ""), "url": r.get("url", ""),
                 "content": r.get("content", "")[:600]}
                for r in resp.json().get("results", [])]
    except Exception:
        return []


# ── 시스템 프롬프트 ───────────────────────────────────────────────────────────
def build_system_prompt(pdf_ctx: str, web_ctx: str, is_gmap: bool, is_tour: bool) -> str:
    prompt = """당신은 친절하고 유용한 동유럽 여행 및 구글맵 사용법 전문 챗봇입니다.
질문에 자세하고 실용적으로 답해주세요. 질문과 같은 언어로 답변하세요 (한국어 질문 → 한국어 답변).

## 관련 사이트 안내 지침
- 투어·액티비티 예약: GetYourGuide (https://www.getyourguide.com)
- 체코·슬로바키아 셔틀버스: CK Shuttle (https://www.ckshuttle.cz)
- 동유럽 버스: FlixBus (https://www.flixbus.com), RegioJet (https://www.regiojet.com)
- 기차·교통 통합 검색: Omio (https://www.omio.com), Trainline (https://www.thetrainline.com)
- 숙소 예약: Booking.com (https://www.booking.com), Hostelworld (https://www.hostelworld.com)
- 빈·잘츠부르크 교통: ÖBB (https://www.oebb.at)
- 체코 철도: České dráhy (https://www.cd.cz) / 헝가리 철도: MÁV (https://www.mavcsoport.hu)
- 할슈타트: https://www.hallstatt.net / Austria Tourism: https://www.austria.info

## 동유럽 도시 간 이동
- CK Shuttle (https://www.ckshuttle.cz): 프라하·빈·부다페스트·잘츠부르크 연결
- Bean Shuttle (https://www.beanshuttle.com): 숙소 픽업·드롭오프 포함 프라이빗 셔틀

## 잘츠부르크 → 할슈타트
직행버스 없음. 기차+페리 조합 또는 GetYourGuide 당일치기 투어 추천: https://www.getyourguide.com

## 빈 미술사 박물관 사진 명소
알베르티나 미술관(Albertina Museum) 2층 야외 테라스 — 영화 《비포 선셋(Before Sunset)》 촬영지. 무료 입장.

## 현지 투어 예약 주의사항
- 투어는 도착 당일이나 다음날 최대한 빨리 예약하세요.
- 작가의 실제 경험: 부다페스트 야경 투어 당일 짙은 안개 → 다음날 어부의 요새(Fisherman's Bastion)에서 재도전 성공.
- 경험한 투어: 리스본→신트라 / 런던→옥스퍼드·야경 / 로마→폼페이·포지타노 / 에딘버러→하이랜드 / 타이페이→지우펀
- GetYourGuide, 클룩(Klook) 예약 시 Free cancellation 필수 확인."""

    if is_gmap:
        prompt += "\n\n## 구글맵 질문 지침\n추천 영상: https://youtu.be/oZ57SmPTh9s\n단계별로 쉽게, 시니어 여행자에게 유용한 기능 강조."

    if is_tour:
        prompt += """

## ⚠️ 현지 투어 예약 관련 질문 — 아래 내용을 반드시 모두 포함하여 답변하세요.

핵심 팁: 현지 투어는 도착 당일이나 다음날 최대한 빨리 예약하고 진행하세요!
날씨 등 예상치 못한 상황으로 실패해도 일정 안에 재도전할 수 있습니다.

작가의 실제 경험 — 부다페스트 야경 투어:
부다페스트에서 야경 투어를 했는데 당일 짙은 안개가 껴서 야경을 제대로 보지 못했습니다.
다음날 직접 어부의 요새(Fisherman's Bastion)에 올라가서 야경을 다시 감상했습니다.
만약 마지막 날에 투어를 했다면 아예 못 봤을 거예요. 투어는 일정 초반에!

작가가 직접 경험한 현지 투어 사례:
1. 포르투갈 리스본 → 신트라(Sintra) 당일치기: 동화 같은 궁전 마을, 차로 40분
2. 영국 런던 → 옥스퍼드(Oxford) 당일치기: 해리포터 촬영지, 버스 1시간 30분
3. 영국 런던 → 런던 야경 투어: 템스강변·타워브리지·빅벤 야경 가이드 투어
4. 이탈리아 로마 → 남부 투어(폼페이·포지타노): 고대 도시와 아말피 해안을 하루에
5. 스코틀랜드 에딘버러 → 하이랜드 투어: 네스 호수·글렌코, 운전 불필요 → 시니어 추천
6. 대만 타이페이 → 지우펀(九份) 투어: 《센과 치히로》 배경, 저녁 야경 최고
7. 부다페스트 어부의 요새: 야경 재도전 명소, 무료 입장 가능

예약 플랫폼: GetYourGuide (https://www.getyourguide.com), 클룩(Klook) — Free cancellation 표시 필수 확인
집결 장소는 구글맵에 미리 저장해두세요."""

    if pdf_ctx:
        prompt += f"\n\n## 참고 자료 (여행기 & 구글맵 가이드)\n{pdf_ctx}"
    if web_ctx:
        prompt += f"\n\n## 최신 인터넷 정보\n{web_ctx}"
    if pdf_ctx or web_ctx:
        prompt += "\n\n위 정보를 종합하여 답변하세요. 출처(여행기 경험 / 구글맵 가이드 / 웹 정보)를 구분해서 설명하면 더 좋습니다."
    return prompt


# ── RAG 검색 ──────────────────────────────────────────────────────────────────
def search_pdf(question: str, stores: dict, k: int = 4) -> str:
    results = []
    for label, store in [("여행기", stores.get("travel")), ("구글맵 가이드", stores.get("googlemap"))]:
        if store:
            for doc in store.similarity_search(question, k=k):
                results.append(f"[{label}] {doc.page_content}")
    return "\n\n".join(results[:6])


# ── 스트리밍 콜백 ─────────────────────────────────────────────────────────────
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")


# ── 도구 상자 HTML ────────────────────────────────────────────────────────────
def render_toolbox():
    pills_html = "".join(
        f'<a href="?q={label}" class="pill-btn">{label}</a>'
        for label, _ in TRAVEL_TIPS
    )
    st.markdown(f"""
<div class="toolbox-wrap">
  <p class="toolbox-title">🔧 📱 유럽여행 필수 앱 활용법</p>
  <div class="toolbox-scroll">
    {pills_html}
  </div>
</div>
""", unsafe_allow_html=True)


# ── 제안 버튼 HTML (노란색 그리드) ───────────────────────────────────────────
def render_suggestions():
    items_html = "".join(
        f'<a href="?q={label}" class="sug-btn">{label}</a>'
        for label, _ in SUGGESTIONS
    )
    st.markdown(f'<div class="sug-grid">{items_html}</div>', unsafe_allow_html=True)


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    # ── query_params 처리 (HTML 링크 클릭) ───────────────────────────────────
    params = st.query_params
    if "q" in params:
        clicked_label = params["q"]
        # 라벨로 실제 질문 찾기
        question_map = {label: q for label, q in SUGGESTIONS}
        question_map.update({label: q for label, q in TRAVEL_TIPS})
        st.session_state.pending_question = question_map.get(clicked_label, clicked_label)
        st.query_params.clear()
        st.rerun()

    stores = load_vector_stores()

    # ── 사이드바 ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ 설정")
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.markdown("---")
        if os.path.exists(COVER_PATH):
            st.image(COVER_PATH, width=160)
            st.markdown("[📖 교보문고에서 책 보기](https://product.kyobobook.co.kr/detail/S000215426392)")
        st.markdown("동유럽 여행기 PDF + 구글맵 강의 PDF + Tavily 웹 검색을 결합한 AI 여행 챗봇입니다.")

    # ── 첫 화면 (대화 없음) ───────────────────────────────────────────────────
    if not st.session_state.messages:
        # 1. 책 표지
        if os.path.exists(COVER_PATH):
            st.markdown('<div class="cover-center">', unsafe_allow_html=True)
            st.image(COVER_PATH, width=180)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(
                '<a href="https://product.kyobobook.co.kr/detail/S000215426392" '
                'target="_blank" class="book-link">📖 교보문고에서 책 보기</a>',
                unsafe_allow_html=True,
            )

        # 2. 헤더 텍스트
        st.markdown("""
<p class="hero-title">작가와 함께하는 설레는 유럽 여행,<br>무엇이든 물어보세요</p>
<p class="hero-sub">작가의 에세이와 구글맵 꿀팁으로 답해드려요.</p>
""", unsafe_allow_html=True)

        # 3. 노란 제안 버튼 그리드
        render_suggestions()

        # 4. 하단 도구 상자 (코랄 알약 버튼)
        render_toolbox()

    else:
        # ── 채팅 화면 ─────────────────────────────────────────────────────────
        st.title("✈️ 똑똑한 유럽여행 도우미")

        # 채팅 이력 표시
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("images"):
                    cols = st.columns(min(len(msg["images"]), 3))
                    for i, img_path in enumerate(msg["images"]):
                        if os.path.exists(img_path):
                            with cols[i % 3]:
                                st.image(img_path, use_container_width=True)

        # 채팅 화면에서도 도구 상자 표시
        render_toolbox()

    # ── 채팅 입력 (항상 하단 고정) ───────────────────────────────────────────
    user_input = st.chat_input("질문을 입력하세요... (예: 프라하에서 빈까지 어떻게 가나요?)")

    question = st.session_state.pending_question or user_input
    if st.session_state.pending_question:
        st.session_state.pending_question = None

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            answer_box = st.empty()

            is_gmap_q = is_gmap_question(question)
            is_tour_q = is_tour_question(question)

            images = get_relevant_images(question) if is_gmap_q else []

            with st.spinner("📖 여행 자료 검색 중..."):
                pdf_context = search_pdf(question, stores)

            web_results = []
            if not is_gmap_q:
                with st.spinner("🌐 최신 정보 검색 중..."):
                    web_results = tavily_search(question)

            web_context = "\n\n".join(
                f"[웹 {i+1}] {r['title']}\n{r['content']}\n출처: {r['url']}"
                for i, r in enumerate(web_results)
            )

            system_prompt = build_system_prompt(pdf_context, web_context, is_gmap_q, is_tour_q)

            messages = [SystemMessage(content=system_prompt)]
            for m in st.session_state.messages[:-1]:
                if m["role"] == "user":
                    messages.append(HumanMessage(content=m["content"]))
            messages.append(HumanMessage(content=question))

            handler = StreamHandler(answer_box)
            llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.7,
                             streaming=True, callbacks=[handler])
            llm.invoke(messages)
            final_answer = handler.text
            answer_box.markdown(final_answer)

            if images:
                st.markdown("---")
                st.markdown("**📸 관련 슬라이드 이미지**")
                cols = st.columns(min(len(images), 3))
                for i, img_path in enumerate(images):
                    if os.path.exists(img_path):
                        with cols[i % 3]:
                            st.image(img_path, use_container_width=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "images": images,
        })
        st.rerun()


if __name__ == "__main__":
    main()
