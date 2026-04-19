import os
import sys
import re
import json
import requests
import numpy as np

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
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "artifacts", "api-server", "data")
SLIDES_DIR = os.path.join(ROOT_DIR, "artifacts", "api-server", "public", "gmap-slides")
COVER_PATH = os.path.join(ROOT_DIR, "artifacts", "pdf-chatbot", "public", "book-cover.png")

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="똑똑한 유럽여행 도우미",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* 전체 폰트 크기 시니어 친화적으로 */
html, body, [class*="css"] { font-size: 17px !important; }
h1 { font-size: 2rem !important; }
h2 { font-size: 1.5rem !important; }
.stButton > button {
    font-size: 16px !important;
    padding: 10px 16px !important;
    border-radius: 10px !important;
    border: 1.5px solid #4a90d9 !important;
    background: white !important;
    color: #1a1a2e !important;
    margin: 3px !important;
    white-space: normal !important;
    text-align: left !important;
}
.stButton > button:hover {
    background: #e8f0fe !important;
    border-color: #1a73e8 !important;
}
.stChatMessage { font-size: 16px !important; }
.suggestion-btn > button {
    background: linear-gradient(135deg,#1a73e8,#4a90d9) !important;
    color: white !important;
    border: none !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)


# ── PDF 로딩 & FAISS 인덱싱 (캐시) ───────────────────────────────────────────
@st.cache_resource(show_spinner="📚 여행 자료를 불러오는 중...")
def load_vector_stores():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    stores = {}
    for name, fname in [("travel", "travel.pdf"), ("googlemap", "googlemap.pdf")]:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            continue
        reader = PdfReader(fpath)
        raw = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                raw += t
        splitter = CharacterTextSplitter(
            separator="\n\n", chunk_size=800, chunk_overlap=100
        )
        docs = splitter.create_documents([raw])
        if docs:
            stores[name] = FAISS.from_documents(docs, embeddings)
    return stores


# ── 구글맵 관련 질문 감지 ─────────────────────────────────────────────────────
def is_gmap_question(q: str) -> bool:
    pat = (
        r"구글\s*맵|구글\s*지도|google\s*map|길찾기|내비|나침반|스트리트뷰|위성\s*지도|"
        r"장소\s*검색|즐겨찾기\s*저장|오프라인\s*지도|라이브\s*뷰|live\s*view|"
        r"위치\s*공유|리뷰|평점|후기|기차\s*예약|유럽\s*기차|열차\s*예약|train|rail|"
        r"버스\s*예약|비행기\s*예약|번역기|실시간\s*대화|구글\s*번역|준비물|짐\s*싸기|"
        r"챙길\s*것|packing"
    )
    return bool(re.search(pat, q, re.IGNORECASE))


# ── 현지 투어 관련 질문 감지 ──────────────────────────────────────────────────
def is_tour_question(q: str) -> bool:
    pat = r"투어|tour|주의사항|현지.*예약|예약.*주의|당일치기|야경.*투어|투어.*예약"
    return bool(re.search(pat, q, re.IGNORECASE))


# ── 슬라이드 이미지 목록 반환 ─────────────────────────────────────────────────
def page_path(n: int) -> str:
    return os.path.join(SLIDES_DIR, f"page-{str(n).zfill(3)}.png")


TOPIC_IMAGES = [
    {"pat": r"설치|다운로드|플레이\s*스토어|install",
     "pages": [3, 4, 5, 6]},
    {"pat": r"화면\s*구성|인터페이스|검색창|나침반|현재\s*위치|기본\s*화면",
     "pages": [15, 16, 17, 18]},
    {"pat": r"지도\s*유형|위성|지형|3d|스트리트뷰",
     "pages": [19, 20, 21, 22, 24, 26]},
    {"pat": r"맛집|음식점|레스토랑|카페|브런치",
     "pages": [31, 32, 33, 34, 35]},
    {"pat": r"리뷰|평점|후기|별점",
     "pages": [43, 44, 45]},
    {"pat": r"길찾기|경로|내비|navigation|도보|대중교통|자전거",
     "pages": [48, 51, 52, 53, 55, 56]},
    {"pat": r"라이브\s*뷰|live\s*view|증강현실|ar",
     "pages": [83, 84, 85, 86, 87, 88, 89]},
    {"pat": r"즐겨찾기|저장|목록|나만의\s*여행",
     "pages": [93, 94, 95, 96]},
    {"pat": r"위치\s*공유|현재\s*위치\s*보내|동선|공유.*위치",
     "files": ["mylocation-1.png", "mylocation-2.png", "mylocation-3.png",
               "mylocation-4.png", "mylocation-5.png"]},
    {"pat": r"기차\s*예약|유럽\s*기차|열차\s*예약|train|rail|버스\s*예약|비행기\s*예약",
     "files": [f"train-{i}.png" for i in range(1, 9)]},
    {"pat": r"번역기|실시간\s*대화|구글\s*번역|google\s*translate",
     "files": [f"translate-{i}.png" for i in range(1, 7)]},
    {"pat": r"준비물|짐\s*싸기|챙길\s*것|packing|체크리스트.*여행|여행.*체크리스트",
     "files": [f"packing-{i}.png" for i in range(1, 6)]},
]


def get_relevant_images(q: str, max_imgs: int = 6) -> list:
    matched_files = []
    matched_pages = []
    for topic in TOPIC_IMAGES:
        if re.search(topic["pat"], q, re.IGNORECASE):
            if "files" in topic:
                matched_files.extend(topic["files"])
            if "pages" in topic:
                matched_pages.extend(topic["pages"])
    if matched_files:
        paths = [os.path.join(SLIDES_DIR, f) for f in matched_files]
        return [p for p in paths if os.path.exists(p)][:max_imgs]
    if matched_pages:
        paths = [page_path(n) for n in matched_pages]
        return [p for p in paths if os.path.exists(p)][:max_imgs]
    # 기본 이미지
    defaults = [page_path(1), page_path(7), page_path(15)]
    return [p for p in defaults if os.path.exists(p)]


# ── Tavily 웹 검색 ────────────────────────────────────────────────────────────
def tavily_search(query: str) -> list:
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": False,
            },
            timeout=10,
        )
        if not resp.ok:
            return []
        data = resp.json()
        results = []
        for r in data.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:600],
            })
        return results
    except Exception:
        return []


# ── 시스템 프롬프트 생성 ──────────────────────────────────────────────────────
def build_system_prompt(pdf_context: str, web_context: str,
                        is_gmap_q: bool, is_tour_q: bool) -> str:
    prompt = """당신은 친절하고 유용한 동유럽 여행 및 구글맵 사용법 전문 챗봇입니다.
질문에 자세하고 실용적으로 답해주세요. 질문과 같은 언어로 답변하세요 (한국어 질문 → 한국어 답변).

## 관련 사이트 안내 지침
답변 내용에 따라 아래 사이트를 자연스럽게 언급하고, URL을 그대로 포함해 주세요:

- 투어·액티비티 예약: GetYourGuide (https://www.getyourguide.com)
- 체코·슬로바키아 셔틀버스: CK Shuttle (https://www.ckshuttle.cz)
- 동유럽 버스: FlixBus (https://www.flixbus.com), RegioJet (https://www.regiojet.com)
- 기차·교통 통합 검색: Omio (https://www.omio.com), Trainline (https://www.thetrainline.com)
- 숙소 예약: Booking.com (https://www.booking.com), Hostelworld (https://www.hostelworld.com)
- 빈·잘츠부르크 교통: ÖBB (https://www.oebb.at)
- 체코 철도: České dráhy (https://www.cd.cz)
- 헝가리 철도: MÁV (https://www.mavcsoport.hu)
- 할슈타트 관련: 잘츠카머구트 공식 (https://www.hallstatt.net)
- 크리스마스 마켓 정보: Austria Tourism (https://www.austria.info)

관련 사이트가 있을 때만 언급하세요.

## 동유럽 도시 간 이동·택시·셔틀 질문 필수 안내
사용자가 동유럽 도시 간 이동, 택시, 셔틀, 교통수단 등을 묻는 경우 반드시 아래 두 서비스를 모두 안내하세요:
- CK Shuttle (https://www.ckshuttle.cz): 프라하·빈·부다페스트·잘츠부르크 연결 셔틀버스
- Bean Shuttle (https://www.beanshuttle.com): 숙소 픽업·드롭오프 포함 프라이빗 셔틀

## 잘츠부르크 → 할슈타트 이동 질문 필수 안내
직행버스 없음. 기차+페리 조합 또는 GetYourGuide 당일치기 투어 추천: https://www.getyourguide.com

## 빈(비엔나) 미술사 박물관 사진 명소 질문 필수 안내
알베르티나 미술관(Albertina Museum) 2층 야외 테라스 — 영화 《비포 선셋(Before Sunset)》 촬영지. 무료 입장 가능.

## 현지 투어 예약 주의사항 질문 필수 안내
- 현지 투어는 가능한 한 빨리, 도착 당일이나 다음날 하는 것을 강력 추천합니다.
- 작가의 실제 경험: 부다페스트 야경 투어 당일 짙은 안개 → 다음날 어부의 요새(Fisherman's Bastion)에서 재도전 성공.
- 작가가 직접 경험한 현지 투어 사례들:
  - 리스본 → 신트라 당일치기 / 런던 → 옥스퍼드 당일치기 / 런던 야경 투어
  - 로마 → 폼페이·포지타노 남부 투어 / 에딘버러 → 하이랜드 투어 / 타이페이 → 지우펀 투어
- GetYourGuide, 클룩(Klook) 예약 시 Free cancellation 필수 확인."""

    if is_gmap_q:
        prompt += """

## 구글맵 관련 질문 지침
구글맵 사용법을 물어보는 경우, 아래 유튜브 영상을 반드시 참고 자료로 언급해 주세요:
- 추천 영상: https://youtu.be/oZ57SmPTh9s
단계별로 쉽게 설명하고, 해외 여행자에게 특히 유용한 기능을 강조하세요."""

    if is_tour_q:
        prompt += """

## ⚠️ 이 질문은 현지 투어 예약 관련 질문입니다. 아래 내용을 반드시 모두 포함하여 답변하세요.

핵심 팁: 현지 투어는 도착 당일이나 다음날 최대한 빨리 예약하고 진행하세요.
날씨 등 예상치 못한 상황으로 실패해도 일정 안에 재도전할 수 있습니다.

작가의 실제 경험 — 부다페스트 야경 투어:
부다페스트에서 야경 투어를 했는데 당일 짙은 안개가 껴서 야경을 제대로 보지 못했습니다.
다음날 직접 어부의 요새(Fisherman's Bastion)에 올라가서 야경을 다시 감상했습니다.
만약 마지막 날에 투어를 했다면 아예 못 봤을 거예요. 투어는 일정 초반에!

작가가 직접 경험한 현지 투어 사례 목록:
1. 포르투갈 리스본 → 신트라(Sintra) 당일치기: 동화 같은 궁전 마을, 차로 40분
2. 영국 런던 → 옥스퍼드(Oxford) 당일치기: 해리포터 촬영지, 버스 1시간 30분
3. 영국 런던 → 런던 야경 투어: 템스강변·타워브리지·빅벤 야경 가이드 투어
4. 이탈리아 로마 → 남부 투어(폼페이·포지타노): 고대 도시와 아말피 해안을 하루에
5. 스코틀랜드 에딘버러 → 하이랜드 투어: 네스 호수·글렌코, 운전 불필요 → 시니어 추천
6. 대만 타이페이 → 지우펀(九份) 투어: 《센과 치히로》 배경, 저녁 야경 최고
7. 부다페스트 어부의 요새: 야경 재도전 명소, 무료 입장 가능

예약 플랫폼: GetYourGuide (https://www.getyourguide.com), 클룩(Klook) — Free cancellation 표시 필수 확인
집결 장소: 구글맵에 미리 저장해두세요."""

    if pdf_context:
        prompt += f"\n\n## 참고 자료 (여행기 & 구글맵 가이드)\n{pdf_context}"
    if web_context:
        prompt += f"\n\n## 최신 인터넷 정보\n{web_context}"
    if pdf_context or web_context:
        prompt += "\n\n위 정보를 종합하여 답변하세요. 출처(여행기 경험 / 구글맵 가이드 / 웹 정보)를 구분해서 설명하면 더 좋습니다."

    return prompt


# ── RAG 검색 ──────────────────────────────────────────────────────────────────
def search_pdf(question: str, stores: dict, k: int = 4) -> str:
    results = []
    for label, store in [("여행기", stores.get("travel")),
                         ("구글맵 가이드", stores.get("googlemap"))]:
        if store is None:
            continue
        docs = store.similarity_search(question, k=k)
        for doc in docs:
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


# ── 제안 질문 ─────────────────────────────────────────────────────────────────
SUGGESTIONS = [
    "빈 미술사 박물관 인생 사진 명소?",
    "현지 투어 예약 전 주의사항!",
    "유럽여행 필수준비물이 뭐야?",
    "구글맵 길찾기 어떻게 써?",
    "프라하에서 빈 이동 방법?",
    "기차 예약은 어디서 해?",
]

TRAVEL_TIPS = [
    ("🗺️ 길찾기", "구글맵으로 길찾기 하는 방법을 알려줘"),
    ("⭐ 리뷰 보기", "구글맵에서 맛집 리뷰 보는 방법을 알려줘"),
    ("👁️ 라이브뷰", "구글맵 라이브뷰 사용법을 알려줘"),
    ("📍 위치공유", "구글맵으로 위치 공유하는 방법을 알려줘"),
    ("🚂 기차예약", "유럽 기차 예약 방법을 알려줘"),
    ("🗣️ 번역기", "구글 번역기 실시간 대화 기능 알려줘"),
]


# ── 메인 앱 ──────────────────────────────────────────────────────────────────
def main():
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_tips" not in st.session_state:
        st.session_state.show_tips = False
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    # 벡터 스토어 로딩
    stores = load_vector_stores()

    # ── 헤더 ────────────────────────────────────────────────────────────────
    col_title, col_cover = st.columns([5, 1])
    with col_title:
        st.title("✈️ 똑똑한 유럽여행 도우미")
        st.markdown(
            "동유럽 여행기 + 구글맵 활용법 + 실시간 웹 검색으로 답변해 드립니다. "
            "**시니어 여행자를 위한 쉽고 친절한 안내!**"
        )
    with col_cover:
        if os.path.exists(COVER_PATH):
            st.image(COVER_PATH, width=100)
            st.markdown(
                '<a href="https://product.kyobobook.co.kr/detail/S000215426392" '
                'target="_blank" style="font-size:13px;">📖 책 보기</a>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── 채팅 이력 표시 ──────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # 이미지가 있으면 표시
            if msg.get("images"):
                cols = st.columns(min(len(msg["images"]), 3))
                for i, img_path in enumerate(msg["images"]):
                    if os.path.exists(img_path):
                        with cols[i % 3]:
                            st.image(img_path, use_container_width=True)

    # ── 처음 화면: 제안 버튼 ───────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("### 💬 무엇이 궁금하세요?")

        # 앱 활용법 토글 버튼
        tip_label = "📱 유럽 여행 필수 앱 활용법 ▼" if not st.session_state.show_tips else "📱 유럽 여행 필수 앱 활용법 ▲"
        if st.button(tip_label, key="toggle_tips", use_container_width=False):
            st.session_state.show_tips = not st.session_state.show_tips
            st.rerun()

        # 앱 활용법 서브 버튼
        if st.session_state.show_tips:
            cols = st.columns(3)
            for i, (label, question) in enumerate(TRAVEL_TIPS):
                with cols[i % 3]:
                    if st.button(label, key=f"tip_{i}", use_container_width=True):
                        st.session_state.pending_question = question
                        st.session_state.show_tips = False
                        st.rerun()

        st.markdown("---")

        # 제안 질문 버튼
        cols2 = st.columns(2)
        for i, sug in enumerate(SUGGESTIONS):
            with cols2[i % 2]:
                if st.button(f"💡 {sug}", key=f"sug_{i}", use_container_width=True):
                    st.session_state.pending_question = sug
                    st.rerun()

    # ── 채팅 입력 ───────────────────────────────────────────────────────────
    user_input = st.chat_input("질문을 입력하세요... (예: 프라하에서 빈까지 어떻게 가나요?)")

    # pending_question 또는 직접 입력 처리
    question = st.session_state.pending_question or user_input
    if st.session_state.pending_question:
        st.session_state.pending_question = None

    if question:
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # ── 답변 생성 ──────────────────────────────────────────────────────
        with st.chat_message("assistant"):
            answer_box = st.empty()

            is_gmap_q = is_gmap_question(question)
            is_tour_q = is_tour_question(question)

            # 이미지 검색 (구글맵/앱 관련 질문)
            images = []
            if is_gmap_q:
                images = get_relevant_images(question)

            # PDF RAG 검색
            with st.spinner("📖 여행 자료 검색 중..."):
                pdf_context = search_pdf(question, stores)

            # 웹 검색 (구글맵 질문 제외)
            web_results = []
            if not is_gmap_q:
                with st.spinner("🌐 최신 정보 검색 중..."):
                    web_results = tavily_search(question)

            web_context = ""
            if web_results:
                web_context = "\n\n".join(
                    f"[웹 {i+1}] {r['title']}\n{r['content']}\n출처: {r['url']}"
                    for i, r in enumerate(web_results)
                )

            # 시스템 프롬프트 빌드
            system_prompt = build_system_prompt(
                pdf_context, web_context, is_gmap_q, is_tour_q
            )

            # 대화 이력 메시지 구성
            messages = [SystemMessage(content=system_prompt)]
            for m in st.session_state.messages[:-1]:  # 마지막 user 제외 (아래서 추가)
                if m["role"] == "user":
                    messages.append(HumanMessage(content=m["content"]))
            messages.append(HumanMessage(content=question))

            # 스트리밍 LLM 호출
            handler = StreamHandler(answer_box)
            llm = ChatOpenAI(
                model_name="gpt-4.1-mini",
                temperature=0.7,
                streaming=True,
                callbacks=[handler],
            )
            response = llm.invoke(messages)
            final_answer = handler.text
            answer_box.markdown(final_answer)

            # 이미지 표시
            if images:
                st.markdown("---")
                st.markdown("**📸 관련 슬라이드 이미지**")
                cols = st.columns(min(len(images), 3))
                for i, img_path in enumerate(images):
                    if os.path.exists(img_path):
                        with cols[i % 3]:
                            st.image(img_path, use_container_width=True)

        # 어시스턴트 메시지 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "images": images,
        })

    # ── 사이드바: 대화 초기화 ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ 설정")
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.show_tips = False
            st.rerun()
        st.markdown("---")
        st.markdown(
            "**똑똑한 유럽여행 도우미**\n\n"
            "동유럽 여행기 PDF + 구글맵 강의 PDF + Tavily 웹 검색을 결합한 AI 여행 챗봇입니다."
        )
        if os.path.exists(COVER_PATH):
            st.image(COVER_PATH, width=150)
            st.markdown(
                '[📖 교보문고에서 책 보기](https://product.kyobobook.co.kr/detail/S000215426392)'
            )


if __name__ == "__main__":
    main()
