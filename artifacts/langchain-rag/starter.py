import os
import sys

sys.path.insert(0, "/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages")

# ─── API 키 y prefix 보정 ─────────────────────────────────────────────────────
_key = os.environ.get("OPENAI_API_KEY", "")
if _key.startswith("y") and _key[1:].startswith("sk-"):
    os.environ["OPENAI_API_KEY"] = _key[1:]

import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# ─── 페이지 설정 ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LangChain RAG 문서봇",
    page_icon="📄",
    layout="wide",
)

# ─── 제목 ─────────────────────────────────────────────────────────────────────
st.title("📄 LangChain RAG 파이프라인")
st.markdown("""
**LangChain의 주요 개념 실습**: Loader → Splitter → Storage → Retriever → Generator
""")

# ─── 파이프라인 다이어그램 ────────────────────────────────────────────────────
with st.expander("🔗 파이프라인 구조 보기", expanded=False):
    st.code("""
PDF 파일
   │
   ▼
[1] Loader       : PDF에서 텍스트 추출 (PyPDF2)
   │
   ▼
[2] Splitter     : 텍스트를 청크로 분할 (CharacterTextSplitter)
                   chunk_size=1000 / chunk_overlap=200
   │
   ▼
[3] Storage      : 청크를 벡터 DB에 저장 (FAISS + OpenAI Embeddings)
   │
   ▼
[4] Retriever    : 질문과 유사한 문서 검색 (similarity_search, k=3)
   │
   ▼
[5] Generator    : 검색 결과 기반으로 LLM이 답변 생성 (gpt-4.1-mini)
""", language="text")

# ─── 1. Loader ────────────────────────────────────────────────────────────────
st.header("① Loader — PDF 텍스트 추출")


def get_pdf_text(uploaded_file) -> str:
    """PDF 파일에서 텍스트를 추출합니다."""
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text


uploaded_file = st.file_uploader("📎 PDF 파일을 업로드하세요", type=["pdf"])

if uploaded_file:
    with st.spinner("PDF 텍스트 추출 중..."):
        raw_text = get_pdf_text(uploaded_file)
    st.success(f"✅ 텍스트 추출 완료: **{len(raw_text):,}** 글자")
    with st.expander("추출된 텍스트 미리보기 (앞 500자)"):
        st.text(raw_text[:500])

    # ─── 2. Splitter ─────────────────────────────────────────────────────────
    st.header("② Splitter — 청크 분할")

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider("Chunk Size (최대 글자 수)", 300, 2000, 1000, 100)
    with col2:
        chunk_overlap = st.slider("Chunk Overlap (겹치는 글자 수)", 0, 500, 200, 50)

    with st.spinner("청크 분할 중..."):
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_splits = text_splitter.create_documents([raw_text])

    st.info(f"📦 총 **{len(all_splits)}**개의 청크 생성됨")
    with st.expander("청크 예시 (첫 3개)"):
        for i, chunk in enumerate(all_splits[:3]):
            st.markdown(f"**청크 {i+1}** ({len(chunk.page_content)}자)")
            st.text(chunk.page_content[:300])
            st.divider()

    # ─── 3. Storage ──────────────────────────────────────────────────────────
    st.header("③ Storage — 벡터 DB 저장 (FAISS)")

    if "vectorstore" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
        with st.spinner("OpenAI 임베딩 생성 후 FAISS 저장 중..."):
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)
            st.session_state["vectorstore"] = vectorstore
            st.session_state["last_file"] = uploaded_file.name
        st.success(f"✅ {len(all_splits)}개 청크가 벡터 DB에 저장됨")
    else:
        vectorstore = st.session_state["vectorstore"]
        st.success("✅ 기존 벡터 DB 재사용 중")

    # ─── 4. Retriever + 5. Generator (Chaining) ──────────────────────────────
    st.header("④ Retriever + ⑤ Generator — 질문 & 답변")

    col_k, col_model = st.columns(2)
    with col_k:
        k = st.slider("Retriever top-k (검색할 문서 수)", 1, 10, 3)
    with col_model:
        model_name = st.selectbox("LLM 모델", ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1"])

    query_text = st.text_input("💬 질문을 입력하세요", placeholder="예: 임차인의 대항력은 어떻게 취득하나요?")

    if query_text:
        # ── 4. Retriever ──────────────────────────────────────────────────────
        with st.spinner("관련 문서 검색 중..."):
            docs_list = vectorstore.similarity_search(query_text, k=k)
            docs = ""
            for i, doc in enumerate(docs_list):
                docs += f"'문서{i+1}':{doc.page_content}\n"

        with st.expander(f"🔍 검색된 관련 문서 {len(docs_list)}개"):
            for i, doc in enumerate(docs_list):
                st.markdown(f"**문서 {i+1}**")
                st.text(doc.page_content[:300])
                st.divider()

        # ── 5. Generator (Chaining) ───────────────────────────────────────────
        st.markdown("**🤖 답변**")
        answer_box = st.empty()

        # rag_prompt 구성 (SystemMessage + HumanMessage 체이닝)
        rag_prompt = [
            SystemMessage(
                content="너는 문서에 대해 질의응답을 하는 '문서봇'이야. 주어진 문서를 참고하여 사용자의 질문에 답변을 해줘. 문서에 내용이 정확하게 나와있지 않으면 대답하지 마."
            ),
            HumanMessage(
                content=f"질문:{query_text}\n\n{docs}"
            ),
        ]

        # 스트리밍 콜백 구현
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        from langchain.callbacks.base import BaseCallbackHandler

        class StreamlitCallbackHandler(BaseCallbackHandler):
            def __init__(self, container):
                self.container = container
                self.text = ""

            def on_llm_new_token(self, token: str, **kwargs):
                self.text += token
                self.container.markdown(self.text)

        callback = StreamlitCallbackHandler(answer_box)

        llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
            streaming=True,
            callbacks=[callback],
        )

        with st.spinner("LLM 답변 생성 중..."):
            response = llm(rag_prompt)

        st.divider()
        st.caption(f"모델: `{model_name}` | 검색 문서: `{k}개` | 청크: `{len(all_splits)}개`")

else:
    st.info("👆 왼쪽 상단에서 PDF 파일을 업로드하면 파이프라인이 시작됩니다.")

    st.markdown("---")
    st.markdown("""
### 📚 LangChain RAG 파이프라인 개요

| 단계 | 모듈 | 역할 |
|------|------|------|
| ① | **Loader** | PDF에서 텍스트 추출 |
| ② | **Splitter** | 텍스트를 청크로 분할 |
| ③ | **Storage** | 청크를 벡터 DB에 저장 |
| ④ | **Retriever** | 질문과 유사한 문서 검색 |
| ⑤ | **Generator** | LLM이 문서 기반으로 답변 생성 |

**Chain = Retriever → Generator** (검색 결과를 LLM에게 context로 전달)
""")
