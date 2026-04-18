import os
import sys
import io

sys.path.insert(0, "/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages")

_key = os.environ.get("OPENAI_API_KEY", "")
if _key.startswith("y") and _key[1:].startswith("sk-"):
    os.environ["OPENAI_API_KEY"] = _key[1:]

import streamlit as st
import fitz  # PyMuPDF

st.set_page_config(page_title="PDF 압축기", page_icon="📦", layout="wide")

st.title("📦 PDF 압축기")
st.markdown("사진이 많아 용량이 큰 PDF에서 이미지를 제거하거나 압축해서 작은 파일로 만들어 드려요.")

# ── 모드 선택 ─────────────────────────────────────────────────────────────────
mode = st.radio(
    "압축 방식을 선택하세요",
    ["🗑️ 이미지 완전 제거 (텍스트만 남김, 최소 용량)", "🖼️ 이미지 품질 낮춤 (페이지 외형 유지)"],
    index=0,
    help="챗봇에 올릴 목적이라면 '이미지 완전 제거'를 추천해요. 텍스트만 필요하거든요!"
)

st.info("💡 여행 챗봇에 업로드할 PDF라면 **이미지 완전 제거**를 선택하세요. 챗봇은 텍스트만 읽어요.")

# ── 파일 업로드 ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("📎 PDF 파일 업로드 (최대 500MB)", type=["pdf"])

if uploaded:
    original_bytes = uploaded.read()
    original_size_mb = len(original_bytes) / (1024 * 1024)

    st.write(f"**원본 크기:** {original_size_mb:.1f} MB ({uploaded.name})")

    if st.button("🚀 압축 시작", type="primary"):
        with st.spinner("PDF 처리 중..."):
            try:
                doc = fitz.open(stream=original_bytes, filetype="pdf")
                total_pages = len(doc)

                progress = st.progress(0, text="페이지 분석 중...")

                if "이미지 완전 제거" in mode:
                    # ── 방법 1: 이미지를 흰색으로 덮어쓰고 객체 삭제 ──────────
                    for i, page in enumerate(doc):
                        img_list = page.get_images(full=True)
                        for img in img_list:
                            try:
                                # 이미지 위치 구해서 흰 사각형으로 덮기 (redact)
                                rects = page.get_image_rects(img[0])
                                for rect in rects:
                                    page.add_redact_annot(rect, fill=(1, 1, 1))
                            except Exception:
                                pass
                        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE)
                        progress.progress((i + 1) / total_pages, text=f"이미지 제거 중... ({i+1}/{total_pages}페이지)")

                    output_bytes_io = io.BytesIO()
                    doc.save(
                        output_bytes_io,
                        garbage=4,        # 미사용 객체 완전 제거
                        deflate=True,
                        deflate_images=True,
                        deflate_fonts=True,
                        clean=True,
                    )
                    label = "이미지제거"

                else:
                    # ── 방법 2: 각 페이지를 낮은 DPI로 렌더링해서 재저장 ──────
                    dpi = 120  # 원본보통 300 DPI → 120 DPI로 축소
                    new_doc = fitz.open()
                    mat = fitz.Matrix(dpi / 72, dpi / 72)

                    for i, page in enumerate(doc):
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                        # 각 페이지를 JPEG으로 압축 후 새 PDF에 삽입
                        jpeg_bytes = pix.tobytes("jpeg", jpg_quality=55)
                        img_doc = fitz.open("pdf", fitz.open("jpeg", jpeg_bytes).convert_to_pdf())
                        new_doc.insert_pdf(img_doc)
                        progress.progress((i + 1) / total_pages, text=f"페이지 압축 중... ({i+1}/{total_pages}페이지)")

                    output_bytes_io = io.BytesIO()
                    new_doc.save(output_bytes_io, garbage=4, deflate=True)
                    label = "이미지압축"

                progress.progress(1.0, text="완료!")
                output_bytes = output_bytes_io.getvalue()
                compressed_size_mb = len(output_bytes) / (1024 * 1024)
                reduction = (1 - compressed_size_mb / original_size_mb) * 100

                # ── 결과 표시 ─────────────────────────────────────────────────
                col1, col2, col3 = st.columns(3)
                col1.metric("원본 크기", f"{original_size_mb:.1f} MB")
                col2.metric("압축 후 크기", f"{compressed_size_mb:.1f} MB")
                col3.metric("용량 감소", f"{reduction:.0f}%", delta=f"-{original_size_mb - compressed_size_mb:.1f} MB")

                # ── 다운로드 버튼 ─────────────────────────────────────────────
                base_name = uploaded.name.replace(".pdf", "")
                output_filename = f"{base_name}_{label}.pdf"

                st.success(f"✅ 압축 완료! {original_size_mb:.1f}MB → {compressed_size_mb:.1f}MB ({reduction:.0f}% 감소)")

                st.download_button(
                    label="⬇️ 압축된 PDF 다운로드",
                    data=output_bytes,
                    file_name=output_filename,
                    mime="application/pdf",
                    type="primary",
                )

                if "이미지 완전 제거" in mode:
                    st.info("📌 이 파일을 여행 챗봇에 업로드하면 이제 용량 문제 없이 사용할 수 있어요!")

            except Exception as e:
                st.error(f"처리 중 오류가 발생했어요: {str(e)}")

else:
    st.markdown("""
---
### 사용 방법

1. **위에서 PDF 파일 업로드** (최대 500MB)
2. **압축 방식 선택**
   - **이미지 완전 제거**: 사진을 모두 없애고 텍스트만 남겨요 → 챗봇용으로 최적
   - **이미지 품질 낮춤**: 사진을 작게 만들어요 → 보기용으로 적합
3. **압축 시작** 클릭
4. **다운로드**해서 여행 챗봇에 업로드하세요!
    """)
