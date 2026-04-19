export default function StreamlitWelcome() {
  const suggestions = [
    "💡 빈 미술사 박물관 인생 사진 명소?",
    "💡 현지 투어 예약 전 주의사항!",
    "💡 유럽여행 필수 준비물이 뭐야?",
    "💡 구글맵 길찾기 어떻게 써?",
    "💡 프라하에서 빈 이동 방법?",
    "💡 기차 예약은 어디서 해?",
  ];

  const tips = [
    "🗺️ 길찾기",
    "⭐ 리뷰 보기",
    "👁️ 라이브뷰",
    "📍 위치공유",
    "🚂 기차예약",
    "🗣️ 번역기",
  ];

  return (
    <div
      style={{
        fontFamily:
          '"Source Sans Pro", "Noto Sans KR", sans-serif',
        background: "#ffffff",
        minHeight: "100vh",
        padding: "2rem 2.5rem 6rem",
        maxWidth: "720px",
        margin: "0 auto",
        boxSizing: "border-box",
      }}
    >
      {/* 책 표지 */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginBottom: "20px" }}>
        <img
          src="/book-cover.png"
          alt="책 표지"
          style={{
            width: "180px",
            borderRadius: "8px",
            boxShadow: "0 4px 16px rgba(0,0,0,0.18)",
          }}
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
          }}
        />
        <a
          href="https://product.kyobobook.co.kr/detail/S000215426392"
          target="_blank"
          rel="noreferrer"
          style={{
            marginTop: "8px",
            fontSize: "14px",
            color: "#1a73e8",
            textDecoration: "none",
          }}
        >
          📖 교보문고에서 책 보기
        </a>
      </div>

      {/* 헤더 텍스트 */}
      <div style={{ marginBottom: "18px" }}>
        <p
          style={{
            fontSize: "1.9rem",
            fontWeight: 900,
            color: "#1a1a2e",
            lineHeight: 1.25,
            wordBreak: "keep-all",
            margin: "0 0 4px 0",
          }}
        >
          작가와 함께하는 설레는 유럽 여행,
          <br />
          무엇이든 물어보세요
        </p>
        <p
          style={{
            fontSize: "1.05rem",
            fontWeight: 500,
            color: "#444",
            lineHeight: 1.25,
            wordBreak: "keep-all",
            margin: 0,
          }}
        >
          작가의 에세이와 구글맵 꿀팁으로 답해드려요.
        </p>
      </div>

      {/* 노란 제안 버튼 6개 — 2열 */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "14px",
          marginBottom: "20px",
        }}
      >
        {suggestions.map((text, i) => (
          <button
            key={i}
            style={{
              background: "#FFD600",
              color: "#1a1a1a",
              fontSize: "16px",
              fontWeight: 700,
              padding: "20px 14px",
              borderRadius: "14px",
              border: "2px solid #e6c000",
              cursor: "pointer",
              textAlign: "center",
              wordBreak: "keep-all",
              lineHeight: 1.4,
              fontFamily: "inherit",
            }}
            onClick={() => {}}
          >
            {text}
          </button>
        ))}
      </div>

      {/* 회색 도구 상자 — 코랄 알약 버튼 */}
      <div
        style={{
          background: "#f0f2f6",
          borderRadius: "16px",
          padding: "18px 20px 20px",
          marginBottom: "16px",
        }}
      >
        <p
          style={{
            fontSize: "16px",
            fontWeight: 800,
            color: "#1a1a2e",
            margin: "0 0 14px 0",
          }}
        >
          🔧 📱 유럽여행 필수 앱 활용법
        </p>
        <div
          style={{
            display: "flex",
            overflowX: "auto",
            gap: "10px",
            paddingBottom: "4px",
          }}
        >
          {tips.map((label, i) => (
            <button
              key={i}
              style={{
                background: "#FF6B6B",
                color: "white",
                fontSize: "15px",
                fontWeight: 700,
                padding: "13px 20px",
                borderRadius: "28px",
                border: "none",
                cursor: "pointer",
                whiteSpace: "nowrap",
                flexShrink: 0,
                fontFamily: "inherit",
              }}
              onClick={() => {}}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* 채팅 입력창 (하단 고정 시뮬레이션) */}
      <div
        style={{
          position: "fixed",
          bottom: 0,
          left: "50%",
          transform: "translateX(-50%)",
          width: "100%",
          maxWidth: "720px",
          background: "#ffffff",
          borderTop: "1px solid #e0e0e0",
          padding: "12px 24px",
          boxSizing: "border-box",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            background: "#f8f8f8",
            border: "1px solid #d0d0d0",
            borderRadius: "12px",
            padding: "12px 16px",
            gap: "10px",
          }}
        >
          <span style={{ color: "#888", fontSize: "15px", flex: 1 }}>
            질문을 입력하세요... (예: 프라하에서 빈까지 어떻게 가나요?)
          </span>
          <button
            style={{
              background: "#1a73e8",
              color: "white",
              border: "none",
              borderRadius: "8px",
              padding: "8px 14px",
              cursor: "pointer",
              fontSize: "16px",
            }}
          >
            ➤
          </button>
        </div>
      </div>
    </div>
  );
}
