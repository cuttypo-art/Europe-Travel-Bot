import { useState, useRef, useEffect } from "react";
import { useChatWithPdf, useGetPdfStatus } from "@workspace/api-client-react";
import { Send, User, Bot, Loader2, Globe, BookOpen, ExternalLink } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";

const URL_REGEX = /https?:\/\/[^\s,)"\u200B\u3001\u3002\uff01\uff0c\uff0e\u0022\u0027]+/g;

function renderTextWithLinks(text: string) {
  const parts = text.split(URL_REGEX);
  const urls = text.match(URL_REGEX) ?? [];
  return parts.reduce<React.ReactNode[]>((acc, part, i) => {
    acc.push(<span key={`t${i}`}>{part}</span>);
    if (urls[i]) {
      const href = urls[i].replace(/[.,;:!?)]+$/, "");
      let label = href.replace(/^https?:\/\/(www\.)?/, "").split("/")[0];
      acc.push(
        <a
          key={`u${i}`}
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-0.5 text-blue-600 underline underline-offset-2 hover:text-blue-800 transition-colors break-all"
        >
          {label}
          <ExternalLink className="h-3 w-3 shrink-0 inline" />
        </a>
      );
    }
    return acc;
  }, []);
}

type WebResult = { title: string; url: string; content: string };
type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  webResults?: WebResult[];
};

const SUGGESTIONS = [
  "동유럽 도시 간 이동, 어떤 교통수단이 편해?",
  "잘츠부르크에서 할슈타트 당일치기 가능해?",
  "유럽 숙소 고를 때 체크리스트가 뭐야?",
  "유럽 소매치기 방지 꿀팁 알려줘",
  "유럽 크리스마스 마켓 필수 먹거리는?",
  "부다페스트 야경 명소 어디야?",
];

export function ChatInterface() {
  const { data: status } = useGetPdfStatus();
  const chatMutation = useChatWithPdf();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messages.length > 0) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const sendMessage = async (text: string) => {
    if (!text.trim() || chatMutation.isPending) return;
    setInput("");
    const newMessages: ChatMessage[] = [...messages, { role: "user", content: text.trim() }];
    setMessages(newMessages);
    try {
      const response = await chatMutation.mutateAsync({
        data: {
          question: text.trim(),
          history: messages.map(m => ({ role: m.role, content: m.content })),
        },
      });
      setMessages([
        ...newMessages,
        {
          role: "assistant",
          content: response.answer,
          sources: response.sources,
          webResults: (response as any).webResults ?? [],
        },
      ]);
    } catch {
      setMessages([
        ...newMessages,
        { role: "assistant", content: "죄송해요, 오류가 발생했어요. 다시 시도해 주세요." },
      ]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => { e.preventDefault(); sendMessage(input); };
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(input); }
  };

  return (
    <div
      className="flex flex-col h-full rounded-3xl overflow-hidden"
      style={{
        background: "#ffffff",
        border: "1px solid #e5e7eb",
        boxShadow: "0 4px 24px rgba(0,0,0,0.07)",
      }}
    >
      {/* 메시지 영역 */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 ? (
          <WelcomeScreen hasPdf={!!status?.indexed} onSuggest={q => { setInput(q); sendMessage(q); }} />
        ) : (
          messages.map((msg, idx) => <MessageBubble key={idx} msg={msg} />)
        )}
        {chatMutation.isPending && <TypingIndicator />}
        <div ref={messagesEndRef} />
      </div>

      {/* ── Pill 입력창 ── */}
      <div
        className="p-4"
        style={{ borderTop: "1px solid #e5e7eb", background: "#f9fafb" }}
      >
        <form
          onSubmit={handleSubmit}
          className="flex items-end gap-2 rounded-full px-4 py-2"
          style={{
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
          }}
        >
          <Textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="동유럽 여행에 대해 무엇이든 물어보세요..."
            className="flex-1 border-0 bg-transparent shadow-none focus-visible:ring-0 resize-none min-h-[28px] max-h-[120px] py-1.5 text-sm placeholder:text-gray-400"
            rows={1}
            disabled={chatMutation.isPending}
          />
          <button
            type="submit"
            disabled={!input.trim() || chatMutation.isPending}
            className="h-9 w-9 rounded-full flex items-center justify-center shrink-0 transition-all hover:scale-105 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed mb-0.5"
            style={{ background: "linear-gradient(135deg, #3b82f6, #6366f1)" }}
          >
            <Send className="h-4 w-4 text-white" />
          </button>
        </form>
        <p className="text-[11px] text-gray-400 mt-2 text-center">
          여행기 내용 + 최신 인터넷 정보를 함께 검색해서 답변드려요
        </p>
      </div>
    </div>
  );
}

/* ── SVG 아이콘 세트 (단색 라인 아트) ─────────────────────────────────── */
const ICON_COLOR = "#3b82f6";
const ICON_PROPS = { width: 36, height: 36, viewBox: "0 0 24 24", fill: "none", stroke: ICON_COLOR, strokeWidth: 1.4, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };

function IconSkyline() {
  return (
    <svg {...ICON_PROPS}>
      <rect x="2" y="13" width="4" height="8" rx="0.5" />
      <rect x="7" y="9" width="4" height="12" rx="0.5" />
      <rect x="12" y="5" width="5" height="16" rx="0.5" />
      <rect x="18" y="10" width="4" height="11" rx="0.5" />
      <line x1="12" y1="5" x2="14" y2="2" />
      <line x1="13" y1="2" x2="15" y2="2" />
      <line x1="1" y1="21" x2="23" y2="21" />
    </svg>
  );
}
function IconPlane() {
  return (
    <svg {...ICON_PROPS}>
      <path d="M22 2L11 13" />
      <path d="M22 2L15 22L11 13L2 9L22 2Z" />
    </svg>
  );
}
function IconMap() {
  return (
    <svg {...ICON_PROPS}>
      <polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6" />
      <line x1="8" y1="2" x2="8" y2="18" />
      <line x1="16" y1="6" x2="16" y2="22" />
    </svg>
  );
}
function IconCoffee() {
  return (
    <svg {...ICON_PROPS}>
      <path d="M18 8h1a4 4 0 0 1 0 8h-1" />
      <path d="M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z" />
      <line x1="6" y1="1" x2="6" y2="4" />
      <line x1="10" y1="1" x2="10" y2="4" />
      <line x1="14" y1="1" x2="14" y2="4" />
    </svg>
  );
}
function IconCastle() {
  return (
    <svg {...ICON_PROPS}>
      <line x1="3" y1="6" x2="3" y2="3" />
      <line x1="7" y1="6" x2="7" y2="3" />
      <line x1="3" y1="3" x2="7" y2="3" />
      <line x1="13" y1="6" x2="13" y2="3" />
      <line x1="17" y1="6" x2="17" y2="3" />
      <line x1="13" y1="3" x2="17" y2="3" />
      <rect x="2" y="6" width="7" height="15" rx="0.5" />
      <rect x="12" y="6" width="7" height="15" rx="0.5" />
      <rect x="8" y="11" width="7" height="10" rx="0.5" />
      <line x1="1" y1="21" x2="23" y2="21" />
    </svg>
  );
}

/* ── 책 표지 프레임 ─────────────────────────────────────────────────── */
function BookCoverFrame() {
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  return (
    <div
      className="relative overflow-hidden transition-transform duration-200 group-hover:scale-[1.03]"
      style={{
        width: 110,
        height: 155,
        borderRadius: 10,
        border: "1px solid #e0e7ef",
        boxShadow: "0 6px 24px rgba(0,0,0,0.13), 0 1.5px 4px rgba(0,0,0,0.08)",
        background: "linear-gradient(135deg, #e8f0fe 0%, #f8faff 100%)",
      }}
    >
      {!failed && (
        <img
          src="/pdf-chatbot/book-cover.png"
          alt="동유럽 여행 에세이 책 표지"
          className="w-full h-full object-cover"
          onLoad={() => setLoaded(true)}
          onError={() => setFailed(true)}
        />
      )}
      {(!loaded || failed) && (
        <div className="absolute inset-0 flex flex-col items-center justify-center p-3 text-center">
          <IconPlane />
          <p className="text-[10px] font-bold mt-2 leading-tight text-gray-700">
            동유럽<br />여행 에세이
          </p>
          <p className="text-[9px] text-gray-400 mt-1">교보문고 ebook</p>
        </div>
      )}
      {/* 광택 효과 */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{ background: "linear-gradient(120deg, rgba(255,255,255,0.15) 0%, transparent 55%)" }}
      />
    </div>
  );
}

/* ── Welcome / Hero 섹션 ─────────────────────────────────────────────── */
const BOOK_URL = "https://ebook-product.kyobobook.co.kr/dig/epd/ebook/E000012350958";

function WelcomeScreen({ onSuggest }: { hasPdf: boolean; onSuggest: (q: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-start pt-4 pb-2 text-center">

      {/* 1. SVG 라인아트 아이콘 세트 */}
      <div className="flex items-end justify-center gap-7 mb-4">
        <div className="opacity-50"><IconCastle /></div>
        <div className="opacity-70"><IconMap /></div>
        <div className="opacity-100"><IconPlane /></div>
        <div className="opacity-70"><IconCoffee /></div>
        <div className="opacity-50"><IconSkyline /></div>
      </div>

      {/* 2. 책 표지 이미지 프레임 */}
      <a
        href={BOOK_URL}
        target="_blank"
        rel="noopener noreferrer"
        className="block mb-4 group"
        title="교보문고에서 책 보기"
      >
        <BookCoverFrame />
        <p className="text-[11px] text-blue-500 mt-2 font-medium group-hover:underline">
          📖 교보문고에서 보기
        </p>
      </a>

      {/* 3. 타이틀 */}
      <div className="mb-4">
        <h2 className="text-2xl font-extrabold tracking-tight text-gray-800 leading-tight">
          동유럽 여행, 뭐든 물어봐!
        </h2>
        <p className="text-sm text-gray-400 mt-2 max-w-xs mx-auto leading-relaxed">
          출간 작가의 생생한 동유럽 여행 에세이와<br />
          최신 인터넷 정보를 바탕으로 답해드려요.
        </p>
      </div>

      {/* 4. 질문 카드 */}
      <div className="grid grid-cols-2 gap-3 w-full max-w-xl">
        {SUGGESTIONS.map(q => (
          <SuggestionChip key={q} label={q} onClick={() => onSuggest(q)} />
        ))}
      </div>
    </div>
  );
}

function SuggestionChip({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="group text-left text-sm text-gray-700 font-medium px-4 py-3.5 rounded-2xl transition-all duration-200 ease-out"
      style={{
        background: "#ffffff",
        border: "1px solid #e5e7eb",
        boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
      }}
      onMouseEnter={e => {
        const el = e.currentTarget;
        el.style.transform = "translateY(-3px)";
        el.style.boxShadow = "0 8px 20px rgba(59,130,246,0.15)";
        el.style.borderColor = "rgba(59,130,246,0.3)";
      }}
      onMouseLeave={e => {
        const el = e.currentTarget;
        el.style.transform = "translateY(0)";
        el.style.boxShadow = "0 2px 8px rgba(0,0,0,0.06)";
        el.style.borderColor = "#e5e7eb";
      }}
    >
      {label}
    </button>
  );
}

/* ── 메시지 버블 ─────────────────────────────────────────────────────── */
function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <div
          className="h-8 w-8 rounded-full flex items-center justify-center shrink-0 mt-1"
          style={{ background: "linear-gradient(135deg, #3b82f6, #6366f1)" }}
        >
          <Bot className="h-4 w-4 text-white" />
        </div>
      )}

      <div className={`max-w-[78%] ${isUser ? "order-1" : "order-2"}`}>
        <div
          className="p-4 text-sm leading-relaxed whitespace-pre-wrap"
          style={
            isUser
              ? {
                  background: "linear-gradient(135deg, #3b82f6, #6366f1)",
                  color: "white",
                  borderRadius: "20px 20px 4px 20px",
                  boxShadow: "0 4px 14px rgba(99,102,241,0.25)",
                }
              : {
                  background: "#ffffff",
                  color: "#1f2937",
                  borderRadius: "20px 20px 20px 4px",
                  border: "1px solid #e5e7eb",
                  boxShadow: "0 2px 10px rgba(0,0,0,0.05)",
                }
          }
        >
          {isUser ? msg.content : renderTextWithLinks(msg.content)}
        </div>

        {!isUser && (
          <div className="mt-2 space-y-1.5 ml-1">
            <div className="flex gap-1 flex-wrap">
              {msg.sources && msg.sources.length > 0 && (
                <Badge variant="secondary" className="text-[11px] gap-1 h-5">
                  <BookOpen className="h-2.5 w-2.5" /> 여행기 참고
                </Badge>
              )}
              {msg.webResults && msg.webResults.length > 0 && (
                <Badge className="text-[11px] gap-1 h-5 bg-blue-50 text-blue-600 border-blue-100 hover:bg-blue-50">
                  <Globe className="h-2.5 w-2.5" /> 웹 검색 {msg.webResults.length}건
                </Badge>
              )}
            </div>

            {msg.sources && msg.sources.length > 0 && (
              <details className="group">
                <summary className="text-[11px] text-gray-400 cursor-pointer hover:text-gray-600 list-none flex items-center gap-1">
                  <span className="group-open:hidden">▶</span>
                  <span className="hidden group-open:inline">▼</span>
                  📖 여행기 출처 보기
                </summary>
                <div className="mt-1 space-y-1">
                  {msg.sources.map((s, i) => (
                    <div key={i} className="text-[11px] bg-gray-50 p-2 rounded-xl border text-gray-500 line-clamp-3">
                      "{s.trim()}"
                    </div>
                  ))}
                </div>
              </details>
            )}

            {msg.webResults && msg.webResults.length > 0 && (
              <details className="group">
                <summary className="text-[11px] text-gray-400 cursor-pointer hover:text-gray-600 list-none flex items-center gap-1">
                  <span className="group-open:hidden">▶</span>
                  <span className="hidden group-open:inline">▼</span>
                  🌐 웹 검색 결과 보기
                </summary>
                <div className="mt-1 space-y-1.5">
                  {msg.webResults.map((r, i) => (
                    <a
                      key={i}
                      href={r.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block text-[11px] p-2.5 rounded-xl border hover:bg-blue-50 transition-colors"
                      style={{ background: "#eff6ff", borderColor: "#bfdbfe" }}
                    >
                      <div className="flex items-center gap-1 font-medium text-blue-600 mb-0.5">
                        <ExternalLink className="h-3 w-3 shrink-0" />
                        <span className="line-clamp-1">{r.title}</span>
                      </div>
                      <p className="text-gray-500 line-clamp-2">{r.content}</p>
                    </a>
                  ))}
                </div>
              </details>
            )}
          </div>
        )}
      </div>

      {isUser && (
        <div className="h-8 w-8 rounded-full bg-gray-100 flex items-center justify-center shrink-0 order-2 mt-1">
          <User className="h-4 w-4 text-gray-500" />
        </div>
      )}
    </div>
  );
}

/* ── 타이핑 인디케이터 ────────────────────────────────────────────────── */
function TypingIndicator() {
  return (
    <div className="flex gap-3 justify-start">
      <div
        className="h-8 w-8 rounded-full flex items-center justify-center shrink-0"
        style={{ background: "linear-gradient(135deg, #3b82f6, #6366f1)" }}
      >
        <Loader2 className="h-4 w-4 text-white animate-spin" />
      </div>
      <div
        className="px-5 py-3.5 flex items-center gap-1.5"
        style={{
          background: "#ffffff",
          borderRadius: "20px 20px 20px 4px",
          border: "1px solid #e5e7eb",
        }}
      >
        <span className="text-xs text-gray-400 mr-1">검색 중</span>
        {[0, 150, 300].map(d => (
          <span
            key={d}
            className="h-1.5 w-1.5 rounded-full bg-blue-400 animate-bounce"
            style={{ animationDelay: `${d}ms` }}
          />
        ))}
      </div>
    </div>
  );
}
