import { useState, useRef, useEffect } from "react";
import { useChatWithPdf, useGetPdfStatus } from "@workspace/api-client-react";
import { Send, User, Bot, Loader2, Globe, BookOpen, ExternalLink, MapPin } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";

type WebResult = {
  title: string;
  url: string;
  content: string;
};

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
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
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
        }
      });

      setMessages([
        ...newMessages,
        {
          role: "assistant",
          content: response.answer,
          sources: response.sources,
          webResults: (response as any).webResults ?? [],
        }
      ]);
    } catch {
      setMessages([
        ...newMessages,
        { role: "assistant", content: "죄송해요, 답변 생성 중 오류가 발생했어요. 다시 시도해 주세요." }
      ]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(input);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  return (
    <div className="flex flex-col h-full bg-card rounded-2xl border shadow-sm overflow-hidden">
      {/* 메시지 영역 */}
      <div className="flex-1 overflow-y-auto p-5 space-y-6">
        {messages.length === 0 ? (
          <WelcomeScreen hasPdf={!!status?.indexed} onSuggest={setInput} />
        ) : (
          messages.map((msg, idx) => (
            <MessageBubble key={idx} msg={msg} />
          ))
        )}

        {chatMutation.isPending && <TypingIndicator />}
        <div ref={messagesEndRef} />
      </div>

      {/* 입력창 */}
      <div className="p-4 bg-card border-t">
        <form onSubmit={handleSubmit} className="relative flex items-end gap-2">
          <Textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="동유럽 여행에 대해 무엇이든 물어보세요..."
            className="pr-12 min-h-[52px] max-h-[160px] py-3 resize-none rounded-xl bg-muted/50 border-transparent focus-visible:bg-background"
            rows={1}
            disabled={chatMutation.isPending}
          />
          <Button
            type="submit"
            size="icon"
            className="absolute right-2 bottom-2 h-8 w-8 rounded-lg transition-transform hover:scale-105"
            disabled={!input.trim() || chatMutation.isPending}
          >
            <Send className="h-4 w-4" />
          </Button>
        </form>
        <p className="text-[11px] text-muted-foreground mt-2 text-center">
          여행기 내용 + 최신 인터넷 정보를 함께 검색해서 답변드려요
        </p>
      </div>
    </div>
  );
}

function WelcomeScreen({ hasPdf, onSuggest }: { hasPdf: boolean; onSuggest: (q: string) => void }) {
  return (
    <div className="h-full flex flex-col items-center justify-center py-8 text-center space-y-6">
      <div className="bg-primary/5 p-5 rounded-full">
        <MapPin className="h-10 w-10 text-primary" />
      </div>
      <div>
        <h2 className="text-xl font-bold mb-1">안녕하세요! 동유럽 여행 챗봇이에요 ✈️</h2>
        <p className="text-sm text-muted-foreground max-w-md">
          {hasPdf
            ? "여행기와 최신 인터넷 정보를 바탕으로 동유럽 여행 질문에 답해드려요."
            : "최신 인터넷 정보를 바탕으로 동유럽 여행 질문에 답해드려요."}
        </p>
      </div>
      <div className="grid grid-cols-2 gap-2 w-full max-w-lg">
        {SUGGESTIONS.map(q => (
          <button
            key={q}
            onClick={() => onSuggest(q)}
            className="text-left text-sm bg-secondary hover:bg-secondary/70 text-secondary-foreground px-4 py-3 rounded-xl transition-colors border border-border"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0 mt-1">
          <Bot className="h-4 w-4 text-primary" />
        </div>
      )}

      <div className={`max-w-[78%] ${isUser ? "order-1" : "order-2"}`}>
        <div className={`p-4 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
          isUser
            ? "bg-primary text-primary-foreground rounded-tr-sm"
            : "bg-muted text-foreground rounded-tl-sm"
        }`}>
          {msg.content}
        </div>

        {!isUser && (
          <div className="mt-2 space-y-2">
            {/* 출처 배지 */}
            <div className="flex gap-1 flex-wrap">
              {msg.sources && msg.sources.length > 0 && (
                <Badge variant="secondary" className="text-xs gap-1">
                  <BookOpen className="h-3 w-3" /> 여행기 참고
                </Badge>
              )}
              {msg.webResults && msg.webResults.length > 0 && (
                <Badge variant="secondary" className="text-xs gap-1 text-blue-600">
                  <Globe className="h-3 w-3" /> 웹 검색 {msg.webResults.length}건
                </Badge>
              )}
            </div>

            {/* 여행기 출처 */}
            {msg.sources && msg.sources.length > 0 && (
              <details className="group">
                <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground list-none flex items-center gap-1">
                  <span className="group-open:hidden">▶</span>
                  <span className="hidden group-open:inline">▼</span>
                  📖 여행기 출처 보기
                </summary>
                <div className="mt-1 space-y-1">
                  {msg.sources.map((s, i) => (
                    <div key={i} className="text-xs bg-secondary/60 p-2 rounded-lg border text-muted-foreground line-clamp-3">
                      "{s.trim()}"
                    </div>
                  ))}
                </div>
              </details>
            )}

            {/* 웹 검색 결과 */}
            {msg.webResults && msg.webResults.length > 0 && (
              <details className="group">
                <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground list-none flex items-center gap-1">
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
                      className="block text-xs bg-blue-50 dark:bg-blue-950/30 p-2.5 rounded-lg border border-blue-200 dark:border-blue-800 hover:bg-blue-100 transition-colors"
                    >
                      <div className="flex items-center gap-1 font-medium text-blue-700 dark:text-blue-300 mb-0.5">
                        <ExternalLink className="h-3 w-3 shrink-0" />
                        <span className="line-clamp-1">{r.title}</span>
                      </div>
                      <p className="text-muted-foreground line-clamp-2">{r.content}</p>
                    </a>
                  ))}
                </div>
              </details>
            )}
          </div>
        )}
      </div>

      {isUser && (
        <div className="h-8 w-8 rounded-full bg-secondary flex items-center justify-center shrink-0 order-2 mt-1">
          <User className="h-4 w-4" />
        </div>
      )}
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex gap-3 justify-start">
      <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
        <Loader2 className="h-4 w-4 text-primary animate-spin" />
      </div>
      <div className="bg-muted p-4 rounded-2xl rounded-tl-sm flex items-center gap-1.5">
        <span className="text-xs text-muted-foreground mr-1">검색 중...</span>
        {[0, 150, 300].map(d => (
          <span key={d} className="h-1.5 w-1.5 bg-foreground/40 rounded-full animate-bounce" style={{ animationDelay: `${d}ms` }} />
        ))}
      </div>
    </div>
  );
}
