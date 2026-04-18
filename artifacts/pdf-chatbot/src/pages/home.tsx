import { useState } from "react";
import { ChatInterface } from "@/components/ChatInterface";
import { useGetPdfStatus } from "@workspace/api-client-react";
import { BookOpen, Globe, RotateCcw, Plane } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function Home() {
  const { data: status } = useGetPdfStatus();
  const [chatKey, setChatKey] = useState(0);
  const goHome = () => setChatKey(k => k + 1);

  return (
    <div
      className="min-h-[100dvh] text-foreground flex flex-col font-sans"
      style={{ background: "linear-gradient(135deg, #e8f4fd 0%, #fef9f0 50%, #f0f4ff 100%)" }}
    >
      {/* 헤더 */}
      <header
        className="px-6 py-3 flex items-center justify-between sticky top-0 z-20"
        style={{
          background: "#ffffff",
          borderBottom: "1px solid #e5e7eb",
          boxShadow: "0 1px 8px rgba(0,0,0,0.06)",
        }}
      >
        {/* 로고 */}
        <button
          onClick={goHome}
          className="flex items-center gap-3 hover:opacity-75 transition-opacity"
        >
          <div
            className="h-9 w-9 rounded-xl flex items-center justify-center"
            style={{ background: "linear-gradient(135deg, #3b82f6, #6366f1)" }}
          >
            <Plane className="h-5 w-5 text-white" strokeWidth={1.8} />
          </div>
          <div className="text-left">
            <h1 className="text-base font-bold tracking-tight leading-none text-gray-800">
              똑똑한 유럽여행 도우미
            </h1>
          </div>
        </button>

        <div className="flex items-center gap-2">
          <div className="hidden sm:flex items-center gap-1.5">
            <span
              className="flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-medium"
              style={{
                background: status?.indexed ? "rgba(22,163,74,0.1)" : "rgba(245,158,11,0.1)",
                color: status?.indexed ? "#15803d" : "#b45309",
                border: `1px solid ${status?.indexed ? "rgba(22,163,74,0.2)" : "rgba(245,158,11,0.2)"}`,
              }}
            >
              <span
                className="h-1.5 w-1.5 rounded-full"
                style={{ background: status?.indexed ? "#22c55e" : "#f59e0b" }}
              />
              <BookOpen className="h-3 w-3" />
              {status?.indexed ? "여행기 로딩됨" : "여행기 준비 중"}
            </span>

            <span
              className="flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-medium"
              style={{
                background: "rgba(59,130,246,0.1)",
                color: "#1d4ed8",
                border: "1px solid rgba(59,130,246,0.2)",
              }}
            >
              <span className="h-1.5 w-1.5 rounded-full bg-blue-400 animate-pulse" />
              <Globe className="h-3 w-3" />
              검색 활성
            </span>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={goHome}
            className="flex items-center gap-1.5 text-xs rounded-full h-8 px-3"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            새 대화
          </Button>
        </div>
      </header>

      <main className="flex-1 flex flex-col max-w-3xl w-full mx-auto px-4 py-6 h-[calc(100dvh-57px)]">
        <ChatInterface key={chatKey} />
      </main>
    </div>
  );
}
