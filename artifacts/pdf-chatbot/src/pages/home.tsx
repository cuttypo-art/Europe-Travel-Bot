import { ChatInterface } from "@/components/ChatInterface";
import { useGetPdfStatus } from "@workspace/api-client-react";
import { BookOpen, Globe, MapPin } from "lucide-react";

export default function Home() {
  const { data: status } = useGetPdfStatus();

  return (
    <div className="min-h-[100dvh] bg-background text-foreground flex flex-col font-sans">
      {/* 헤더 */}
      <header className="border-b bg-card px-6 py-4 flex items-center justify-between sticky top-0 z-20 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 bg-primary rounded-xl flex items-center justify-center text-xl">
            ✈️
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight leading-none">동유럽 여행 챗봇</h1>
            <p className="text-xs text-muted-foreground mt-0.5">여행기 + 최신 인터넷 정보 기반 답변</p>
          </div>
        </div>

        {/* 상태 배지 */}
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <BookOpen className="h-3.5 w-3.5" />
            {status?.indexed
              ? <span className="text-green-600 font-medium">여행기 로딩됨</span>
              : <span className="text-amber-500">여행기 준비 중</span>}
          </span>
          <span className="flex items-center gap-1">
            <Globe className="h-3.5 w-3.5 text-blue-500" />
            <span className="text-blue-600 font-medium">인터넷 검색 켜짐</span>
          </span>
        </div>
      </header>

      {/* 채팅 메인 영역 */}
      <main className="flex-1 flex flex-col max-w-4xl w-full mx-auto px-4 py-6 h-[calc(100dvh-65px)]">
        <ChatInterface />
      </main>
    </div>
  );
}
