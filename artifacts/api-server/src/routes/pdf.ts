import { Router, Request, Response } from "express";
import multer from "multer";
import OpenAI from "openai";
import fs from "fs";
import path from "path";

const pdfParse: (buffer: Buffer) => Promise<{ text: string; numpages: number }> =
  (globalThis as any).require("pdf-parse");

const router = Router();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 200 * 1024 * 1024 } });

function getOpenAI(): OpenAI {
  let key = process.env.OPENAI_API_KEY ?? "";
  if (key.startsWith("y") && key.slice(1).startsWith("sk-")) key = key.slice(1);
  return new OpenAI({ apiKey: key });
}

function getTavilyKey(): string {
  return process.env.TAVILY_API_KEY ?? "";
}

interface Chunk {
  text: string;
  embedding: number[];
}

interface VectorStore {
  filename: string;
  chunks: Chunk[];
}

interface WebResult {
  title: string;
  url: string;
  content: string;
}

let vectorStore: VectorStore | null = null;

// ── 서버 시작 시 data/travel.pdf 자동 로딩 ───────────────────────────────────
export async function autoLoadTravelPdf() {
  const dataDir = path.resolve(process.cwd(), "data");
  const pdfPath = path.join(dataDir, "travel.pdf");

  if (!fs.existsSync(pdfPath)) {
    console.log("[travel-pdf] data/travel.pdf 없음 — 파일을 넣으면 자동 로딩됩니다.");
    return;
  }

  try {
    console.log("[travel-pdf] data/travel.pdf 로딩 중...");
    const buffer = fs.readFileSync(pdfPath);
    await indexPdfBuffer(buffer, "travel.pdf");
    console.log(`[travel-pdf] 로딩 완료: ${vectorStore?.chunks.length}개 청크`);
  } catch (err: any) {
    console.error("[travel-pdf] 로딩 실패:", err.message);
  }
}

// ── 공통 PDF 인덱싱 함수 ─────────────────────────────────────────────────────
async function indexPdfBuffer(buffer: Buffer, filename: string) {
  const pdfData = await pdfParse(buffer);
  const text = pdfData.text.replace(/\s+/g, " ").trim();
  if (text.length < 50) throw new Error("PDF에서 텍스트를 읽을 수 없습니다.");

  const rawChunks = splitIntoChunks(text);
  const chunks: Chunk[] = [];

  const batchSize = 20;
  for (let i = 0; i < rawChunks.length; i += batchSize) {
    const batch = rawChunks.slice(i, i + batchSize);
    const embRes = await getOpenAI().embeddings.create({
      model: "text-embedding-3-small",
      input: batch.map(c => c.slice(0, 8000)),
    });
    for (let j = 0; j < batch.length; j++) {
      chunks.push({ text: batch[j], embedding: embRes.data[j].embedding });
    }
  }

  vectorStore = { filename, chunks };
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-10);
}

function splitIntoChunks(text: string, chunkSize = 800, overlap = 100): string[] {
  const sentences = text.split(/(?<=[.!?])\s+/);
  const chunks: string[] = [];
  let current = "";
  for (const sentence of sentences) {
    if ((current + sentence).length > chunkSize && current.length > 0) {
      chunks.push(current.trim());
      const words = current.split(" ");
      const overlapText = words.slice(-Math.floor(overlap / 6)).join(" ");
      current = overlapText + " " + sentence;
    } else {
      current += (current ? " " : "") + sentence;
    }
  }
  if (current.trim()) chunks.push(current.trim());
  return chunks.filter(c => c.length > 50);
}

async function getEmbedding(text: string): Promise<number[]> {
  const res = await getOpenAI().embeddings.create({
    model: "text-embedding-3-small",
    input: text.slice(0, 8000),
  });
  return res.data[0].embedding;
}

async function tavilySearch(query: string): Promise<WebResult[]> {
  const apiKey = getTavilyKey();
  if (!apiKey) return [];

  try {
    const response = await fetch("https://api.tavily.com/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        api_key: apiKey,
        query,
        search_depth: "basic",
        max_results: 5,
        include_answer: false,
      }),
    });

    if (!response.ok) return [];
    const data: any = await response.json();
    return (data.results ?? []).map((r: any) => ({
      title: r.title ?? "",
      url: r.url ?? "",
      content: (r.content ?? "").slice(0, 600),
    }));
  } catch {
    return [];
  }
}

// ── 채팅 (PDF RAG + Tavily 항상 동시 실행) ───────────────────────────────────
router.post("/chat", async (req: Request, res: Response) => {
  const { question, history = [] } = req.body;

  if (!question || typeof question !== "string") {
    res.status(400).json({ error: "Question is required" });
    return;
  }

  try {
    // PDF RAG와 Tavily 검색 병렬 실행
    const [webResults, pdfResult] = await Promise.all([
      tavilySearch(question),
      (async () => {
        if (!vectorStore) return { context: "", sources: [] };
        const queryEmbedding = await getEmbedding(question);
        const scored = vectorStore.chunks
          .map(chunk => ({ chunk, score: cosineSimilarity(queryEmbedding, chunk.embedding) }))
          .sort((a, b) => b.score - a.score)
          .slice(0, 5);
        const context = scored.map((s, i) => `[여행기 ${i + 1}] ${s.chunk.text}`).join("\n\n");
        const sources = scored.slice(0, 3).map(s =>
          s.chunk.text.slice(0, 200) + (s.chunk.text.length > 200 ? "..." : "")
        );
        return { context, sources };
      })(),
    ]);

    const { context: pdfContext, sources } = pdfResult;

    const webContext = webResults.length > 0
      ? webResults.map((r, i) => `[웹 ${i + 1}] ${r.title}\n${r.content}\n출처: ${r.url}`).join("\n\n")
      : "";

    let systemPrompt = `당신은 친절하고 유용한 동유럽 여행 전문 챗봇입니다.
여행자의 질문에 자세하고 실용적으로 답해주세요. 질문과 같은 언어로 답변하세요 (한국어 질문 → 한국어 답변).`;

    if (pdfContext) {
      systemPrompt += `\n\n## 여행기 (저자의 직접 경험)\n${pdfContext}`;
    }
    if (webContext) {
      systemPrompt += `\n\n## 최신 인터넷 정보\n${webContext}`;
    }
    if (pdfContext || webContext) {
      systemPrompt += `\n\n위 정보를 종합하여 답변하세요. 여행기의 개인 경험과 최신 웹 정보를 구분해서 설명하면 더 좋습니다.`;
    }

    const messages: OpenAI.ChatCompletionMessageParam[] = [
      { role: "system", content: systemPrompt },
      ...history.map((m: { role: string; content: string }) => ({
        role: m.role as "user" | "assistant",
        content: m.content,
      })),
      { role: "user", content: question },
    ];

    const completion = await getOpenAI().chat.completions.create({
      model: "gpt-4o-mini",
      messages,
      temperature: 0.3,
    });

    const answer = completion.choices[0].message.content ?? "";
    res.json({ answer, sources, webResults });
  } catch (err: any) {
    res.status(500).json({ error: `Failed to generate answer: ${err.message}` });
  }
});

// ── 상태 조회 ─────────────────────────────────────────────────────────────────
router.get("/status", (_req: Request, res: Response) => {
  if (!vectorStore) {
    res.json({ indexed: false });
  } else {
    res.json({
      indexed: true,
      filename: vectorStore.filename,
      chunkCount: vectorStore.chunks.length,
    });
  }
});

// ── 관리자용: PDF 교체 업로드 (숨겨진 엔드포인트) ─────────────────────────────
router.post("/admin/upload", (req: Request, res: Response) => {
  upload.single("file")(req, res, async (err: any) => {
    if (err) {
      res.status(400).json({ error: err.code === "LIMIT_FILE_SIZE" ? "파일이 너무 큽니다 (최대 200MB)" : err.message });
      return;
    }
    if (!req.file) { res.status(400).json({ error: "파일 없음" }); return; }
    if (!req.file.mimetype.includes("pdf")) { res.status(400).json({ error: "PDF만 가능" }); return; }

    try {
      const rawName = req.file.originalname;
      const filename = (() => {
        try { return Buffer.from(rawName, "latin1").toString("utf8"); } catch { return rawName; }
      })();

      // data/ 에도 저장 (서버 재시작 시 자동 로딩)
      const dataDir = path.resolve(process.cwd(), "data");
      if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });
      fs.writeFileSync(path.join(dataDir, "travel.pdf"), req.file.buffer);

      await indexPdfBuffer(req.file.buffer, filename);
      res.json({ success: true, filename, chunkCount: vectorStore?.chunks.length });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });
});

export default router;
