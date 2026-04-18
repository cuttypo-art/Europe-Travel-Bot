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

const vectorStores: { travel: VectorStore | null; googlemap: VectorStore | null } = {
  travel: null,
  googlemap: null,
};

// ── 서버 시작 시 PDF 자동 로딩 ───────────────────────────────────────────────
export async function autoLoadTravelPdf() {
  const dataDir = path.resolve(process.cwd(), "data");

  // travel.pdf
  const travelPath = path.join(dataDir, "travel.pdf");
  if (fs.existsSync(travelPath)) {
    try {
      console.log("[pdf] travel.pdf 로딩 중...");
      const buf = fs.readFileSync(travelPath);
      vectorStores.travel = await buildVectorStore(buf, "travel.pdf");
      console.log(`[pdf] travel.pdf 완료: ${vectorStores.travel.chunks.length}청크`);
    } catch (err: any) {
      console.error("[pdf] travel.pdf 실패:", err.message);
    }
  }

  // googlemap.pdf
  const gmapPath = path.join(dataDir, "googlemap.pdf");
  if (fs.existsSync(gmapPath)) {
    try {
      console.log("[pdf] googlemap.pdf 로딩 중...");
      const buf = fs.readFileSync(gmapPath);
      vectorStores.googlemap = await buildVectorStore(buf, "googlemap.pdf");
      console.log(`[pdf] googlemap.pdf 완료: ${vectorStores.googlemap.chunks.length}청크`);
    } catch (err: any) {
      console.error("[pdf] googlemap.pdf 실패:", err.message);
    }
  }
}

// ── PDF → VectorStore 빌드 ───────────────────────────────────────────────────
async function buildVectorStore(buffer: Buffer, filename: string): Promise<VectorStore> {
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
  return { filename, chunks };
}

// ── 관리자 업로드용 (travel.pdf 교체) ────────────────────────────────────────
async function indexPdfBuffer(buffer: Buffer, filename: string) {
  vectorStores.travel = await buildVectorStore(buffer, filename);
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

// ── 구글맵 관련 질문 감지 ────────────────────────────────────────────────────
function isGoogleMapQuestion(q: string): boolean {
  return /구글\s*맵|구글\s*지도|google\s*map|길찾기|내비|나침반|스트리트뷰|위성\s*지도|장소\s*검색|즐겨찾기\s*저장|오프라인\s*지도|라이브\s*뷰|live\s*view|위치\s*공유|리뷰|평점|후기/i.test(q);
}

// ── 구글맵 슬라이드 이미지 URL 반환 ─────────────────────────────────────────
function pageUrl(n: number): string {
  return `/api/gmap-slides/page-${String(n).padStart(3, "0")}.png`;
}

interface TopicImages { keywords: RegExp; pages: number[] }

const GMAP_TOPIC_IMAGES: TopicImages[] = [
  { keywords: /설치|다운로드|플레이\s*스토어|install/i,               pages: [3, 4, 5, 6] },
  { keywords: /화면\s*구성|인터페이스|검색창|나침반|현재\s*위치|기본\s*화면/i, pages: [15, 16, 17, 18] },
  { keywords: /지도\s*유형|위성|지형|3d|스트리트뷰/i,                 pages: [19, 20, 21, 22, 24, 26] },
  { keywords: /맛집|음식점|레스토랑|카페|브런치/i,                     pages: [31, 32, 33, 34, 35] },
  { keywords: /한식|한국\s*식당|한국\s*음식/i,                         pages: [36, 37, 38, 39, 40] },
  { keywords: /리뷰|평점|후기|별점/i,                                  pages: [43, 44, 45] },
  { keywords: /예약|reservation|booking/i,                            pages: [79, 80, 81, 82] },
  { keywords: /길찾기|경로|내비|navigation|도보|대중교통|자전거/i,      pages: [48, 51, 52, 53, 55, 56] },
  { keywords: /라이브\s*뷰|live\s*view|증강현실|ar/i,                  pages: [83, 84, 85, 86, 87, 88, 89] },
  { keywords: /즐겨찾기|저장|목록|나만의\s*여행|travel\s*map/i,        pages: [93, 94, 95, 96] },
  { keywords: /위치\s*공유|현재\s*위치\s*보내|동선/i,                  pages: [129, 130, 131, 132, 133, 134] },
  { keywords: /지도\s*비교|citymapper|maps\.me/i,                      pages: [138] },
];

function getRelevantGmapImages(question: string, max = 4): string[] {
  const matched: number[] = [];
  for (const topic of GMAP_TOPIC_IMAGES) {
    if (topic.keywords.test(question)) {
      matched.push(...topic.pages);
    }
  }
  // 중복 제거 후 최대 max개 반환
  const unique = [...new Set(matched)].slice(0, max);
  // 매칭 없으면 기본 인터페이스 이미지 보여주기
  if (unique.length === 0) return [pageUrl(1), pageUrl(7), pageUrl(15)];
  return unique.map(pageUrl);
}

// ── 두 VectorStore 통합 RAG 검색 ─────────────────────────────────────────────
async function searchAllStores(question: string, queryEmbedding: number[]): Promise<{ context: string; sources: string[] }> {
  const results: Array<{ text: string; score: number; label: string }> = [];

  if (vectorStores.travel) {
    vectorStores.travel.chunks.forEach(chunk => {
      results.push({ text: chunk.text, score: cosineSimilarity(queryEmbedding, chunk.embedding), label: "여행기" });
    });
  }
  if (vectorStores.googlemap) {
    vectorStores.googlemap.chunks.forEach(chunk => {
      results.push({ text: chunk.text, score: cosineSimilarity(queryEmbedding, chunk.embedding), label: "구글맵 가이드" });
    });
  }

  const top = results.sort((a, b) => b.score - a.score).slice(0, 6);
  const context = top.map((r, i) => `[${r.label} ${i + 1}] ${r.text}`).join("\n\n");
  const sources = top.slice(0, 3).map(r => r.text.slice(0, 200) + (r.text.length > 200 ? "..." : ""));
  return { context, sources };
}

// ── 시스템 프롬프트 생성 ─────────────────────────────────────────────────────
function buildSystemPrompt(pdfContext: string, webContext: string, isGmapQ: boolean): string {
  let prompt = `당신은 친절하고 유용한 동유럽 여행 및 구글맵 사용법 전문 챗봇입니다.
질문에 자세하고 실용적으로 답해주세요. 질문과 같은 언어로 답변하세요 (한국어 질문 → 한국어 답변).

## 관련 사이트 안내 지침
답변 내용에 따라 아래 사이트를 자연스럽게 언급하고, URL을 그대로 포함해 주세요 (마크다운 없이 URL 그대로):

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

관련 사이트가 있을 때만 언급하세요. URL은 마크다운 형식([텍스트](url)) 없이 평문 URL로 작성하세요.

## 동유럽 도시 간 이동 교통수단 질문 필수 안내
사용자가 동유럽 도시 간 이동, 교통수단, 버스, 기차, 셔틀 등을 묻는 경우 반드시 아래 내용을 포함하세요:
- **CK Shuttle** (https://www.ckshuttle.cz): 프라하·빈·부다페스트·잘츠부르크 등 동유럽 주요 도시를 연결하는 셔틀버스 서비스. 예약이 쉽고 가격이 합리적이며 한국 여행자들에게 특히 인기 있습니다. 도시 간 이동 시 첫 번째로 추천하세요.

## 잘츠부르크 → 할슈타트 이동 질문 필수 안내
사용자가 잘츠부르크에서 할슈타트 가는 방법, 직행버스, 교통편 등을 묻는 경우 반드시 아래 내용을 포함하세요:
- 잘츠부르크에서 할슈타트까지 직행버스는 없으며, 보통 기차+페리 또는 버스+페리 조합으로 이동합니다.
- 가장 편리한 방법은 **GetYourGuide**에서 잘츠부르크 출발 할슈타트 당일치기 투어를 예약하는 것입니다: https://www.getyourguide.com (검색어: "Hallstatt day trip from Salzburg")
- GetYourGuide 투어는 픽업·드롭오프 포함, 한국어 후기도 많아 시니어 여행자에게 특히 추천합니다.`;

  if (isGmapQ) {
    prompt += `\n\n## 구글맵 관련 질문 지침
구글맵 사용법을 물어보는 경우, 아래 유튜브 영상을 반드시 참고 자료로 언급해 주세요:
- 추천 영상: https://youtu.be/oZ57SmPTh9s
또한, 최신 구글맵 관련 유튜브 영상이 웹 검색 결과에 있으면 함께 안내해 주세요.
단계별로 쉽게 설명하고, 해외 여행자에게 특히 유용한 기능을 강조하세요.`;
  }

  if (pdfContext) {
    prompt += `\n\n## 참고 자료 (여행기 & 구글맵 가이드)\n${pdfContext}`;
  }
  if (webContext) {
    prompt += `\n\n## 최신 인터넷 정보\n${webContext}`;
  }
  if (pdfContext || webContext) {
    prompt += `\n\n위 정보를 종합하여 답변하세요. 출처(여행기 경험 / 구글맵 가이드 / 웹 정보)를 구분해서 설명하면 더 좋습니다.`;
  }

  return prompt;
}

// ── 채팅 (PDF RAG + Tavily 항상 동시 실행) ───────────────────────────────────
router.post("/chat", async (req: Request, res: Response) => {
  const { question, history = [] } = req.body;

  if (!question || typeof question !== "string") {
    res.status(400).json({ error: "Question is required" });
    return;
  }

  try {
    const isGmapQ = isGoogleMapQuestion(question);

    // 구글맵 질문은 웹 검색 생략 → 슬라이드 이미지 + PDF로 바로 답변 (속도 향상)
    const [webResults, queryEmbedding] = await Promise.all([
      isGmapQ ? Promise.resolve([]) : tavilySearch(question),
      getEmbedding(question),
    ]);

    const { context: pdfContext, sources } = await searchAllStores(question, queryEmbedding);

    const webContext = webResults.length > 0
      ? webResults.map((r, i) => `[웹 ${i + 1}] ${r.title}\n${r.content}\n출처: ${r.url}`).join("\n\n")
      : "";

    const systemPrompt = buildSystemPrompt(pdfContext, webContext, isGmapQ);

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
    const images = isGmapQ ? getRelevantGmapImages(question) : [];
    res.json({ answer, sources, webResults, images });
  } catch (err: any) {
    res.status(500).json({ error: `Failed to generate answer: ${err.message}` });
  }
});

// ── 채팅 스트리밍 (SSE) ───────────────────────────────────────────────────────
router.post("/chat/stream", async (req: Request, res: Response) => {
  const { question, history = [] } = req.body;

  if (!question || typeof question !== "string") {
    res.status(400).json({ error: "Question is required" });
    return;
  }

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  const send = (event: string, data: unknown) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  try {
    const isGmapQ = isGoogleMapQuestion(question);

    // 구글맵 질문은 웹 검색 생략 → 속도 향상
    const [webResults, queryEmbedding] = await Promise.all([
      isGmapQ ? Promise.resolve([]) : tavilySearch(question),
      getEmbedding(question),
    ]);

    const { context: pdfContext, sources } = await searchAllStores(question, queryEmbedding);

    const webContext = webResults.length > 0
      ? webResults.map((r, i) => `[웹 ${i + 1}] ${r.title}\n${r.content}\n출처: ${r.url}`).join("\n\n")
      : "";

    // 소스 먼저 전송
    send("sources", { sources, webResults });

    const systemPrompt = buildSystemPrompt(pdfContext, webContext, isGmapQ);

    const messages: OpenAI.ChatCompletionMessageParam[] = [
      { role: "system", content: systemPrompt },
      ...history.map((m: { role: string; content: string }) => ({
        role: m.role as "user" | "assistant",
        content: m.content,
      })),
      { role: "user", content: question },
    ];

    const stream = await getOpenAI().chat.completions.create({
      model: "gpt-4o-mini",
      messages,
      temperature: 0.3,
      stream: true,
    });

    for await (const chunk of stream) {
      const token = chunk.choices[0]?.delta?.content ?? "";
      if (token) send("token", { token });
    }

    send("done", {});
    res.end();
  } catch (err: any) {
    send("error", { message: err.message });
    res.end();
  }
});

// ── 상태 조회 ─────────────────────────────────────────────────────────────────
router.get("/status", (_req: Request, res: Response) => {
  const indexed = !!(vectorStores.travel || vectorStores.googlemap);
  res.json({
    indexed,
    travel: vectorStores.travel ? { filename: vectorStores.travel.filename, chunkCount: vectorStores.travel.chunks.length } : null,
    googlemap: vectorStores.googlemap ? { filename: vectorStores.googlemap.filename, chunkCount: vectorStores.googlemap.chunks.length } : null,
  });
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
