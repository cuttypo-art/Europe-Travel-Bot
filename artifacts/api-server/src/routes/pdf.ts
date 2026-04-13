import { Router, Request, Response } from "express";
import multer from "multer";
import OpenAI from "openai";
// pdf-parse v1 exports a CJS function; use globalThis.require which is set by the esbuild banner
const pdfParse: (buffer: Buffer) => Promise<{ text: string; numpages: number }> =
  (globalThis as any).require("pdf-parse");

const router = Router();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } });

function getOpenAI(): OpenAI {
  let key = process.env.OPENAI_API_KEY ?? "";
  if (key.startsWith("y") && key.slice(1).startsWith("sk-")) key = key.slice(1);
  return new OpenAI({ apiKey: key });
}

interface Chunk {
  text: string;
  embedding: number[];
}

interface VectorStore {
  filename: string;
  chunks: Chunk[];
}

let vectorStore: VectorStore | null = null;

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

router.post("/upload", upload.single("file"), async (req: Request, res: Response) => {
  if (!req.file) {
    res.status(400).json({ error: "No file provided" });
    return;
  }
  if (!req.file.mimetype.includes("pdf")) {
    res.status(400).json({ error: "Only PDF files are supported" });
    return;
  }

  try {
    const pdfData = await pdfParse(req.file.buffer);
    const text = pdfData.text.replace(/\s+/g, " ").trim();

    if (text.length < 50) {
      res.status(400).json({ error: "PDF appears to be empty or unreadable" });
      return;
    }

    // Fix filename encoding (multer may receive Latin-1 encoded UTF-8 bytes)
    const rawName = req.file.originalname;
    const filename = (() => {
      try { return Buffer.from(rawName, "latin1").toString("utf8"); } catch { return rawName; }
    })();

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

    res.json({
      success: true,
      filename,
      chunkCount: chunks.length,
      message: `PDF indexed successfully with ${chunks.length} chunks`,
    });
  } catch (err: any) {
    res.status(500).json({ error: `Failed to process PDF: ${err.message}` });
  }
});

router.post("/chat", async (req: Request, res: Response) => {
  const { question, history = [] } = req.body;

  if (!question || typeof question !== "string") {
    res.status(400).json({ error: "Question is required" });
    return;
  }
  if (!vectorStore) {
    res.status(400).json({ error: "No PDF indexed. Please upload a PDF first." });
    return;
  }

  try {
    const queryEmbedding = await getEmbedding(question);

    const scored = vectorStore.chunks
      .map(chunk => ({ chunk, score: cosineSimilarity(queryEmbedding, chunk.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);

    const context = scored.map((s, i) => `[${i + 1}] ${s.chunk.text}`).join("\n\n");
    const sources = scored.slice(0, 3).map(s => s.chunk.text.slice(0, 200) + (s.chunk.text.length > 200 ? "..." : ""));

    const messages: OpenAI.ChatCompletionMessageParam[] = [
      {
        role: "system",
        content: `You are a helpful assistant that answers questions based on the provided document context. 
Answer in the same language as the question. Be concise and accurate. 
If the answer is not found in the context, say so clearly.

Document context:
${context}`,
      },
      ...history.map((m: { role: string; content: string }) => ({
        role: m.role as "user" | "assistant",
        content: m.content,
      })),
      { role: "user", content: question },
    ];

    const completion = await getOpenAI().chat.completions.create({
      model: "gpt-4o-mini",
      messages,
      temperature: 0.1,
    });

    const answer = completion.choices[0].message.content ?? "";
    res.json({ answer, sources });
  } catch (err: any) {
    res.status(500).json({ error: `Failed to generate answer: ${err.message}` });
  }
});

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

export default router;
