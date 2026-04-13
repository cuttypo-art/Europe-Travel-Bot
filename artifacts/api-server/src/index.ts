import app from "./app";
import { logger } from "./lib/logger";

// Fix OpenAI API key if it has a leading 'y' prefix (common input artifact)
const rawKey = process.env.OPENAI_API_KEY ?? "";
if (rawKey.startsWith("y") && rawKey.slice(1).startsWith("sk-")) {
  process.env.OPENAI_API_KEY = rawKey.slice(1);
}

const rawPort = process.env["PORT"];

if (!rawPort) {
  throw new Error(
    "PORT environment variable is required but was not provided.",
  );
}

const port = Number(rawPort);

if (Number.isNaN(port) || port <= 0) {
  throw new Error(`Invalid PORT value: "${rawPort}"`);
}

app.listen(port, (err) => {
  if (err) {
    logger.error({ err }, "Error listening on port");
    process.exit(1);
  }

  logger.info({ port }, "Server listening");
});
