import assert from "node:assert/strict";
import test from "node:test";
import { getEmbeddingModelCatalogEntry } from "../../../dist/engine/models/catalog.js";
import {
  GoogleTextEmbeddingModel,
  prepareGoogleEmbeddingInput,
  resolveGoogleEmbeddingEndpoint,
} from "../../../dist/engine/models/backends/google.js";
import { normalizeEmbeddingEndpoint } from "../../../dist/engine/utils/embedding-endpoint.js";

const vector = (dimension, value = 0.25) => Array(dimension).fill(value);
const entry = getEmbeddingModelCatalogEntry("google/gemini-embedding-001");

function createDependencies() {
  let currentFetch;
  return {
    dependencies: {
      fetch(...args) {
        if (!currentFetch) {
          throw new Error("Fetch dependency is not configured");
        }
        return currentFetch(...args);
      },
    },
    async withFetch(fetch, callback) {
      const previousFetch = currentFetch;
      currentFetch = fetch;
      try {
        return await callback();
      } finally {
        currentFetch = previousFetch;
      }
    },
  };
}

function jsonResponse(body, init = {}) {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    headers: { "content-type": "application/json", ...init.headers },
  });
}

test("Google catalog entry is a remote text model with the Gemini batch endpoint", () => {
  assert.equal(entry.provider, "google");
  assert.equal(entry.backend, "google");
  assert.equal(entry.dimension, 768);
  assert.equal(
    entry.defaultEndpoint,
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents",
  );
});

test("Google endpoint resolution expands API bases and preserves full URLs", () => {
  const full =
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents";
  assert.equal(
    resolveGoogleEmbeddingEndpoint(
      "https://generativelanguage.googleapis.com/v1beta",
      "gemini-embedding-001",
    ),
    full,
  );
  assert.equal(
    resolveGoogleEmbeddingEndpoint(
      "https://generativelanguage.googleapis.com/v1beta/",
      "gemini-embedding-001",
    ),
    full,
  );
  assert.equal(
    resolveGoogleEmbeddingEndpoint("https://proxy.test/v1", "m"),
    "https://proxy.test/v1/models/m:batchEmbedContents",
  );
  assert.equal(resolveGoogleEmbeddingEndpoint(` ${full} `, "x"), full);
  assert.equal(
    resolveGoogleEmbeddingEndpoint("https://proxy.test/custom", "m"),
    "https://proxy.test/custom",
  );
});

test("Google text model builds batch requests, sets task types, and normalizes", async () => {
  const { dependencies, withFetch } = createDependencies();
  assert.throws(
    () => new GoogleTextEmbeddingModel(entry, { apiKey: " " }),
    /requires an API key/,
  );
  assert.throws(
    () => new GoogleTextEmbeddingModel(entry, { apiKey: "k", endpoint: "  " }),
    /requires an endpoint/,
  );

  const model = new GoogleTextEmbeddingModel(
    entry,
    { apiKey: "secret-value", endpoint: "https://example.test/v1beta" },
    dependencies,
  );
  assert.equal(
    model.info.endpoint,
    "https://example.test/v1beta/models/gemini-embedding-001:batchEmbedContents",
  );
  assert.deepEqual(model.info.inputKinds, ["text"]);
  assert.equal(model.info.limits.maxBatchSize, 100);
  assert.equal(model.info.limits.maxInputTokens, 2048);

  let url;
  let request;
  const result = await withFetch(
    async (requestUrl, init) => {
      url = requestUrl;
      request = init;
      return jsonResponse({
        embeddings: [{ values: vector(768, 3) }, { values: vector(768, 4) }],
      });
    },
    () =>
      model.embed(
        [
          { kind: "text", text: "first" },
          { kind: "text", text: "second" },
        ],
        { purpose: "document" },
      ),
  );

  assert.equal(
    url,
    "https://example.test/v1beta/models/gemini-embedding-001:batchEmbedContents",
  );
  assert.equal(request.method, "POST");
  assert.equal(request.headers["x-goog-api-key"], "secret-value");
  assert.equal(request.headers.Authorization, undefined);
  assert.ok(request.signal instanceof AbortSignal);

  const body = JSON.parse(request.body);
  assert.equal(body.requests.length, 2);
  // REST EmbedContentRequest shape: options live under embedContentConfig,
  // not the SDK-only `config` wrapper (which the live API rejects with 400).
  assert.deepEqual(body.requests[0], {
    model: "models/gemini-embedding-001",
    content: { parts: [{ text: "first" }] },
    embedContentConfig: {
      outputDimensionality: 768,
      taskType: "RETRIEVAL_DOCUMENT",
    },
  });
  assert.equal(body.requests[0].config, undefined);
  assert.equal(request.redirect, "error");
  assert.equal(body.requests[1].content.parts[0].text, "second");

  // Truncated Matryoshka outputs are normalized client-side.
  assert.equal(result.vectors.length, 2);
  for (const vec of result.vectors) {
    assert.equal(vec.length, 768);
    const norm = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
    assert.ok(Math.abs(norm - 1) < 1e-9, `norm=${norm}`);
  }
  assert.deepEqual(result.truncated, []);

  let queryBody;
  await withFetch(
    async (_url, init) => {
      queryBody = JSON.parse(init.body);
      return jsonResponse({ embeddings: [{ values: vector(768) }] });
    },
    () => model.embed([{ kind: "text", text: "q" }], { purpose: "query" }),
  );
  assert.equal(
    queryBody.requests[0].embedContentConfig.taskType,
    "RETRIEVAL_QUERY",
  );
});

test("gemini-embedding-2 uses prompt prefixes instead of taskType", async () => {
  assert.deepEqual(
    prepareGoogleEmbeddingInput("gemini-embedding-001", "x", "query"),
    {
      text: "x",
      taskType: "RETRIEVAL_QUERY",
    },
  );
  assert.deepEqual(
    prepareGoogleEmbeddingInput("gemini-embedding-2", "x", "query"),
    {
      text: "task: code retrieval | query: x",
    },
  );
  assert.deepEqual(
    prepareGoogleEmbeddingInput("gemini-embedding-2", "x", "document"),
    { text: "title: none | text: x" },
  );

  const { dependencies, withFetch } = createDependencies();
  const model = new GoogleTextEmbeddingModel(
    getEmbeddingModelCatalogEntry("google/gemini-embedding-2"),
    { apiKey: "k" },
    dependencies,
  );
  const bodies = [];
  const capture = async (_url, init) => {
    bodies.push(JSON.parse(init.body));
    return jsonResponse({ embeddings: [{ values: vector(768) }] });
  };
  await withFetch(capture, () =>
    model.embed([{ kind: "text", text: "doc body" }], { purpose: "document" }),
  );
  await withFetch(capture, () =>
    model.embed([{ kind: "text", text: "find it" }], { purpose: "query" }),
  );
  assert.deepEqual(bodies[0].requests[0].content.parts, [
    { text: "title: none | text: doc body" },
  ]);
  assert.deepEqual(bodies[0].requests[0].embedContentConfig, {
    outputDimensionality: 768,
  });
  assert.deepEqual(bodies[1].requests[0].content.parts, [
    { text: "task: code retrieval | query: find it" },
  ]);
  assert.equal(bodies[1].requests[0].embedContentConfig.taskType, undefined);
});

test("Google requests never follow redirects with the API key attached", async () => {
  const { createServer } = await import("node:http");
  const listen = (handler) =>
    new Promise((resolve) => {
      const server = createServer(handler);
      server.listen(0, "127.0.0.1", () =>
        resolve({ server, url: `http://127.0.0.1:${server.address().port}` }),
      );
    });
  const leaked = [];
  const b = await listen((req, res) => {
    leaked.push({ key: req.headers["x-goog-api-key"], url: req.url });
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ embeddings: [{ values: vector(768) }] }));
  });
  const a = await listen((_req, res) => {
    res.writeHead(307, {
      location: `${b.url}/v1beta/models/gemini-embedding-001:batchEmbedContents`,
    });
    res.end();
  });
  try {
    const model = new GoogleTextEmbeddingModel(entry, {
      apiKey: "secret",
      endpoint: `${a.url}/v1beta`,
    });
    await assert.rejects(
      model.embed([{ kind: "text", text: "private" }]),
      /request failed/,
    );
    assert.deepEqual(leaked, []);
  } finally {
    a.server.close();
    b.server.close();
  }
});

test("normalizeEmbeddingEndpoint canonicalizes Google bases and leaves others alone", () => {
  assert.equal(
    normalizeEmbeddingEndpoint(
      "google/gemini-embedding-2",
      "https://x.test/v1beta/",
    ),
    "https://x.test/v1beta/models/gemini-embedding-2:batchEmbedContents",
  );
  assert.equal(
    normalizeEmbeddingEndpoint("qwen/text-embedding-v4", "https://x.test/v1"),
    "https://x.test/v1",
  );
  assert.equal(
    normalizeEmbeddingEndpoint("local/x", "https://x.test/v1"),
    "https://x.test/v1",
  );
});

test("gemini-embedding-2 uses its own model resource and expanded limits", async () => {
  const { dependencies, withFetch } = createDependencies();
  const entry2 = getEmbeddingModelCatalogEntry("google/gemini-embedding-2");
  assert.equal(entry2.provider, "google");
  assert.equal(entry2.maxInputTokens, 8192);
  const model = new GoogleTextEmbeddingModel(
    entry2,
    { apiKey: "k", endpoint: "https://example.test/v1beta/" },
    dependencies,
  );
  assert.equal(
    model.info.endpoint,
    "https://example.test/v1beta/models/gemini-embedding-2:batchEmbedContents",
  );
  assert.equal(model.info.limits.maxInputTokens, 8192);
  let body;
  await withFetch(
    async (_url, init) => {
      body = JSON.parse(init.body);
      return jsonResponse({ embeddings: [{ values: vector(768) }] });
    },
    () => model.embed([{ kind: "text", text: "x" }]),
  );
  assert.equal(body.requests[0].model, "models/gemini-embedding-2");
  assert.equal(body.requests[0].embedContentConfig.outputDimensionality, 768);
});

test("Google text model validates all response shapes", async () => {
  const { dependencies, withFetch } = createDependencies();
  const model = new GoogleTextEmbeddingModel(
    entry,
    { apiKey: "secret" },
    dependencies,
  );
  const embed = () => model.embed([{ kind: "text", text: "one" }]);

  await assert.rejects(
    withFetch(async () => new Response("nope", { status: 200 }), embed),
    /not valid JSON/,
  );

  await assert.rejects(
    withFetch(
      async () =>
        jsonResponse(
          {
            error: {
              code: 429,
              message: "Quota exceeded",
              status: "RESOURCE_EXHAUSTED",
            },
          },
          { status: 429, headers: { "retry-after": "2" } },
        ),
      embed,
    ),
    (error) => {
      assert.match(error.message, /returned an error/);
      assert.equal(
        error.code,
        "ZVEC_GREP.ENGINE.MODELS.GOOGLE_TEXT_EMBEDDING_API_ERROR",
      );
      assert.match(error.context, /status=429/);
      assert.match(error.context, /retryAfterMs=2000/);
      assert.match(error.context, /providerStatus=RESOURCE_EXHAUSTED/);
      assert.match(error.context, /providerMessage=Quota exceeded/);
      return true;
    },
  );

  await assert.rejects(
    withFetch(async () => jsonResponse({}), embed),
    /did not include embeddings/,
  );

  await assert.rejects(
    withFetch(async () => jsonResponse({ embeddings: [] }), embed),
    /count did not match/,
  );

  await assert.rejects(
    withFetch(
      async () => jsonResponse({ embeddings: [{ values: "bad" }] }),
      embed,
    ),
    /invalid embedding/,
  );

  await assert.rejects(
    withFetch(
      async () => jsonResponse({ embeddings: [{ values: vector(16) }] }),
      embed,
    ),
    /unexpected dimension/,
  );

  await assert.rejects(
    withFetch(async () => {
      throw new Error("socket hang up");
    }, embed),
    (error) => {
      assert.match(error.message, /request failed/);
      assert.equal(
        error.code,
        "ZVEC_GREP.ENGINE.MODELS.GOOGLE_TEXT_EMBEDDING_REQUEST_FAILED",
      );
      return true;
    },
  );
});
