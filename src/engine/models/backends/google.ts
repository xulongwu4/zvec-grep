import { globalConfigPath } from "../../config.js";
import { EngineError, type EngineErrorCode } from "../../errors.js";
import type { Content, TextContent } from "../../types.js";
import {
  BaseEmbeddingModel,
  type CreateEmbeddingModelOptions,
  type EmbeddingModelInfo,
  type EmbeddingResult,
  type NormalizedEmbeddingOptions,
} from "../embeddings.js";
import type { GoogleTextEmbeddingCatalogEntry } from "../catalog.js";
import { traceHeaders } from "../../../observability/trace-context.js";
import { resolveGoogleEmbeddingEndpoint } from "../../utils/embedding-endpoint.js";

// -----------------------------------------------------------------------------
// Google Gemini Embedding API (generativelanguage.googleapis.com)
//
// Uses the batchEmbedContents method so one request carries a whole batch:
// https://ai.google.dev/api/embeddings#method:-models.batchembedcontents
// -----------------------------------------------------------------------------

const DEFAULT_REMOTE_EMBEDDING_TIMEOUT_MS = 60_000;

/** Output size at which Gemini embedding models already return unit vectors. */
const GOOGLE_NATIVE_NORMALIZED_DIMENSION = 3072;

type GoogleDependencies = {
  fetch: typeof globalThis.fetch;
};

const defaultDependencies: GoogleDependencies = {
  fetch: (...args) => globalThis.fetch(...args),
};

type GoogleTaskType = "RETRIEVAL_QUERY" | "RETRIEVAL_DOCUMENT";

/**
 * Build the text and optional `taskType` for one embedding request.
 *
 * `gemini-embedding-001` distinguishes queries from documents with the
 * `taskType` parameter. `gemini-embedding-2` does not accept `taskType`;
 * Google documents prompt prefixes for asymmetric retrieval instead.
 * See https://ai.google.dev/gemini-api/docs/embeddings#migration-from-gemini-embedding-001
 */
export function prepareGoogleEmbeddingInput(
  model: string,
  text: string,
  purpose: NormalizedEmbeddingOptions["purpose"],
): { text: string; taskType?: GoogleTaskType } {
  if (usesTaskInstructionPrefixes(model)) {
    return {
      text:
        purpose === "query"
          ? `task: code retrieval | query: ${text}`
          : `title: none | text: ${text}`,
    };
  }
  return { text, taskType: taskTypeForPurpose(purpose) };
}

function usesTaskInstructionPrefixes(model: string): boolean {
  return model !== "gemini-embedding-001";
}

export { resolveGoogleEmbeddingEndpoint };

export class GoogleTextEmbeddingModel extends BaseEmbeddingModel {
  readonly info: EmbeddingModelInfo;

  private readonly entry: GoogleTextEmbeddingCatalogEntry;
  private readonly apiKey: string;
  private readonly endpoint: string;
  private readonly displayName: string;
  private readonly dependencies: GoogleDependencies;

  constructor(
    entry: GoogleTextEmbeddingCatalogEntry,
    options: CreateEmbeddingModelOptions,
    dependencies: Partial<GoogleDependencies> = {},
  ) {
    super();

    this.entry = entry;
    this.displayName = `Google ${entry.model}`;
    this.dependencies = { ...defaultDependencies, ...dependencies };
    const endpoint =
      options.endpoint === undefined
        ? entry.defaultEndpoint
        : resolveGoogleEmbeddingEndpoint(options.endpoint, entry.model);
    this.info = {
      reference: entry.reference,
      provider: entry.provider,
      name: entry.model,
      dimension: entry.dimension,
      metric: entry.metric,
      endpoint,
      inputKinds: ["text"],
      limits: {
        maxBatchSize: entry.maxBatchSize,
        maxInputTokens: entry.maxInputTokens,
      },
    };

    const apiKey = options.apiKey?.trim() ?? "";
    if (apiKey.length === 0) {
      throw new EngineError(`${this.displayName} model requires an API key`, {
        code: this.errorCode("MISSING_API_KEY"),
        context: `model=${this.info.reference}\nhint=Pass --api-key, set ZVEC_GREP_API_KEY (or GEMINI_API_KEY / GOOGLE_API_KEY), or configure providers.google.apiKey in ${globalConfigPath()}.`,
      });
    }

    if (endpoint.length === 0) {
      throw new EngineError(`${this.displayName} model requires an endpoint`, {
        code: this.errorCode("MISSING_ENDPOINT"),
        context: `model=${this.info.reference}`,
      });
    }

    this.apiKey = apiKey;
    this.endpoint = endpoint;
  }

  protected async doEmbed(
    contents: readonly Content[],
    options: NormalizedEmbeddingOptions,
  ): Promise<EmbeddingResult> {
    const texts = (contents as readonly TextContent[]).map(
      (content) => content.text,
    );
    const modelResource = `models/${this.entry.model}`;
    const requests = texts.map((text) => {
      const input = prepareGoogleEmbeddingInput(
        this.entry.model,
        text,
        options.purpose,
      );
      return {
        model: modelResource,
        content: { parts: [{ text: input.text }] },
        embedContentConfig: {
          outputDimensionality: this.info.dimension,
          ...(input.taskType !== undefined ? { taskType: input.taskType } : {}),
        },
      };
    });

    let response: Response;
    const signal = remoteEmbeddingSignal(options.signal);

    try {
      response = await this.dependencies.fetch(this.endpoint, {
        method: "POST",
        headers: {
          ...traceHeaders(),
          "x-goog-api-key": this.apiKey,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ requests }),
        // The API key travels in a custom header that fetch would forward
        // across origins; never follow redirects away from the authorized
        // endpoint with credentials and workspace content attached.
        redirect: "error",
        signal,
      });
    } catch (cause) {
      throwIfEmbeddingCancelled(options.signal);
      throw new EngineError(`${this.displayName} request failed`, {
        code: this.errorCode("REQUEST_FAILED"),
        context: `model=${this.info.reference} endpoint=${this.endpoint} timeoutMs=${DEFAULT_REMOTE_EMBEDDING_TIMEOUT_MS}`,
        cause,
      });
    }

    let body: unknown;

    try {
      body = await response.json();
    } catch (cause) {
      throw new EngineError(`${this.displayName} response was not valid JSON`, {
        code: this.errorCode("INVALID_JSON"),
        context: `model=${this.info.reference} status=${response.status}`,
        cause,
      });
    }

    if (!response.ok) {
      const error = readProviderError(body);

      throw new EngineError(`${this.displayName} request returned an error`, {
        code: this.errorCode("API_ERROR"),
        context: providerErrorContext(this.entry.model, response, error),
      });
    }

    if (!isRecord(body) || !Array.isArray(body.embeddings)) {
      throw new EngineError(
        `${this.displayName} response did not include embeddings`,
        {
          code: this.errorCode("MISSING_EMBEDDINGS"),
          context: `model=${this.info.reference}`,
        },
      );
    }

    if (body.embeddings.length !== texts.length) {
      throw new EngineError(
        `${this.displayName} response embedding count did not match input`,
        {
          code: this.errorCode("COUNT_MISMATCH"),
          context: `model=${this.info.reference} inputCount=${texts.length} outputCount=${body.embeddings.length}`,
        },
      );
    }

    const normalize =
      this.info.dimension !== GOOGLE_NATIVE_NORMALIZED_DIMENSION;
    const vectors = new Array<number[]>(texts.length);

    // batchEmbedContents returns embeddings in request order.
    for (const [index, item] of body.embeddings.entries()) {
      if (
        !isRecord(item) ||
        !Array.isArray(item.values) ||
        !item.values.every((value) => typeof value === "number")
      ) {
        throw new EngineError(
          `${this.displayName} response included an invalid embedding`,
          {
            code: this.errorCode("INVALID_VECTOR"),
            context: `model=${this.info.reference} index=${index}`,
          },
        );
      }

      const values = item.values as number[];
      if (values.length !== this.info.dimension) {
        throw new EngineError(
          `${this.displayName} response embedding had an unexpected dimension`,
          {
            code: this.errorCode("DIMENSION_MISMATCH"),
            context: `model=${this.info.reference} index=${index} expected=${this.info.dimension} actual=${values.length}`,
          },
        );
      }

      vectors[index] = normalize ? normalizeVector(values) : values;
    }

    return { vectors, truncated: [] };
  }

  private errorCode(suffix: string): EngineErrorCode {
    return `ZVEC_GREP.ENGINE.MODELS.GOOGLE_TEXT_EMBEDDING_${suffix}`;
  }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

function taskTypeForPurpose(
  purpose: NormalizedEmbeddingOptions["purpose"],
): GoogleTaskType {
  return purpose === "query" ? "RETRIEVAL_QUERY" : "RETRIEVAL_DOCUMENT";
}

function normalizeVector(values: readonly number[]): number[] {
  let sumOfSquares = 0;
  for (const value of values) {
    sumOfSquares += value * value;
  }
  if (sumOfSquares === 0) {
    return [...values];
  }
  const inverseNorm = 1 / Math.sqrt(sumOfSquares);
  return values.map((value) => value * inverseNorm);
}

function providerErrorContext(
  model: string,
  response: Response,
  error: { code: string; status: string; message: string },
): string {
  const retryAfter = retryAfterHeaderMs(response.headers.get("retry-after"));
  const retryAfterDetail =
    typeof retryAfter === "number" ? ` retryAfterMs=${retryAfter}` : "";

  return `model=${model} status=${response.status}${retryAfterDetail} providerCode=${error.code} providerStatus=${error.status} providerMessage=${error.message}`;
}

function retryAfterHeaderMs(value: string | null): number | undefined {
  if (!value) {
    return undefined;
  }

  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) {
    return Math.round(seconds * 1000);
  }

  const dateMs = Date.parse(value);
  if (Number.isFinite(dateMs)) {
    return Math.max(0, dateMs - Date.now());
  }

  return undefined;
}

/**
 * Google API errors use the `google.rpc.Status` shape:
 * `{ error: { code: 400, message: "...", status: "INVALID_ARGUMENT" } }`.
 */
function readProviderError(body: unknown): {
  code: string;
  status: string;
  message: string;
} {
  if (!isRecord(body) || !isRecord(body.error)) {
    return { code: "unknown", status: "unknown", message: "unknown" };
  }

  const { code, status, message } = body.error;
  return {
    code:
      typeof code === "number" || typeof code === "string"
        ? String(code)
        : "unknown",
    status: typeof status === "string" ? status : "unknown",
    message: typeof message === "string" ? message : "unknown",
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function remoteEmbeddingSignal(signal: AbortSignal | undefined): AbortSignal {
  const timeout = AbortSignal.timeout(DEFAULT_REMOTE_EMBEDDING_TIMEOUT_MS);
  return signal ? AbortSignal.any([signal, timeout]) : timeout;
}

function throwIfEmbeddingCancelled(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) {
    return;
  }
  throw signal.reason instanceof Error
    ? signal.reason
    : new Error("Embedding request was cancelled.");
}
