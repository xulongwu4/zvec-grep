// Remote embedding endpoint canonicalization. Lives in utils so that
// engine/config.ts (which backends import) and the backends themselves can
// both use it without an import cycle or crossing the models/ boundary.

const BATCH_EMBED_CONTENTS_METHOD = ":batchEmbedContents";

/**
 * Resolve the request URL for a configured Google endpoint.
 *
 * Accepts either a complete `models/<model>:batchEmbedContents` URL or an API
 * base ending in a version segment such as `/v1beta` or `/v1` (optionally with
 * a trailing slash), which is expanded to the batch method for `model`.
 * Any other value is used verbatim so custom proxies keep working.
 */
export function resolveGoogleEmbeddingEndpoint(
  endpoint: string,
  model: string,
): string {
  const trimmed = endpoint.trim();
  if (trimmed.length === 0) {
    return trimmed;
  }
  const withoutSlash = trimmed.replace(/\/+$/, "");
  if (/\/v\d+(?:alpha|beta)?$/i.test(withoutSlash)) {
    return `${withoutSlash}/models/${model}${BATCH_EMBED_CONTENTS_METHOD}`;
  }
  return trimmed;
}

/**
 * Canonicalize a user-supplied remote endpoint for `reference` so that the
 * value persisted in the workspace snapshot, the value compared by CLI and
 * daemon pre-checks, and the value the backend requests are identical.
 *
 * Providers without endpoint expansion return the input unchanged. The
 * provider is derived from the reference prefix rather than the catalog to
 * keep this module dependency-free.
 */
export function normalizeEmbeddingEndpoint(
  reference: string,
  endpoint: string,
): string {
  const separator = reference.indexOf("/");
  if (separator <= 0) {
    return endpoint;
  }
  const provider = reference.slice(0, separator);
  const model = reference.slice(separator + 1);
  if (provider !== "google" || model.length === 0) {
    return endpoint;
  }
  return resolveGoogleEmbeddingEndpoint(endpoint, model);
}
