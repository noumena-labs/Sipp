export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function parseOptionalString(
  value: unknown,
  path: string,
  createError: (message: string) => Error
): string | undefined {
  if (value == null) {
    return undefined;
  }
  if (typeof value !== 'string') {
    throw createError(`\`${path}\` must be a string if present.`);
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

export function parseRequiredString(
  value: unknown,
  path: string,
  createError: (message: string) => Error
): string {
  const parsed = parseOptionalString(value, path, createError);
  if (!parsed) {
    throw createError(`\`${path}\` is required and must be a non-empty string.`);
  }
  return parsed;
}

export function parseOptionalStringArray(
  value: unknown,
  path: string,
  createError: (message: string) => Error
): readonly string[] | undefined {
  if (value == null) {
    return undefined;
  }
  if (!Array.isArray(value) || value.some((entry) => typeof entry !== 'string')) {
    throw createError(`\`${path}\` must be an array of strings if present.`);
  }
  const trimmed = value.map((entry) => entry.trim()).filter((entry) => entry.length > 0);
  return trimmed.length > 0 ? trimmed : undefined;
}

export function parseOptionalNonNegativeInteger(
  value: unknown,
  path: string,
  createError: (message: string) => Error
): number | undefined {
  if (value == null) {
    return undefined;
  }
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || Math.floor(value) !== value) {
    throw createError(`\`${path}\` must be a non-negative integer if present.`);
  }
  return value;
}

export function parsePositiveInteger(
  value: unknown,
  path: string,
  createError: (message: string) => Error
): number | undefined {
  const parsed = parseOptionalNonNegativeInteger(value, path, createError);
  if (parsed === 0) {
    throw createError(`\`${path}\` must be greater than zero if present.`);
  }
  return parsed;
}
