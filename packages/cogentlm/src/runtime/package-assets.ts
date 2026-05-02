const VITE_OPTIMIZED_DEPS_SEGMENT = '/node_modules/.vite/deps/';
const PACKAGE_ROOT = 'node_modules/@noumena-labs/cogentlm';

export function resolveOptimizedPackageAssetUrl(
  packageRelativePath: string,
  importerUrl: string
): string | null {
  let parsed: URL;

  try {
    parsed = new URL(importerUrl);
  } catch {
    return null;
  }

  const optimizedDepsIndex = parsed.pathname.indexOf(VITE_OPTIMIZED_DEPS_SEGMENT);
  if (optimizedDepsIndex < 0) {
    return null;
  }

  const basePath = parsed.pathname.slice(0, optimizedDepsIndex);
  const normalizedRelativePath = packageRelativePath.replace(/^\/+/, '');
  parsed.pathname = `${basePath}/${PACKAGE_ROOT}/${normalizedRelativePath}`;
  parsed.search = '';
  parsed.hash = '';

  return parsed.toString();
}
