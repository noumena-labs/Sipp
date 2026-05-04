export function sliceUnstreamedSuffix(
  streamedOutputText: string,
  finalOutputText: string
): string {
  if (streamedOutputText.length === 0) {
    return finalOutputText;
  }
  if (!finalOutputText.startsWith(streamedOutputText)) {
    return '';
  }
  return finalOutputText.slice(streamedOutputText.length);
}
