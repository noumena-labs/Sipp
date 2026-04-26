import type { CharacterConfig } from '@noumena-labs/cogent-engine/character';
import {
  getRequiredClipActions,
  getUnsupportedAvatarActions,
} from '../actions';
import type { ClipActionName } from '../actions/mixamo';

export interface AvatarRenderAssets {
  readonly characterUrl: string;
  readonly baseUrl: string;
  readonly vrmUrl: string;
  readonly idleUrl: string;
}

export function resolveAvatarRenderAssets(characterUrl: string): AvatarRenderAssets {
  const resolvedCharacterUrl = new URL(characterUrl, window.location.href).toString();
  const baseUrl = new URL('./', resolvedCharacterUrl).toString();
  return {
    characterUrl: resolvedCharacterUrl,
    baseUrl,
    vrmUrl: new URL('avatar.vrm', baseUrl).toString(),
    idleUrl: new URL('animations/idle.fbx', baseUrl).toString(),
  };
}

export function resolveActionClipUrl(
  renderAssets: AvatarRenderAssets,
  actionName: ClipActionName
): string {
  return new URL(`animations/${actionName}.fbx`, renderAssets.baseUrl).toString();
}

export async function validateAvatarRenderAssets(
  config: CharacterConfig,
  renderAssets: AvatarRenderAssets
): Promise<void> {
  const actionNames = config.actions.map((action) => action.id);
  const unsupportedActions = getUnsupportedAvatarActions(actionNames);
  if (unsupportedActions.length > 0) {
    throw new Error(`Unsupported avatar actions: ${unsupportedActions.join(', ')}`);
  }

  await assertUrlReachable(renderAssets.vrmUrl, 'avatar.vrm');
  await assertUrlReachable(renderAssets.idleUrl, 'animations/idle.fbx');

  const clipActions = getRequiredClipActions(actionNames);
  await Promise.all(
    clipActions.map((actionName) =>
      assertUrlReachable(
        resolveActionClipUrl(renderAssets, actionName),
        `animations/${actionName}.fbx`
      )
    )
  );
}

async function assertUrlReachable(url: string, label: string): Promise<void> {
  const response = await fetchReachability(url);
  if (response.ok) {
    return;
  }
  throw new Error(`Missing required render asset ${label}: ${response.status} ${url}`);
}

async function fetchReachability(url: string): Promise<Response> {
  try {
    const response = await fetch(url, { method: 'HEAD' });
    if (response.ok || (response.status !== 405 && response.status !== 501)) {
      return response;
    }
  } catch {
    // Fall back to GET below.
  }

  return fetch(url, { method: 'GET' });
}
