import { INTERACTION_RADIUS } from './reducer.js';
import type { AgentPerception, DecisionContext, DecisionOption, PerceivedObject } from './types.js';

export function buildDecisionContext(perception: AgentPerception): DecisionContext {
  const options: DecisionOption[] = [];
  const lines: string[] = [];
  const reachableObjects = perception.nearbyObjects
    .filter((object) => object.distance <= INTERACTION_RADIUS)
    .sort(compareObjectsForPriority);
  const visibleObjects = perception.nearbyObjects
    .filter((object) => object.distance > INTERACTION_RADIUS)
    .sort(compareObjectsForPriority);
  const visibleAgents = perception.nearbyAgents;
  let hasImmediateAction = false;

  lines.push(`Tick ${perception.tick}.`);
  lines.push(`You are ${describeHolding(perception)}.`);
  if (perception.directorNote) {
    lines.push(`Scene note: ${perception.directorNote}`);
  }

  if (reachableObjects.length > 0) {
    lines.push('Within reach:');
    for (const object of reachableObjects) {
      lines.push(`- ${describeObject(object)}`);
      for (const affordance of object.affordances) {
        options.push({
          label: affordance.label,
          goal: { kind: 'object_action', objectId: object.id, affordance, label: affordance.label },
        });
        hasImmediateAction = true;
      }
    }
  }

  if (perception.self.holding) {
    options.push({
      label: `drop the ${perception.self.holding}`,
      goal: { kind: 'drop', label: `drop the ${perception.self.holding}` },
    });
    hasImmediateAction = true;
  }

  if (!hasImmediateAction && visibleObjects.length > 0) {
    lines.push('Visible points of interest:');
    for (const object of visibleObjects.slice(0, 3)) {
      const label = `go to the ${object.label}`;
      lines.push(`- ${object.label} (${qualitativeDistance(object.distance)})`);
      options.push({
        label,
        goal: { kind: 'go_to_object', objectId: object.id, label },
      });
    }
  }

  if (!hasImmediateAction && visibleAgents.length > 0) {
    lines.push('Visible agents:');
    for (const agent of visibleAgents.slice(0, 2)) {
      const label = `approach ${agent.name}`;
      lines.push(`- ${agent.name} (${qualitativeDistance(agent.distance)}), ${describeOtherAgent(agent)}`);
      options.push({
        label,
        goal: { kind: 'go_to_agent', agentId: agent.id, label },
      });
    }
  }

  options.push({ label: 'wait', goal: { kind: 'wait', label: 'wait' } });

  dedupeOptionsInPlace(options);

  lines.push('Choose your next action from the available options only.');
  return {
    prompt: lines.join('\n'),
    options,
  };
}

function describeHolding(perception: AgentPerception): string {
  return perception.self.holding ? `currently holding ${perception.self.holding}` : 'empty-handed';
}

function describeOtherAgent(agent: AgentPerception['nearbyAgents'][number]): string {
  if (agent.holding) {
    return `holding ${agent.holding}`;
  }
  return 'empty-handed';
}

function describeObject(object: PerceivedObject): string {
  const ownership = object.heldBy ? `held by ${object.heldBy}` : 'free';
  return `${object.label} (${ownership})`;
}

function qualitativeDistance(distance: number): string {
  if (distance <= INTERACTION_RADIUS) return 'within reach';
  if (distance <= 2) return 'very close';
  if (distance <= 4.5) return 'nearby';
  return 'farther away';
}

function compareObjectsForPriority(
  a: AgentPerception['nearbyObjects'][number],
  b: AgentPerception['nearbyObjects'][number]
): number {
  const scoreDiff = getObjectPriorityScore(b) - getObjectPriorityScore(a);
  if (scoreDiff !== 0) {
    return scoreDiff;
  }
  return a.distance - b.distance;
}

function getObjectPriorityScore(object: AgentPerception['nearbyObjects'][number]): number {
  if (object.affordances.some((affordance) => affordance.kind === 'pick_up')) return 40;
  if (object.tags.includes('food')) return 35;
  if (object.tags.includes('seat')) return 30;
  if (object.tags.includes('water')) return 25;
  if (object.tags.includes('decor')) return 10;
  return 0;
}

function dedupeOptionsInPlace(options: DecisionOption[]): void {
  const seen = new Set<string>();
  let writeIndex = 0;
  for (const option of options) {
    if (seen.has(option.label)) continue;
    seen.add(option.label);
    options[writeIndex] = option;
    writeIndex += 1;
  }
  options.length = writeIndex;
}
