import { GOAL_RADIUS, INTERACTION_RADIUS, SABOTAGE_RADIUS } from './reducer.js';
import type { AgentPerception, DecisionContext, DecisionOption, PerceivedAgent, PerceivedObject } from './types.js';

export function buildDecisionContext(perception: AgentPerception): DecisionContext {
  const options: DecisionOption[] = [];
  const lines: string[] = [];
  const banana = findObject(perception, perception.game.bananaObjectId);
  const goal = findObject(perception, perception.game.goalObjectId);
  const carrier = perception.nearbyAgents.find((agent) => agent.holding === perception.game.bananaObjectId);

  lines.push(`Tick ${perception.tick}. Banana Dash score: ${formatScore(perception)}.`);
  lines.push(`You are ${describeHolding(perception)}.`);
  if (perception.directorNote) {
    lines.push(`Director says: ${perception.directorNote}`);
  }

  if (perception.self.holding === perception.game.bananaObjectId) {
    addCarrierOptions(options, lines, goal);
  } else {
    addNonCarrierOptions(options, lines, banana, goal, carrier);
  }

  addAmbientOptions(perception, options, lines);
  options.push({ label: 'wait', goal: { kind: 'wait', label: 'wait' } });
  dedupeOptionsInPlace(options);

  lines.push('Choose your next action from the available options only. Prefer scoring, contesting the banana, or bumping the carrier over sightseeing.');
  return { prompt: lines.join('\n'), options };
}

function addCarrierOptions(
  options: DecisionOption[],
  lines: string[],
  goal: PerceivedObject | undefined
): void {
  lines.push('You have the banana. Get it to home base.');
  if (goal && goal.distance <= GOAL_RADIUS) {
    options.push({
      label: 'score at home base',
      goal: { kind: 'deliver', objectId: goal.id, label: 'score at home base' },
    });
  } else if (goal) {
    options.push({
      label: 'run to home base',
      goal: { kind: 'go_to_object', objectId: goal.id, label: 'run to home base' },
    });
  }
}

function addNonCarrierOptions(
  options: DecisionOption[],
  lines: string[],
  banana: PerceivedObject | undefined,
  goal: PerceivedObject | undefined,
  carrier: PerceivedAgent | undefined
): void {
  if (banana && !banana.heldBy) {
    lines.push(`Banana is ${qualitativeDistance(banana.distance)}.`);
    if (banana.distance <= INTERACTION_RADIUS) {
      options.push({
        label: 'grab banana',
        goal: {
          kind: 'object_action',
          objectId: banana.id,
          affordance: { kind: 'pick_up', label: 'grab banana', status: 'grabbing the banana' },
          label: 'grab banana',
        },
      });
    } else {
      options.push({
        label: 'rush banana',
        goal: { kind: 'go_to_object', objectId: banana.id, label: 'rush banana' },
      });
    }
  }

  if (carrier) {
    lines.push(`${carrier.name} has the banana and is ${qualitativeDistance(carrier.distance)}.`);
    if (carrier.distance <= SABOTAGE_RADIUS) {
      options.push({
        label: `bump ${carrier.name}`,
        goal: { kind: 'sabotage_agent', agentId: carrier.id, label: `bump ${carrier.name}` },
      });
    } else {
      options.push({
        label: `chase ${carrier.name}`,
        goal: { kind: 'go_to_agent', agentId: carrier.id, label: `chase ${carrier.name}` },
      });
    }
  }

  if (!banana && goal) {
    options.push({
      label: 'guard home base',
      goal: { kind: 'go_to_object', objectId: goal.id, label: 'guard home base' },
    });
  }
}

function addAmbientOptions(
  perception: AgentPerception,
  options: DecisionOption[],
  lines: string[]
): void {
  const visibleAgents = perception.nearbyAgents.filter((agent) => agent.holding !== perception.game.bananaObjectId);
  const visibleObjects = perception.nearbyObjects
    .filter((object) => !object.tags.includes('obstacle'))
    .filter((object) => object.id !== perception.game.bananaObjectId)
    .filter((object) => object.id !== perception.game.goalObjectId)
    .sort(compareObjectsForPriority);

  if (options.length <= 1 && visibleObjects.length > 0) {
    lines.push('Other points of interest:');
    for (const object of visibleObjects.slice(0, 2)) {
      const label = `go to the ${object.label}`;
      lines.push(`- ${object.label} (${qualitativeDistance(object.distance)})`);
      options.push({ label, goal: { kind: 'go_to_object', objectId: object.id, label } });
    }
  }

  if (options.length <= 1 && visibleAgents.length > 0) {
    const agent = visibleAgents[0]!;
    options.push({
      label: `approach ${agent.name}`,
      goal: { kind: 'go_to_agent', agentId: agent.id, label: `approach ${agent.name}` },
    });
  }
}

function findObject(perception: AgentPerception, objectId: string): PerceivedObject | undefined {
  return perception.nearbyObjects.find((object) => object.id === objectId);
}

function describeHolding(perception: AgentPerception): string {
  return perception.self.holding ? `carrying ${perception.self.holding}` : 'empty-handed';
}

function formatScore(perception: AgentPerception): string {
  return Object.entries(perception.game.score.deliveries)
    .map(([agentId, score]) => `${agentId} ${score}`)
    .join(', ');
}

function qualitativeDistance(distance: number): string {
  if (distance <= INTERACTION_RADIUS) return 'within reach';
  if (distance <= 2) return 'very close';
  if (distance <= 4.5) return 'nearby';
  return 'farther away';
}

function compareObjectsForPriority(a: PerceivedObject, b: PerceivedObject): number {
  const scoreDiff = getObjectPriorityScore(b) - getObjectPriorityScore(a);
  if (scoreDiff !== 0) return scoreDiff;
  return a.distance - b.distance;
}

function getObjectPriorityScore(object: PerceivedObject): number {
  if (object.tags.includes('score')) return 60;
  if (object.tags.includes('goal')) return 50;
  if (object.affordances.some((affordance) => affordance.kind === 'pick_up')) return 40;
  if (object.tags.includes('food')) return 35;
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
