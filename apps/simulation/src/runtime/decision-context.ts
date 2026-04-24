import { GOAL_RADIUS, INTERACTION_RADIUS, SABOTAGE_RADIUS } from './reducer.js';
import type {
  AgentPerception,
  DecisionContext,
  DecisionOption,
  PerceivedAgent,
  PerceivedObject,
  PowerUpKind,
} from './types.js';

export function buildDecisionContext(perception: AgentPerception): DecisionContext {
  const options: DecisionOption[] = [];
  const lines: string[] = [];
  const banana = findObject(perception, perception.game.bananaObjectId);
  const goal = findObject(perception, perception.game.goalObjectId);
  const carrier = perception.nearbyAgents.find((agent) => agent.holding === perception.game.bananaObjectId);
  const sabotageCoolingDown = perception.self.cooldowns.sabotageUntilTick > perception.tick;
  const powerUps = perception.nearbyObjects
    .filter((object) => object.kind === 'bat' || object.kind === 'ice_cube')
    .sort(compareObjectsForPriority);

  lines.push(`Tick ${perception.tick}. Banana Dash score: ${formatScore(perception)}.`);
  lines.push(`You are ${describeSelfState(perception)}.`);
  if (perception.directorNote) {
    lines.push(`Director says: ${perception.directorNote}`);
  }

  if (perception.self.frozenUntilTick > perception.tick) {
    lines.push(`You are frozen for ${perception.self.frozenUntilTick - perception.tick} more ticks.`);
    options.push({ label: 'wait', goal: { kind: 'wait', label: 'wait out the freeze' } });
    return { prompt: lines.join('\n'), options };
  }

  if (perception.self.holding === perception.game.bananaObjectId) {
    addCarrierOptions(options, lines, goal);
  } else {
    addNonCarrierOptions(perception, options, lines, banana, carrier, powerUps, sabotageCoolingDown);
  }

  addAmbientOptions(perception, options, lines, powerUps);
  options.push({ label: 'wait', goal: { kind: 'wait', label: 'wait' } });
  dedupeOptionsInPlace(options);

  if (sabotageCoolingDown) {
    lines.push('Choose your next action from the available options only. Keep moving, reposition, or grab a power-up while your bump cooldown clears.');
  } else if (perception.self.powerUp) {
    lines.push('Choose your next action from the available options only. A guaranteed slapstick hit with your equipped power-up is valuable if the carrier is reachable.');
  } else {
    lines.push('Choose your next action from the available options only. Scoring matters, but side-lane power-ups can create a better swing than dogpiling every chase.');
  }
  return { prompt: lines.join('\n'), options };
}

function addCarrierOptions(
  options: DecisionOption[],
  lines: string[],
  goal: PerceivedObject | undefined
): void {
  lines.push('You have the banana. Get it to home base before someone clobbers or freezes you.');
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
    options.push({
      label: 'keep running to base',
      goal: { kind: 'deliver', objectId: goal.id, label: 'keep running to base' },
    });
  }
}

function addNonCarrierOptions(
  perception: AgentPerception,
  options: DecisionOption[],
  lines: string[],
  banana: PerceivedObject | undefined,
  carrier: PerceivedAgent | undefined,
  powerUps: readonly PerceivedObject[],
  sabotageCoolingDown: boolean
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

  if (!perception.self.powerUp && powerUps.length > 0) {
    lines.push('Power-ups on the field:');
    for (const powerUp of powerUps.slice(0, 2)) {
      lines.push(`- ${powerUp.label} (${qualitativeDistance(powerUp.distance)})`);
      const label = powerUp.distance <= INTERACTION_RADIUS ? `grab ${powerUp.label}` : `go get the ${powerUp.label}`;
      options.push({
        label,
        goal: powerUp.distance <= INTERACTION_RADIUS
          ? {
              kind: 'object_action',
              objectId: powerUp.id,
              affordance: { kind: 'pick_up', label, status: `grabbing the ${powerUp.label}` },
              label,
            }
          : { kind: 'go_to_object', objectId: powerUp.id, label },
      });
    }
  }

  if (carrier) {
    const carrierState = carrier.frozenUntilTick > perception.tick
      ? `${carrier.name} has the banana but is frozen for ${carrier.frozenUntilTick - perception.tick} more ticks.`
      : `${carrier.name} has the banana and is ${qualitativeDistance(carrier.distance)}.`;
    lines.push(carrierState);

    if (perception.self.powerUp && carrier.distance <= SABOTAGE_RADIUS * 1.4) {
      const label = sabotageLabel(perception.self.powerUp.kind, carrier.name);
      options.push({
        label,
        goal: {
          kind: 'sabotage_agent',
          agentId: carrier.id,
          method: perception.self.powerUp.kind,
          label,
        },
      });
    }

    if (carrier.distance <= SABOTAGE_RADIUS * 1.6 && !sabotageCoolingDown) {
      options.push({
        label: `bump ${carrier.name}`,
        goal: { kind: 'sabotage_agent', agentId: carrier.id, method: 'bump', label: `bump ${carrier.name}` },
      });
    } else if (sabotageCoolingDown) {
      lines.push('Your regular bump is cooling down, so this is a good moment to reposition or hunt a guaranteed hit.');
    }

    options.push({
      label: `chase ${carrier.name}`,
      goal: { kind: 'go_to_agent', agentId: carrier.id, label: `chase ${carrier.name}` },
    });
  }
}

function addAmbientOptions(
  perception: AgentPerception,
  options: DecisionOption[],
  lines: string[],
  powerUps: readonly PerceivedObject[]
): void {
  const visibleAgents = perception.nearbyAgents.filter((agent) => agent.holding !== perception.game.bananaObjectId);
  const visibleObjects = perception.nearbyObjects
    .filter((object) => !object.tags.includes('obstacle'))
    .filter((object) => object.id !== perception.game.bananaObjectId)
    .filter((object) => object.id !== perception.game.goalObjectId)
    .filter((object) => !powerUps.some((powerUp) => powerUp.id === object.id))
    .sort(compareObjectsForPriority);

  if (options.length <= 2 && visibleObjects.length > 0) {
    lines.push('Other points of interest:');
    for (const object of visibleObjects.slice(0, 2)) {
      const label = `go to the ${object.label}`;
      lines.push(`- ${object.label} (${qualitativeDistance(object.distance)})`);
      options.push({ label, goal: { kind: 'go_to_object', objectId: object.id, label } });
    }
  }

  if (options.length <= 2 && visibleAgents.length > 0) {
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

function describeSelfState(perception: AgentPerception): string {
  const parts: string[] = [];
  parts.push(perception.self.holding ? `carrying ${perception.self.holding}` : 'empty-handed');
  if (perception.self.powerUp) {
    parts.push(`equipped with ${labelForPowerUp(perception.self.powerUp.kind)}`);
  }
  if (perception.self.frozenUntilTick > perception.tick) {
    parts.push('currently frozen');
  }
  return parts.join(', ');
}

function formatScore(perception: AgentPerception): string {
  return Object.entries(perception.game.score.deliveries)
    .map(([agentId, score]) => `${agentId} ${score}`)
    .join(', ');
}

function sabotageLabel(powerUp: PowerUpKind, targetName: string): string {
  return powerUp === 'bat' ? `smack ${targetName} with the bat` : `freeze ${targetName} with the ice cube`;
}

function labelForPowerUp(powerUp: PowerUpKind): string {
  return powerUp === 'bat' ? 'the bat' : 'the ice cube';
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
  if (object.kind === 'ice_cube') return 85;
  if (object.kind === 'bat') return 80;
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
