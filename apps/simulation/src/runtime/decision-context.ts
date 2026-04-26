import { BAT_SWING_RADIUS, CHASE_MIN_DISTANCE, GOAL_RADIUS, ICE_THROW_RADIUS, INTERACTION_RADIUS } from './reducer.js';
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
  const hasCarrierInCloseRange = carrier != null && carrier.distance <= CHASE_MIN_DISTANCE;
  const bananaIsLoose = banana != null && !banana.heldBy;
  const powerUps = perception.nearbyObjects
    .filter((object) => object.kind === 'bat' || object.kind === 'ice_cube')
    .sort(compareObjectsForPriority);

  lines.push(`Tick ${perception.tick}. Banana Dash score: ${formatScore(perception)}.`);
  lines.push(`You are ${describeSelfState(perception)}.`);
  if (perception.lastDecision) {
    lines.push(`Heavily consider your last decision: ${perception.lastDecision}. Stay with it if it still makes sense, but change course if the banana or carrier situation is clearly better.`);
  }
  if (perception.directorNote) {
    lines.push(`Director says: ${perception.directorNote}`);
  }

  if (perception.self.frozenUntilTick > perception.tick) {
    lines.push(`You are frozen for ${perception.self.frozenUntilTick - perception.tick} more ticks.`);
    options.push({ label: 'wait', goal: { kind: 'wait', label: 'wait out the freeze' } });
    return { prompt: lines.join('\n'), options };
  }

  const isCarrier = perception.self.holding === perception.game.bananaObjectId;

  if (isCarrier) {
    addCarrierOptions(options, lines, goal);
  } else {
    const shouldPrioritizeCarrier = carrier != null;
    addNonCarrierOptions(
      perception,
      options,
      lines,
      banana,
      carrier,
      powerUps,
      sabotageCoolingDown,
      shouldPrioritizeCarrier
    );
    addCloseAgentOptions(perception, options, lines, sabotageCoolingDown);
    addAmbientOptions(perception, options, lines, powerUps);
    reorderNonCarrierOptions(options, perception, shouldPrioritizeCarrier);
  }

  dedupeOptionsInPlace(options);

  if (isCarrier) {
    lines.push('Choose your next action from the available options only. You have the banana, so focus on scoring at home base.');
  } else if (sabotageCoolingDown) {
    const guidance = hasCarrierInCloseRange
      ? 'Choose your next action from the available options only. The carrier is in contact range, so push them or take another active option.'
      : bananaIsLoose
        ? 'Choose your next action from the available options only. The banana is loose, so keep racing it while sabotage cools down.'
        : 'Choose your next action from the available options only. If another agent has the banana, keep pressure on them, but a nearby power-up is still worth peeling for when it sets up a faster interception.';
    lines.push(guidance);
  } else if (perception.self.powerUp) {
    lines.push('Choose your next action from the available options only. The banana is the goal; use your equipped power-up only to strip a carrier or stop the nearest rival to a loose banana.');
  } else {
    lines.push('Choose your next action from the available options only. The banana is the priority. Power-ups are setup plays only when they help you grab it faster or strip it from a carrier.');
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
  sabotageCoolingDown: boolean,
  shouldPrioritizeCarrier: boolean
): void {
  if (carrier) {
    const carrierIsTooCloseToChase = carrier.distance <= CHASE_MIN_DISTANCE;
    const carrierState = carrier.frozenUntilTick > perception.tick
      ? `${carrier.name} has the banana but is frozen for ${carrier.frozenUntilTick - perception.tick} more ticks.`
      : `${carrier.name} has the banana and is ${qualitativeDistance(carrier.distance)}.`;
    lines.push(carrierState);
    if (carrierIsTooCloseToChase) {
      if (sabotageCoolingDown) {
        lines.push(`${carrier.name} is already in contact range, but your sabotage is cooling down; shove only if it buys time.`);
      } else {
        lines.push(perception.self.powerUp
          ? `${carrier.name} is already in contact range. Prioritize your equipped power-up to knock the banana loose.`
          : `${carrier.name} is already in contact range. Prioritize bumping to knock the banana loose.`);
      }
    } else if (shouldPrioritizeCarrier) {
      lines.push(`The banana is already claimed, so direct pressure on ${carrier.name} is the default; detour only for a close power-up that creates a faster stop.`);
    }

    if (perception.self.powerUp?.kind === 'ice_cube' && carrier.distance <= ICE_THROW_RADIUS && !sabotageCoolingDown) {
      lines.push('Your ice cube can stop the carrier from range.');
      const label = sabotageLabel('ice_cube', carrier.name);
      options.push({
        label,
        goal: {
          kind: 'sabotage_agent',
          agentId: carrier.id,
          method: 'ice_cube',
          label,
        },
      });
    } else if (perception.self.powerUp && carrier.distance <= CHASE_MIN_DISTANCE && !sabotageCoolingDown) {
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

    if (!perception.self.powerUp && carrier.distance <= CHASE_MIN_DISTANCE && !sabotageCoolingDown) {
      options.push({
        label: `bump ${carrier.name}`,
        goal: { kind: 'sabotage_agent', agentId: carrier.id, method: 'bump', label: `bump ${carrier.name}` },
      });
    } else if (sabotageCoolingDown && !carrierIsTooCloseToChase) {
      lines.push('Your sabotage is cooling down, so this is a good moment to reposition or hunt a guaranteed hit.');
    }

    if (perception.self.powerUp?.kind === 'bat' && carrier.distance <= BAT_SWING_RADIUS && !sabotageCoolingDown) {
      lines.push(`${carrier.name} is inside your bat swing arc.`);
    }

    if (!carrierIsTooCloseToChase) {
      options.push({
        label: `chase ${carrier.name} with the banana`,
        goal: { kind: 'go_to_agent', agentId: carrier.id, label: `chase ${carrier.name} with the banana` },
      });
    }
  }

  if (banana && !banana.heldBy) {
    lines.push(`Banana is ${qualitativeDistance(banana.distance)}.`);
    const immediateThreat = findImmediateLooseBananaThreat(perception, banana);
    const powerUp = perception.self.powerUp;
    const tacticalTarget = !sabotageCoolingDown && powerUp
      ? findLooseBananaSabotageTarget(perception, banana)
      : null;
    if (tacticalTarget && powerUp) {
      lines.push(`${tacticalTarget.name} is the nearest rival to the loose banana, so a quick ${labelForPowerUp(powerUp.kind)} play can open the lane.`);
      const label = sabotageLabel(powerUp.kind, tacticalTarget.name);
      options.push({
        label,
        goal: {
          kind: 'sabotage_agent',
          agentId: tacticalTarget.id,
          method: powerUp.kind,
          label,
        },
      });
    } else if (immediateThreat) {
      lines.push(`${immediateThreat.name} is the nearest rival to the loose banana.`);
    }
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
    const visiblePowerUps = powerUps.slice(0, 2);
    for (const [index, powerUp] of visiblePowerUps.entries()) {
      lines.push(`- ${powerUp.label} (${qualitativeDistance(powerUp.distance)})`);
      const labelTarget = visiblePowerUps.length > 1 ? `${ordinalLabel(index)} ${powerUp.label}` : powerUp.label;
      const label = powerUp.distance <= INTERACTION_RADIUS ? `grab ${labelTarget}` : `go get the ${labelTarget}`;
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
}

function addCloseAgentOptions(
  perception: AgentPerception,
  options: DecisionOption[],
  lines: string[],
  sabotageCoolingDown: boolean
): void {
  if (!sabotageCoolingDown) return;

  const closeAgents = perception.nearbyAgents
    .filter((agent) => agent.distance <= CHASE_MIN_DISTANCE)
    .filter((agent) => agent.holding === perception.game.bananaObjectId)
    .sort((a, b) => a.distance - b.distance)
    .slice(0, 3);
  if (closeAgents.length === 0) return;

  lines.push('Close agents you can shove while sabotage is cooling down:');
  const pushOptions: DecisionOption[] = [];
  for (const agent of closeAgents) {
    const label = `push ${agent.name}`;
    lines.push(`- ${agent.name} (${qualitativeDistance(agent.distance)})`);
    pushOptions.push({
      label,
      goal: { kind: 'push_agent', agentId: agent.id, label },
    });
  }
  options.push(...pushOptions);
}

function addAmbientOptions(
  perception: AgentPerception,
  options: DecisionOption[],
  lines: string[],
  powerUps: readonly PerceivedObject[]
): void {
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
}

function reorderNonCarrierOptions(
  options: DecisionOption[],
  perception: AgentPerception,
  shouldPrioritizeCarrier: boolean
): void {
  options.sort((a, b) => getDecisionPriorityScore(b, perception, shouldPrioritizeCarrier) - getDecisionPriorityScore(a, perception, shouldPrioritizeCarrier));
}

function getDecisionPriorityScore(
  option: DecisionOption,
  perception: AgentPerception,
  shouldPrioritizeCarrier: boolean
): number {
  const carrierAgentId = perception.nearbyAgents.find((agent) => agent.holding === perception.game.bananaObjectId)?.id ?? null;
  const selfArchetype = perception.self.archetype ?? perception.self.id;
  const objectDistance = getPerceivedObjectDistance(perception, option.goal.kind === 'object_action' || option.goal.kind === 'go_to_object'
    ? option.goal.objectId
    : null);
  const carrierDistance = carrierAgentId ? getPerceivedAgentDistance(perception, carrierAgentId) : null;
  const powerUpBias = powerUpBiasForArchetype(selfArchetype);
  const chaseBias = chaseBiasForArchetype(selfArchetype);

  switch (option.goal.kind) {
    case 'sabotage_agent':
      return option.goal.agentId === carrierAgentId ? 120 : 96;
    case 'push_agent':
      return option.goal.agentId === carrierAgentId ? 110 : 93;
    case 'go_to_agent':
      if (option.goal.agentId === carrierAgentId) {
        const distanceScore = carrierDistance == null ? 0 : Math.max(0, 10 - carrierDistance * 1.5);
        return (shouldPrioritizeCarrier ? 94 : 80) + chaseBias + distanceScore;
      }
      return 70;
    case 'object_action':
      if (option.goal.objectId === perception.game.bananaObjectId) return 110;
      return scorePowerUpOption(option.goal.objectId, objectDistance, shouldPrioritizeCarrier, powerUpBias);
    case 'go_to_object':
      if (option.goal.objectId === perception.game.bananaObjectId) return 105;
      return scorePowerUpOption(option.goal.objectId, objectDistance, shouldPrioritizeCarrier, powerUpBias);
    case 'wait':
      return 10;
    default:
      return 60;
  }
}

function scorePowerUpOption(
  objectId: string,
  distance: number | null,
  shouldPrioritizeCarrier: boolean,
  powerUpBias: number
): number {
  const distanceBonus = distance == null ? 0 : Math.max(0, 8 - distance * 2);
  const base = shouldPrioritizeCarrier ? 80 : 78;
  if (objectId.includes('bat')) {
    return base + powerUpBias + distanceBonus;
  }
  if (objectId.includes('ice')) {
    return base + powerUpBias + distanceBonus + 1;
  }
  return base + distanceBonus;
}

function getPerceivedObjectDistance(perception: AgentPerception, objectId: string | null): number | null {
  if (!objectId) return null;
  return perception.nearbyObjects.find((object) => object.id === objectId)?.distance ?? null;
}

function getPerceivedAgentDistance(perception: AgentPerception, agentId: string): number | null {
  return perception.nearbyAgents.find((agent) => agent.id === agentId)?.distance ?? null;
}

function findImmediateLooseBananaThreat(
  perception: AgentPerception,
  banana: PerceivedObject | undefined
): PerceivedAgent | null {
  if (!banana || banana.heldBy) return null;
  const selfToBanana = banana.distance;
  return perception.nearbyAgents
    .filter((agent) => agent.frozenUntilTick <= perception.tick)
    .map((agent) => ({ agent, bananaDistance: distanceBetweenDirections(agent.distance, agent.direction, selfToBanana, banana.direction) }))
    .filter((entry) => entry.bananaDistance < selfToBanana)
    .sort((a, b) => a.bananaDistance - b.bananaDistance || a.agent.distance - b.agent.distance)
    .map((entry) => entry.agent)[0] ?? null;
}

function findLooseBananaSabotageTarget(
  perception: AgentPerception,
  banana: PerceivedObject | undefined
): PerceivedAgent | null {
  const threat = findImmediateLooseBananaThreat(perception, banana);
  if (!threat || !perception.self.powerUp) return null;
  const range = perception.self.powerUp.kind === 'ice_cube' ? ICE_THROW_RADIUS : BAT_SWING_RADIUS;
  return threat.distance <= range ? threat : null;
}

function distanceBetweenDirections(
  distanceA: number,
  directionA: { readonly x: number; readonly z: number },
  distanceB: number,
  directionB: { readonly x: number; readonly z: number }
): number {
  const ax = directionA.x * distanceA;
  const az = directionA.z * distanceA;
  const bx = directionB.x * distanceB;
  const bz = directionB.z * distanceB;
  const dx = ax - bx;
  const dz = az - bz;
  return Math.sqrt(dx * dx + dz * dz);
}

function powerUpBiasForArchetype(archetype: string): number {
  switch (archetype) {
    case 'mira':
      return 4;
    case 'beck':
      return 3;
    case 'sol':
      return 2;
    case 'aria':
      return 0;
    default:
      return 0;
  }
}

function chaseBiasForArchetype(archetype: string): number {
  switch (archetype) {
    case 'aria':
      return 6;
    case 'sol':
      return 2;
    case 'beck':
      return 1;
    case 'mira':
      return -2;
    default:
      return 0;
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
  return powerUp === 'bat' ? `smack ${targetName} with the bat` : `throw the ice cube at ${targetName}`;
}

function labelForPowerUp(powerUp: PowerUpKind): string {
  return powerUp === 'bat' ? 'the bat' : 'the ice cube';
}

function ordinalLabel(index: number): string {
  return index === 0 ? 'closest' : 'second';
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
