import { Action, Goal, Thing, AgentState, LocationInfo, Phase, ExecutionPlan } from './types.js';

interface EmscriptenModule {
  _malloc(size: number): number;
  _free(ptr: number): void;
  lengthBytesUTF8(text: string): number;
  stringToUTF8(text: string, outPtr: number, maxBytesToWrite: number): void;
  UTF8ToString(ptr: number): string;
  ccall(ident: string, returnType: string | null, argTypes: string[], args: unknown[]): unknown | Promise<unknown>;
  HEAP8: Int8Array;
  HEAPU8: Uint8Array;
}

interface ActionLayout {
  size: number;
  id: number;
  name: number;
  description: number;
}

interface GoalLayout {
  size: number;
  id: number;
  name: number;
  description: number;
  priority: number;
}

interface ThingLayout {
  size: number;
  id: number;
  name: number;
  description: number;
  distanceToAgent: number;
  actions: number;
  actionsCount: number;
}

interface AgentStateLayout {
  size: number;
  name: number;
  persona: number;
  activeAction: number;
  activeThing: number;
}

interface LocationLayout {
  size: number;
  locationCompressed: number;
}

interface PhaseLayout {
  size: number;
  step: number;
  action: number;
  thing: number;
}

interface ExecutionPlanLayout {
  size: number;
  agentId: number;
  phases: number;
  phasesCount: number;
}

interface AbiLayout {
  action: ActionLayout;
  goal: GoalLayout;
  thing: ThingLayout;
  agentState: AgentStateLayout;
  location: LocationLayout;
  phase: PhaseLayout;
  executionPlan: ExecutionPlanLayout;
}

const MAX_READ_ARRAY_LENGTH = 100000;
const EXPECTED_ABI_VERSION_PREFIX = '1.';

function readInt(module: EmscriptenModule, symbol: string): number {
  const value = module.ccall(symbol, 'number', [], []);
  if (typeof value !== 'number') {
    throw new Error(`ABI symbol ${symbol} returned an async result`);
  }
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`Invalid ABI value for ${symbol}: ${value}`);
  }
  return value;
}

function readStringSymbol(module: EmscriptenModule, symbol: string): string {
  const value = module.ccall(symbol, 'string', [], []);
  if (typeof value !== 'string') {
    throw new Error(`ABI symbol ${symbol} returned a non-string value`);
  }
  return value;
}

function validateFieldRange(
  structName: string,
  structSize: number,
  fieldName: string,
  offset: number,
  width: number
): void {
  if (offset + width > structSize) {
    throw new Error(`ABI layout mismatch: ${structName}.${fieldName} exceeds struct size (${offset} + ${width} > ${structSize})`);
  }
}

function validateAbiLayout(abi: AbiLayout): void {
  validateFieldRange('ActionFFI', abi.action.size, 'Id', abi.action.id, 4);
  validateFieldRange('ActionFFI', abi.action.size, 'Name', abi.action.name, 4);
  validateFieldRange('ActionFFI', abi.action.size, 'Description', abi.action.description, 4);

  validateFieldRange('GoalFFI', abi.goal.size, 'Id', abi.goal.id, 4);
  validateFieldRange('GoalFFI', abi.goal.size, 'Name', abi.goal.name, 4);
  validateFieldRange('GoalFFI', abi.goal.size, 'Description', abi.goal.description, 4);
  validateFieldRange('GoalFFI', abi.goal.size, 'Priority', abi.goal.priority, 4);

  validateFieldRange('ThingFFI', abi.thing.size, 'Id', abi.thing.id, 4);
  validateFieldRange('ThingFFI', abi.thing.size, 'Name', abi.thing.name, 4);
  validateFieldRange('ThingFFI', abi.thing.size, 'Description', abi.thing.description, 4);
  validateFieldRange('ThingFFI', abi.thing.size, 'DistanceToAgent', abi.thing.distanceToAgent, 4);
  validateFieldRange('ThingFFI', abi.thing.size, 'Actions', abi.thing.actions, 4);
  validateFieldRange('ThingFFI', abi.thing.size, 'ActionsCount', abi.thing.actionsCount, 4);

  validateFieldRange('AgentStateFFI', abi.agentState.size, 'Name', abi.agentState.name, 4);
  validateFieldRange('AgentStateFFI', abi.agentState.size, 'Persona', abi.agentState.persona, 4);
  validateFieldRange('AgentStateFFI', abi.agentState.size, 'ActiveAction', abi.agentState.activeAction, abi.action.size);
  validateFieldRange('AgentStateFFI', abi.agentState.size, 'ActiveThing', abi.agentState.activeThing, abi.thing.size);

  validateFieldRange('LocationFFI', abi.location.size, 'LocationCompressed', abi.location.locationCompressed, 4);

  validateFieldRange('PhaseFFI', abi.phase.size, 'Step', abi.phase.step, 4);
  validateFieldRange('PhaseFFI', abi.phase.size, 'Action', abi.phase.action, abi.action.size);
  validateFieldRange('PhaseFFI', abi.phase.size, 'Thing', abi.phase.thing, abi.thing.size);

  validateFieldRange('ExecutionPlanFFI', abi.executionPlan.size, 'AgentId', abi.executionPlan.agentId, 4);
  validateFieldRange('ExecutionPlanFFI', abi.executionPlan.size, 'Phases', abi.executionPlan.phases, 4);
  validateFieldRange('ExecutionPlanFFI', abi.executionPlan.size, 'PhasesCount', abi.executionPlan.phasesCount, 4);
}

function resolveAbi(module: EmscriptenModule): AbiLayout {
  const abiVersion = readStringSymbol(module, 'CE_Unity_GetVersion');
  if (!abiVersion.startsWith(EXPECTED_ABI_VERSION_PREFIX)) {
    throw new Error(`Incompatible runtime ABI version "${abiVersion}". Expected prefix "${EXPECTED_ABI_VERSION_PREFIX}".`);
  }

  const abi: AbiLayout = {
    action: {
      size: readInt(module, 'CE_Unity_SizeOfActionFFI'),
      id: readInt(module, 'CE_Unity_Offset_ActionFFI_Id'),
      name: readInt(module, 'CE_Unity_Offset_ActionFFI_Name'),
      description: readInt(module, 'CE_Unity_Offset_ActionFFI_Description')
    },
    goal: {
      size: readInt(module, 'CE_Unity_SizeOfGoalFFI'),
      id: readInt(module, 'CE_Unity_Offset_GoalFFI_Id'),
      name: readInt(module, 'CE_Unity_Offset_GoalFFI_Name'),
      description: readInt(module, 'CE_Unity_Offset_GoalFFI_Description'),
      priority: readInt(module, 'CE_Unity_Offset_GoalFFI_Priority')
    },
    thing: {
      size: readInt(module, 'CE_Unity_SizeOfThingFFI'),
      id: readInt(module, 'CE_Unity_Offset_ThingFFI_Id'),
      name: readInt(module, 'CE_Unity_Offset_ThingFFI_Name'),
      description: readInt(module, 'CE_Unity_Offset_ThingFFI_Description'),
      distanceToAgent: readInt(module, 'CE_Unity_Offset_ThingFFI_DistanceToAgent'),
      actions: readInt(module, 'CE_Unity_Offset_ThingFFI_Actions'),
      actionsCount: readInt(module, 'CE_Unity_Offset_ThingFFI_ActionsCount')
    },
    agentState: {
      size: readInt(module, 'CE_Unity_SizeOfAgentStateFFI'),
      name: readInt(module, 'CE_Unity_Offset_AgentStateFFI_Name'),
      persona: readInt(module, 'CE_Unity_Offset_AgentStateFFI_Persona'),
      activeAction: readInt(module, 'CE_Unity_Offset_AgentStateFFI_ActiveAction'),
      activeThing: readInt(module, 'CE_Unity_Offset_AgentStateFFI_ActiveThing')
    },
    location: {
      size: readInt(module, 'CE_Unity_SizeOfLocationFFI'),
      locationCompressed: readInt(module, 'CE_Unity_Offset_LocationFFI_LocationCompressed')
    },
    phase: {
      size: readInt(module, 'CE_Unity_SizeOfPhaseFFI'),
      step: readInt(module, 'CE_Unity_Offset_PhaseFFI_Step'),
      action: readInt(module, 'CE_Unity_Offset_PhaseFFI_Action'),
      thing: readInt(module, 'CE_Unity_Offset_PhaseFFI_Thing')
    },
    executionPlan: {
      size: readInt(module, 'CE_Unity_SizeOfExecutionPlanFFI'),
      agentId: readInt(module, 'CE_Unity_Offset_ExecutionPlanFFI_AgentId'),
      phases: readInt(module, 'CE_Unity_Offset_ExecutionPlanFFI_Phases'),
      phasesCount: readInt(module, 'CE_Unity_Offset_ExecutionPlanFFI_PhasesCount')
    }
  };

  validateAbiLayout(abi);
  return abi;
}

/**
 * Handles allocating, writing, reading, and freeing C-structs
 * in WebAssembly linear memory.
 */
export class MemoryManager {
  private static readonly abiCache = new WeakMap<object, AbiLayout>();
  private readonly module: EmscriptenModule;
  private readonly abi: AbiLayout;
  private allocations: number[] = [];
  private heapBuffer: ArrayBufferLike | null = null;
  private heapView: DataView | null = null;

  constructor(module: EmscriptenModule) {
    this.module = module;
    const cached = MemoryManager.abiCache.get(module as unknown as object);
    if (cached) {
      this.abi = cached;
    } else {
      const abi = resolveAbi(module);
      MemoryManager.abiCache.set(module as unknown as object, abi);
      this.abi = abi;
    }
  }

  private get view(): DataView {
    const currentBuffer = this.module.HEAP8.buffer;
    if (this.heapBuffer !== currentBuffer || !this.heapView) {
      this.heapBuffer = currentBuffer;
      this.heapView = new DataView(currentBuffer);
    }
    return this.heapView;
  }

  private zeroRegion(ptr: number, size: number) {
    this.module.HEAPU8.fill(0, ptr, ptr + size);
  }

  public malloc(size: number): number {
    if (!Number.isInteger(size) || size <= 0) {
      throw new Error(`Invalid allocation size: ${size}`);
    }
    const ptr = this.module._malloc(size);
    if (!ptr) {
      throw new Error(`WASM allocation failed for ${size} bytes`);
    }
    this.allocations.push(ptr);
    return ptr;
  }

  public freeAllocations() {
    for (let i = this.allocations.length - 1; i >= 0; i -= 1) {
      this.module._free(this.allocations[i]);
    }
    this.allocations = [];
  }

  public writeString(str: string | undefined): number {
    if (str === undefined) {
      return 0;
    }
    const lengthBytes = this.module.lengthBytesUTF8(str) + 1;
    const ptr = this.malloc(lengthBytes);
    this.module.stringToUTF8(str, ptr, lengthBytes);
    return ptr;
  }

  public readString(ptr: number): string {
    if (ptr === 0) {
      return '';
    }
    return this.module.UTF8ToString(ptr);
  }

  public writeAction(action: Action, ptr: number = this.malloc(this.abi.action.size)): number {
    this.view.setUint32(ptr + this.abi.action.id, this.writeString(action.id), true);
    this.view.setUint32(ptr + this.abi.action.name, this.writeString(action.name), true);
    this.view.setUint32(ptr + this.abi.action.description, this.writeString(action.description), true);
    return ptr;
  }

  public readAction(ptr: number): Action {
    return {
      id: this.readString(this.view.getUint32(ptr + this.abi.action.id, true)),
      name: this.readString(this.view.getUint32(ptr + this.abi.action.name, true)),
      description: this.readString(this.view.getUint32(ptr + this.abi.action.description, true))
    };
  }

  public writeActionArray(actions: Action[]): number {
    if (actions.length === 0) {
      return 0;
    }
    const ptr = this.malloc(actions.length * this.abi.action.size);
    for (let i = 0; i < actions.length; i += 1) {
      this.writeAction(actions[i], ptr + (i * this.abi.action.size));
    }
    return ptr;
  }

  public writeGoal(goal: Goal, ptr: number = this.malloc(this.abi.goal.size)): number {
    this.view.setUint32(ptr + this.abi.goal.id, this.writeString(goal.id), true);
    this.view.setUint32(ptr + this.abi.goal.name, this.writeString(goal.name), true);
    this.view.setUint32(ptr + this.abi.goal.description, this.writeString(goal.description), true);
    this.view.setFloat32(ptr + this.abi.goal.priority, goal.priority, true);
    return ptr;
  }

  public writeGoalArray(goals: Goal[]): number {
    if (goals.length === 0) {
      return 0;
    }
    const ptr = this.malloc(goals.length * this.abi.goal.size);
    for (let i = 0; i < goals.length; i += 1) {
      this.writeGoal(goals[i], ptr + (i * this.abi.goal.size));
    }
    return ptr;
  }

  public writeThing(thing: Thing, ptr: number = this.malloc(this.abi.thing.size)): number {
    this.view.setUint32(ptr + this.abi.thing.id, this.writeString(thing.id), true);
    this.view.setUint32(ptr + this.abi.thing.name, this.writeString(thing.name), true);
    this.view.setUint32(ptr + this.abi.thing.description, this.writeString(thing.description), true);
    this.view.setFloat32(ptr + this.abi.thing.distanceToAgent, thing.distanceToAgent, true);

    const actions = thing.actions ?? [];
    const actionsCount = actions.length;
    const actionsPtr = actionsCount > 0 ? this.writeActionArray(actions) : 0;
    this.view.setUint32(ptr + this.abi.thing.actions, actionsPtr, true);
    this.view.setInt32(ptr + this.abi.thing.actionsCount, actionsCount, true);
    return ptr;
  }

  public readThing(ptr: number): Thing {
    const id = this.readString(this.view.getUint32(ptr + this.abi.thing.id, true));
    const name = this.readString(this.view.getUint32(ptr + this.abi.thing.name, true));
    const description = this.readString(this.view.getUint32(ptr + this.abi.thing.description, true));
    const distanceToAgent = this.view.getFloat32(ptr + this.abi.thing.distanceToAgent, true);
    const actionsPtr = this.view.getUint32(ptr + this.abi.thing.actions, true);
    const actionsCount = this.view.getInt32(ptr + this.abi.thing.actionsCount, true);

    if (actionsCount < 0 || actionsCount > MAX_READ_ARRAY_LENGTH) {
      throw new Error(`Invalid actionsCount from WASM: ${actionsCount}`);
    }
    if (actionsCount > 0 && actionsPtr === 0) {
      throw new Error('Invalid WASM thing layout: actions pointer is null but actionsCount > 0');
    }

    const actions: Action[] = [];
    for (let i = 0; i < actionsCount; i += 1) {
      actions.push(this.readAction(actionsPtr + (i * this.abi.action.size)));
    }

    return { id, name, description, distanceToAgent, actions };
  }

  public writeThingArray(things: Thing[]): number {
    if (things.length === 0) {
      return 0;
    }
    const ptr = this.malloc(things.length * this.abi.thing.size);
    for (let i = 0; i < things.length; i += 1) {
      this.writeThing(things[i], ptr + (i * this.abi.thing.size));
    }
    return ptr;
  }

  public writeLocation(location: LocationInfo, ptr: number = this.malloc(this.abi.location.size)): number {
    this.view.setUint32(
      ptr + this.abi.location.locationCompressed,
      this.writeString(location.locationCompressed),
      true
    );
    return ptr;
  }

  public writeAgentState(state: AgentState, ptr: number = this.malloc(this.abi.agentState.size)): number {
    this.view.setUint32(ptr + this.abi.agentState.name, this.writeString(state.name), true);
    this.view.setUint32(ptr + this.abi.agentState.persona, this.writeString(state.persona), true);

    if (state.activeAction) {
      this.writeAction(state.activeAction, ptr + this.abi.agentState.activeAction);
    } else {
      this.zeroRegion(ptr + this.abi.agentState.activeAction, this.abi.action.size);
    }

    if (state.activeThing) {
      this.writeThing(state.activeThing, ptr + this.abi.agentState.activeThing);
    } else {
      this.zeroRegion(ptr + this.abi.agentState.activeThing, this.abi.thing.size);
    }

    return ptr;
  }

  public readPhase(ptr: number): Phase {
    const step = this.view.getInt32(ptr + this.abi.phase.step, true);
    const action = this.readAction(ptr + this.abi.phase.action);
    const thing = this.readThing(ptr + this.abi.phase.thing);
    return { step, action, thing };
  }

  public readExecutionPlan(ptr: number): ExecutionPlan {
    const agentIdPtr = this.view.getUint32(ptr + this.abi.executionPlan.agentId, true);
    const phasesPtr = this.view.getUint32(ptr + this.abi.executionPlan.phases, true);
    const phasesCount = this.view.getInt32(ptr + this.abi.executionPlan.phasesCount, true);

    if (phasesCount < 0 || phasesCount > MAX_READ_ARRAY_LENGTH) {
      throw new Error(`Invalid phasesCount from WASM: ${phasesCount}`);
    }
    if (phasesCount > 0 && phasesPtr === 0) {
      throw new Error('Invalid WASM execution plan layout: phases pointer is null but phasesCount > 0');
    }

    const phases: Phase[] = [];
    for (let i = 0; i < phasesCount; i += 1) {
      phases.push(this.readPhase(phasesPtr + (i * this.abi.phase.size)));
    }

    return {
      agentId: this.readString(agentIdPtr),
      phases
    };
  }
}
