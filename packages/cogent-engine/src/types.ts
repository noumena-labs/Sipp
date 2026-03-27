export interface Action {
  id: string;
  name: string;
  description: string;
}

export interface Goal {
  id: string;
  name: string;
  description: string;
  priority: number;
}

export interface Thing {
  id: string;
  name: string;
  description: string;
  distanceToAgent: number;
  actions?: Action[];
}

export interface AgentState {
  name: string;
  persona: string;
  activeAction?: Action;
  activeThing?: Thing;
}

export interface LocationInfo {
  locationCompressed: string;
}

export interface Phase {
  step: number;
  action: Action;
  thing: Thing;
}

export interface ExecutionPlan {
  agentId: string;
  phases: Phase[];
}
