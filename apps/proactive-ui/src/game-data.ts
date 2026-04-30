export type GearCategory = 'hydration' | 'navigation' | 'sun' | 'signal' | 'power' | 'first-aid';

export type GearTab = 'brief' | 'gear' | 'launch';

export interface GearItem {
  readonly id: string;
  readonly name: string;
  readonly category: GearCategory;
  readonly tab: GearTab;
  readonly weight: number;
  readonly cost: number;
  readonly summary: string;
  readonly tradeoff: string;
}

export interface CategoryGoal {
  readonly id: GearCategory;
  readonly label: string;
  readonly prompt: string;
}

export interface FieldKitScore {
  readonly selectedItems: readonly GearItem[];
  readonly coveredCategories: readonly GearCategory[];
  readonly missingCategories: readonly GearCategory[];
  readonly totalWeight: number;
  readonly totalCost: number;
  readonly readiness: number;
  readonly weightOk: boolean;
  readonly budgetOk: boolean;
  readonly readyToLaunch: boolean;
}

export const FIELD_KIT_LIMITS = {
  maxWeight: 9.5,
  maxBudget: 420,
  stormMinutes: 18,
} as const;

export const CATEGORY_GOALS: readonly CategoryGoal[] = [
  { id: 'hydration', label: 'Hydration', prompt: 'Carry enough water for dry heat.' },
  { id: 'navigation', label: 'Navigation', prompt: 'Keep the route readable if dust rolls in.' },
  { id: 'sun', label: 'Sun cover', prompt: 'Avoid exposure during the midday crossing.' },
  { id: 'signal', label: 'Signal', prompt: 'Give the team a way to call for extraction.' },
  { id: 'power', label: 'Power', prompt: 'Keep electronics alive through the ridge.' },
  { id: 'first-aid', label: 'First aid', prompt: 'Patch injuries before they become mission-ending.' },
] as const;

export const GEAR_ITEMS: readonly GearItem[] = [
  {
    id: 'water-cache',
    name: 'Twin water bladders',
    category: 'hydration',
    tab: 'gear',
    weight: 3.1,
    cost: 74,
    summary: 'Six liters with hose routing for hands-free travel.',
    tradeoff: 'Heavy, but solves the biggest desert risk.',
  },
  {
    id: 'electrolytes',
    name: 'Electrolyte tabs',
    category: 'hydration',
    tab: 'gear',
    weight: 0.2,
    cost: 18,
    summary: 'Small tablets for heat cramps and long walking pace.',
    tradeoff: 'Helpful support, but not a replacement for water volume.',
  },
  {
    id: 'paper-map',
    name: 'Laminated ridge map',
    category: 'navigation',
    tab: 'gear',
    weight: 0.3,
    cost: 22,
    summary: 'Weatherproof route map with marked dry washes.',
    tradeoff: 'Cheap and reliable, but needs another tool for precise bearings.',
  },
  {
    id: 'sat-compass',
    name: 'Satellite compass',
    category: 'navigation',
    tab: 'gear',
    weight: 0.9,
    cost: 136,
    summary: 'GPS, bearing, and breadcrumb trail in one rugged unit.',
    tradeoff: 'Power hungry and expensive.',
  },
  {
    id: 'shade-scarf',
    name: 'UV shade scarf',
    category: 'sun',
    tab: 'gear',
    weight: 0.4,
    cost: 31,
    summary: 'Wraps face and neck when glare and wind pick up.',
    tradeoff: 'Lightweight, but does not create rest shade.',
  },
  {
    id: 'pop-tarp',
    name: 'Pop-up shade tarp',
    category: 'sun',
    tab: 'gear',
    weight: 1.7,
    cost: 92,
    summary: 'Quick shelter for a heat stop or equipment repair.',
    tradeoff: 'Bulky compared with personal sun cover.',
  },
  {
    id: 'flare-kit',
    name: 'Signal flare kit',
    category: 'signal',
    tab: 'launch',
    weight: 0.8,
    cost: 58,
    summary: 'Visible distress signal if radio contact fails.',
    tradeoff: 'Single-use and only helpful when rescuers have line of sight.',
  },
  {
    id: 'beacon',
    name: 'Pocket rescue beacon',
    category: 'signal',
    tab: 'launch',
    weight: 0.6,
    cost: 168,
    summary: 'One-button satellite SOS with tracking pings.',
    tradeoff: 'Excellent safety, but consumes nearly half the budget.',
  },
  {
    id: 'solar-roll',
    name: 'Solar roll charger',
    category: 'power',
    tab: 'gear',
    weight: 1.2,
    cost: 114,
    summary: 'Flexible solar panel for beacon, compass, or phone.',
    tradeoff: 'Works best during exposed travel, not inside shade.',
  },
  {
    id: 'battery-bank',
    name: 'Rugged battery bank',
    category: 'power',
    tab: 'gear',
    weight: 0.9,
    cost: 69,
    summary: 'Pre-charged backup power sealed against grit.',
    tradeoff: 'Reliable once, but cannot recover charge in the field.',
  },
  {
    id: 'med-roll',
    name: 'Trauma med roll',
    category: 'first-aid',
    tab: 'launch',
    weight: 1.1,
    cost: 83,
    summary: 'Bandages, blister kit, antiseptic, and compression wrap.',
    tradeoff: 'Weighty, but covers the most likely field injuries.',
  },
  {
    id: 'snake-kit',
    name: 'Bite response card',
    category: 'first-aid',
    tab: 'launch',
    weight: 0.1,
    cost: 12,
    summary: 'Laminated bite protocol and emergency coordinates.',
    tradeoff: 'Useful guidance, not a complete medical kit.',
  },
] as const;

export function calculateFieldKitScore(selectedIds: ReadonlySet<string>): FieldKitScore {
  const selectedItems = GEAR_ITEMS.filter((item) => selectedIds.has(item.id));
  const coveredCategories = CATEGORY_GOALS
    .map((goal) => goal.id)
    .filter((category) => selectedItems.some((item) => item.category === category));
  const missingCategories = CATEGORY_GOALS
    .map((goal) => goal.id)
    .filter((category) => !coveredCategories.includes(category));
  const totalWeight = roundOne(selectedItems.reduce((sum, item) => sum + item.weight, 0));
  const totalCost = selectedItems.reduce((sum, item) => sum + item.cost, 0);
  const coverageScore = Math.round((coveredCategories.length / CATEGORY_GOALS.length) * 72);
  const weightScore = totalWeight <= FIELD_KIT_LIMITS.maxWeight
    ? 14
    : Math.max(0, 14 - Math.ceil((totalWeight - FIELD_KIT_LIMITS.maxWeight) * 5));
  const budgetScore = totalCost <= FIELD_KIT_LIMITS.maxBudget
    ? 14
    : Math.max(0, 14 - Math.ceil((totalCost - FIELD_KIT_LIMITS.maxBudget) / 18));
  const readiness = Math.min(100, coverageScore + weightScore + budgetScore);
  const weightOk = totalWeight <= FIELD_KIT_LIMITS.maxWeight;
  const budgetOk = totalCost <= FIELD_KIT_LIMITS.maxBudget;

  return {
    selectedItems,
    coveredCategories,
    missingCategories,
    totalWeight,
    totalCost,
    readiness,
    weightOk,
    budgetOk,
    readyToLaunch: missingCategories.length === 0 && weightOk && budgetOk,
  };
}

export function categoryLabel(category: GearCategory): string {
  return CATEGORY_GOALS.find((goal) => goal.id === category)?.label ?? category;
}

function roundOne(value: number): number {
  return Math.round(value * 10) / 10;
}
