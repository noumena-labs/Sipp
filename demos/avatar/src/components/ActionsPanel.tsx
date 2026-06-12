//////////////////////////////////////////////////////////////////////////////
//
// ActionsPanel.tsx
//
// - Manual action trigger surface for demoing avatar emotes and fantasy
//   effects without asking the model to emit cues.
//
//////////////////////////////////////////////////////////////////////////////

import type { CharacterConfig } from '@noumena-labs/sipp/character';

type ActionSpec = CharacterConfig['actions'][number];

interface ActionGroup {
  readonly label: string;
  readonly actionIds: readonly string[];
}

interface ActionsPanelProps {
  readonly actions: CharacterConfig['actions'];
  readonly disabled?: boolean;
  readonly onTrigger: (actionId: string, cueLabel: string) => void;
}

const ACTION_GROUPS: readonly ActionGroup[] = [
  {
    label: 'Emotes',
    actionIds: [
      'wave',
      'salute',
      'nod',
      'shake_head',
      'thinking',
      'bashful',
      'excited',
      'happy_blissful',
      'joy_jump',
      'upset_angry',
      'crying',
      'sad_idle',
    ],
  },
  {
    label: 'Face',
    actionIds: ['smile', 'look_sad', 'gasp', 'look_angry', 'settle'],
  },
  {
    label: 'Gaze',
    actionIds: ['look_at_you', 'glance_left', 'glance_right', 'look_up', 'look_down'],
  },
  {
    label: 'Magic',
    actionIds: ['summon_familiar', 'cast_starbolt', 'raise_ward', 'summon_rune_circle'],
  },
];

export function ActionsPanel({ actions, disabled, onTrigger }: ActionsPanelProps) {
  const actionMap = new Map(actions.map((action) => [action.id, action]));
  const renderedGroups = ACTION_GROUPS.map((group) => ({
    ...group,
    actions: group.actionIds.map((id) => actionMap.get(id)).filter(isActionSpec),
  })).filter((group) => group.actions.length > 0);

  if (renderedGroups.length === 0) {
    return null;
  }

  return (
    <section className="actions-panel glass-panel" aria-label="Manual avatar actions">
      <div className="actions-panel-header">
        <span className="panel-eyebrow">Actions</span>
      </div>
      <div className="actions-panel-groups">
        {renderedGroups.map((group) => (
          <div key={group.label} className="action-group">
            <div className="action-group-label">{group.label}</div>
            <div className="action-buttons">
              {group.actions.map((action) => (
                <button
                  key={action.id}
                  type="button"
                  className={`manual-action-button ${group.label.toLowerCase()}`}
                  title={action.description ?? action.id}
                  onClick={() => onTrigger(action.id, action.cue ?? action.id.replace(/_/g, ' '))}
                  disabled={disabled}
                >
                  {formatActionLabel(action.cue ?? action.id)}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function isActionSpec(action: ActionSpec | undefined): action is ActionSpec {
  return action != null;
}

function formatActionLabel(value: string): string {
  return value
    .replace(/_/g, ' ')
    .split(' ')
    .filter((part) => part.length > 0)
    .map((part) => part[0].toUpperCase() + part.slice(1))
    .join(' ');
}
