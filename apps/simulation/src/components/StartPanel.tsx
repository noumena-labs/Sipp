//////////////////////////////////////////////////////////////////////////////
//
// components/StartPanel.tsx
//
// - Pre-simulation start screen. Lets the user choose the local GGUF model
//   before entering the Banana Dash simulation.
//
//////////////////////////////////////////////////////////////////////////////

export interface StartPanelProps {
  readonly modelUrl: string;
  readonly onModelUrlChange: (modelUrl: string) => void;
  readonly scenarioSettings: StartScenarioSettings;
  readonly onScenarioSettingsChange: (settings: StartScenarioSettings) => void;
  readonly onLoad: (modelUrl: string) => void | Promise<void>;
  readonly status: string;
  readonly busy: boolean;
}

export interface StartScenarioSettings {
  readonly obstaclesEnabled: boolean;
  readonly obstacleTarget: number;
  readonly batsEnabled: boolean;
  readonly batTarget: number;
  readonly iceCubesEnabled: boolean;
  readonly iceCubeTarget: number;
}

export function StartPanel(props: StartPanelProps) {
  const trimmedModelUrl = props.modelUrl.trim();
  const settings = props.scenarioSettings;

  const updateSettings = (patch: Partial<StartScenarioSettings>): void => {
    props.onScenarioSettingsChange({ ...settings, ...patch });
  };

  return (
    <div className="start-panel glass-panel">
      <div className="panel-eyebrow">Simulation</div>
      <h1 className="start-title">Banana Dash</h1>
      <p className="start-copy">
        Watch four agents and one director run a fast Banana Dash match. A local LLM acts as their decision brain,
        with every choice funneled through the model for consistent monitoring, decision-making, and interaction at
        fast, low latency.
      </p>

      <label className="field start-field">
        <span>Model GGUF</span>
        <input
          type="text"
          value={props.modelUrl}
          disabled={props.busy}
          placeholder="https://.../model.gguf"
          onChange={(e) => props.onModelUrlChange(e.target.value)}
        />
      </label>

      <div className="start-options" aria-label="Scenario options">
        <StartOptionSlider
          label="Obstacles"
          enabled={settings.obstaclesEnabled}
          value={settings.obstacleTarget}
          min={1}
          max={20}
          disabled={props.busy}
          onEnabledChange={(enabled) => updateSettings({ obstaclesEnabled: enabled })}
          onValueChange={(value) => updateSettings({ obstacleTarget: value })}
        />
        <StartOptionSlider
          label="Bats"
          enabled={settings.batsEnabled}
          value={settings.batTarget}
          min={1}
          max={5}
          disabled={props.busy}
          onEnabledChange={(enabled) => updateSettings({ batsEnabled: enabled })}
          onValueChange={(value) => updateSettings({ batTarget: value })}
        />
        <StartOptionSlider
          label="Ice cubes"
          enabled={settings.iceCubesEnabled}
          value={settings.iceCubeTarget}
          min={1}
          max={5}
          disabled={props.busy}
          onEnabledChange={(enabled) => updateSettings({ iceCubesEnabled: enabled })}
          onValueChange={(value) => updateSettings({ iceCubeTarget: value })}
        />
      </div>

      <button
        type="button"
        className="start-load-button"
        disabled={props.busy || trimmedModelUrl.length === 0}
        onClick={() => props.onLoad(trimmedModelUrl)}
      >
        Load
      </button>

      <div className="status start-status">{props.status}</div>
    </div>
  );
}

interface StartOptionSliderProps {
  readonly label: string;
  readonly enabled: boolean;
  readonly value: number;
  readonly min: number;
  readonly max: number;
  readonly disabled: boolean;
  readonly onEnabledChange: (enabled: boolean) => void;
  readonly onValueChange: (value: number) => void;
}

function StartOptionSlider(props: StartOptionSliderProps) {
  const sliderDisabled = props.disabled || !props.enabled;

  return (
    <div className={`start-option-row${props.enabled ? '' : ' disabled'}`}>
      <div className="start-option-header">
        <label className="start-option-toggle">
          <input
            type="checkbox"
            checked={props.enabled}
            disabled={props.disabled}
            onChange={(e) => props.onEnabledChange(e.target.checked)}
          />
          <span>{props.label}</span>
        </label>
        <span className="start-option-value">{props.enabled ? `up to ${props.value}` : 'off'}</span>
      </div>
      <input
        type="range"
        min={props.min}
        max={props.max}
        value={props.value}
        disabled={sliderDisabled}
        onChange={(e) => props.onValueChange(Number(e.target.value))}
      />
    </div>
  );
}
