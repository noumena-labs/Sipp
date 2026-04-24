//////////////////////////////////////////////////////////////////////////////
//
// components/StartPanel.tsx
//
// - Pre-simulation start screen. Frames the investor-facing local inference
//   thesis concisely and exposes the demo configuration controls.
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
      <div className="panel-eyebrow">Noumena Labs</div>
      <h1 className="start-title">High-Performance Local Inference Demo</h1>
      <p className="start-lead">
        AI inference is expensive, slow, and difficult to tune for real-time interactive systems. 
        We are making a distributed, local-first open source library so developers can run inference cheaply, quickly, and continuously across devices.
        This forms a core piece of our technology stack enabling hybrid AI computing for delegated presence. 
      </p>

      <div className="start-story-grid">
        <section className="start-story-card">
          <div className="start-story-label">Why It Matters</div>
          <p className="start-story-copy">
            Interfaces are becoming adaptive: users expect software to understand their task, respond in real time, 
            and surface the right action without convoluted workflows. 
            Current AI infrastructure do not allow this interactive speed and privacy-first.  
          </p>
        </section>

        <section className="start-story-card">
          <div className="start-story-label">What We Built</div>
          <p className="start-story-copy">
            We are building a ground-up WebGPU inference engine for running local LLMs responsively and interactively.
            With runtime harnesses, scaffolding, and developer tools, it gives teams the foundation to build the next generation of real-time AI-native user experiences.
          </p>
        </section>
      </div>

      <div className="start-config-head">
        <div className="start-config-title">PoC: Banana Dash</div>
        <div className="start-config-copy">A demo consisting of 4 brains and 1 judge as the that fight for bananas.</div>
      </div>

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
        Load Demo
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
