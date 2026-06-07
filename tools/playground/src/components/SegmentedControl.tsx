export interface SegmentedOption<T extends string> {
  readonly disabled?: boolean;
  readonly label: string;
  readonly title?: string;
  readonly value: T;
}

interface SegmentedControlProps<T extends string> {
  readonly ariaLabel: string;
  readonly disabled?: boolean;
  readonly onChange: (value: T) => void;
  readonly options: readonly SegmentedOption<T>[];
  readonly value: T;
}

export function SegmentedControl<T extends string>({
  ariaLabel,
  disabled = false,
  onChange,
  options,
  value,
}: SegmentedControlProps<T>) {
  return (
    <div className="segmented-control" role="group" aria-label={ariaLabel}>
      {options.map((option) => (
        <button
          className={`segmented-control__item ${value === option.value ? 'active' : ''}`}
          disabled={disabled || option.disabled}
          key={option.value}
          onClick={() => onChange(option.value)}
          title={option.title}
          type="button"
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
