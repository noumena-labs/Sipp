import type { ReactNode } from 'react';

interface MetricCardProps {
  readonly label: string;
  readonly tone?: 'default' | 'ok' | 'warn';
  readonly value: ReactNode;
}

export function MetricCard({ label, value, tone = 'default' }: MetricCardProps) {
  return (
    <div className={`metric-card ${tone !== 'default' ? `metric-card-${tone}` : ''}`}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}
