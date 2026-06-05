import React from 'react';

interface MetricCardProps {
  label: string;
  value: React.ReactNode;
  tone?: 'default' | 'ok' | 'warn';
}

export function MetricCard({ label, value, tone = 'default' }: MetricCardProps) {
  return (
    <div className={`metric-card ${tone !== 'default' ? `metric-card-${tone}` : ''}`}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}
