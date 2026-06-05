import type { ReactNode } from 'react';

type StatusTone = 'neutral' | 'ok' | 'warn' | 'danger' | 'info';

interface StatusBadgeProps {
  readonly children: ReactNode;
  readonly tone?: StatusTone;
}

export function StatusBadge({ children, tone = 'neutral' }: StatusBadgeProps) {
  return <span className={`status-badge status-badge-${tone}`}>{children}</span>;
}
