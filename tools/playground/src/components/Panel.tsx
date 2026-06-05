import type { ReactNode } from 'react';

interface PanelProps {
  readonly actions?: ReactNode;
  readonly children: ReactNode;
  readonly className?: string;
  readonly title: string;
}

export function Panel({ actions, children, className = '', title }: PanelProps) {
  return (
    <section className={`panel ${className}`.trim()}>
      <div className="panel-header">
        <h2>{title}</h2>
        {actions == null ? null : <div className="panel-actions">{actions}</div>}
      </div>
      {children}
    </section>
  );
}
