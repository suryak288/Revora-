import React from 'react';
import styles from './GuardrailPanel.module.css';

interface GuardrailPanelProps {
  guardrailReasons: Record<string, string[]>;
}

export default function GuardrailPanel({ guardrailReasons }: GuardrailPanelProps) {
  const formatAction = (action: string) => {
    return action.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <span className={styles.headerIcon}>⚠️</span>
        <h4 className={styles.title}>Policy Exclusions Applied</h4>
      </div>
      
      <div className={styles.exclusionList}>
        {Object.entries(guardrailReasons).map(([action, reasons]) => (
          <div key={action} className={styles.exclusionItem}>
            <span className={styles.actionName}>{formatAction(action)}</span>
            <ul className={styles.reasonList}>
              {reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}
