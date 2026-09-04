import React from 'react';
import styles from './DecisionResult.module.css';
import { DecideResponse } from '../lib/types';

interface DecisionResultProps {
  decision: DecideResponse;
}

export default function DecisionResult({ decision }: DecisionResultProps) {
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  };

  const formatPercent = (val: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'percent',
      minimumFractionDigits: 1,
      maximumFractionDigits: 1
    }).format(val);
  };
  
  const formatAction = (action: string) => {
    return action.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  };

  const isPositiveInc = decision.selected_incremental_expected_net_recovery > 0;

  return (
    <div className={styles.container}>
      <div className={styles.recommendationHeader}>
        <div className={styles.recommendationLabel}>Revora recommends</div>
      </div>
      
      <div className={styles.recommendationBlock}>
        <h3 className={styles.actionName}>
          {formatAction(decision.selected_intervention)}
        </h3>
        
        <div className={styles.primaryMetricValue}>
          {isPositiveInc ? '+' : ''}{formatCurrency(decision.selected_incremental_expected_net_recovery)}
        </div>
        <div className={styles.primaryMetricLabel}>incremental expected net recovery</div>
      </div>

      <div className={styles.metricsGrid}>
        <div className={styles.metricBlock}>
          <span className={styles.metricValue}>{formatPercent(decision.selected_probability)}</span>
          <span className={styles.metricLabel}>probability</span>
        </div>
        <div className={styles.metricBlock}>
          <span className={styles.metricValue}>{formatCurrency(decision.selected_expected_recovered_amount)}</span>
          <span className={styles.metricLabel}>expected gross recovery</span>
        </div>
        <div className={styles.metricBlock}>
          <span className={styles.metricValue}>{formatCurrency(decision.intervention_cost)}</span>
          <span className={styles.metricLabel}>intervention cost</span>
        </div>
        <div className={styles.metricBlock}>
          <span className={styles.metricValue}>{formatCurrency(decision.selected_expected_net_recovery)}</span>
          <span className={styles.metricLabel}>expected net recovery</span>
        </div>
      </div>
      
      <div className={styles.reasoningBlock}>
        <div className={styles.reasoningLabel}>Why this decision</div>
        <p className={styles.reasoningText}>
          {decision.decision_reason}
        </p>
      </div>
    </div>
  );
}
