import React from 'react';
import styles from './EvaluationPanel.module.css';

export default function EvaluationPanel() {
  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <h3 className={styles.title}>Does the policy create value?</h3>
        <span className={styles.badge}>Offline Evaluation</span>
      </div>
      
      <p className={styles.description}>
        These metrics are statistically defensible IPS (Inverse Propensity Score) estimates derived from the held-out test batch (Milestone 5). They represent projected policy performance, not live production data.
      </p>

      <div className={styles.statsContainer}>
        <div className={styles.statBox}>
          <span className={styles.statLabel}>Test Batch Size</span>
          <span className={styles.statValue}>1,000 cases</span>
        </div>
        
        <div className={styles.statBox}>
          <span className={styles.statLabel}>No-Action Estimated Net</span>
          <span className={styles.statValue}>~45.9%</span>
        </div>
        
        <div className={styles.statBox}>
          <span className={styles.statLabel}>RecoveryOS Estimated Net</span>
          <span className={`${styles.statValue} ${styles.highlight}`}>~53.2%</span>
        </div>
        
        <div className={styles.statBox}>
          <span className={styles.statLabel}>Estimated Incremental Value</span>
          <span className={`${styles.statValue} ${styles.success}`}>+7.3 pts</span>
        </div>
      </div>
    </div>
  );
}
