import React from 'react';
import styles from './AuditTrail.module.css';
import { AuditEvent } from '../lib/types';

interface AuditTrailProps {
  events: AuditEvent[];
}

export default function AuditTrail({ events }: AuditTrailProps) {
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  };
  
  const formatAction = (action: string) => {
    return action.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h2 className={styles.title}>Audit trail</h2>
      </div>
      
      {events.length === 0 ? (
        <div className={styles.emptyState}>No audit events found.</div>
      ) : (
        <div className={styles.timeline}>
          {events.map((event, idx) => {
            const date = new Date(event.executed_at);
            const timeString = date.toLocaleTimeString('en-US', { hour12: false });
            
            return (
              <div key={idx} className={styles.timelineEvent}>
                <div className={styles.eventMeta}>
                  <span className={styles.timestamp}>{timeString}</span>
                  <span className={styles.actionBadge}>{formatAction(event.selected_intervention)}</span>
                  <span className={`${styles.statusIndicator} ${styles['status' + event.execution_status.charAt(0).toUpperCase() + event.execution_status.slice(1)]}`}>
                    {event.execution_status}
                  </span>
                </div>
                
                <p className={styles.eventMessage}>{event.execution_message}</p>
                
                {(event.expected_net_recovery !== undefined && event.incremental_expected_net_recovery !== undefined) && (
                  <div className={styles.eventFinancials}>
                    <span>Expected Net <strong>{formatCurrency(event.expected_net_recovery)}</strong></span>
                    <span className={event.incremental_expected_net_recovery > 0 ? styles.highlight : ''}>
                      Incremental <strong>{event.incremental_expected_net_recovery > 0 ? '+' : ''}{formatCurrency(event.incremental_expected_net_recovery)}</strong>
                    </span>
                  </div>
                )}
                
                <div className={styles.executionId}>{event.execution_id}</div>

                {(event.source || event.execution_mode) && (
                  <div className={styles.eventBadges}>
                    {event.source === 'razorpay_test' && (
                      <span className={styles.badgeRazorpay}>RAZORPAY TEST MODE</span>
                    )}
                    {event.source === 'synthetic' && (
                      <span className={styles.badgeSynthetic}>OFFLINE SYNTHETIC</span>
                    )}
                    {event.execution_mode === 'razorpay_test_api' && (
                      <span className={styles.badgeApi}>TEST API</span>
                    )}
                    {event.execution_mode === 'simulated' && (
                      <span className={styles.badgeSimulated}>SIMULATED</span>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
