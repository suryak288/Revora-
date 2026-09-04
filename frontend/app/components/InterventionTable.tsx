import React from 'react';
import styles from './InterventionTable.module.css';
import { InterventionEconomics } from '../lib/types';

interface InterventionTableProps {
  economics: InterventionEconomics[];
  selectedIntervention: string;
}

export default function InterventionTable({ economics, selectedIntervention }: InterventionTableProps) {
  
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

  return (
    <div className={styles.container}>
      <h2 className={styles.title}>Other recovery options considered</h2>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Action</th>
            <th>Probability</th>
            <th>Net</th>
            <th>Incremental</th>
          </tr>
        </thead>
        <tbody>
          {economics.map(eco => {
            const isSelected = eco.intervention === selectedIntervention;
            const isExcluded = !eco.eligible;
            
            let rowClass = '';
            if (isSelected) rowClass = styles.rowRecommended;
            if (isExcluded) rowClass = styles.rowExcluded;
            
            return (
              <tr key={eco.intervention} className={rowClass}>
                <td className={styles.actionName}>
                  {formatAction(eco.intervention)}
                </td>
                
                {isExcluded ? (
                  <>
                    <td className={styles.numeric}>—</td>
                    <td className={styles.numeric}>—</td>
                    <td className={styles.statusLabel}>
                      <span className={styles.statusExcluded}>EXCLUDED</span>
                    </td>
                  </>
                ) : (
                  <>
                    <td className={styles.numeric}>{formatPercent(eco.predicted_probability)}</td>
                    <td className={styles.numeric}>{formatCurrency(eco.expected_net_recovery)}</td>
                    <td className={`${styles.numeric} ${eco.incremental_expected_net_recovery > 0 ? styles.positive : ''}`}>
                      {eco.incremental_expected_net_recovery > 0 ? '+' : ''}
                      {formatCurrency(eco.incremental_expected_net_recovery)}
                    </td>
                  </>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
