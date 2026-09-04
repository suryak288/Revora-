import React, { useState } from 'react';
import styles from './BatchSimulation.module.css';
import { simulateBatch, ApiError } from '../lib/api';
import { BatchSimulationReport } from '../lib/types';

export default function BatchSimulation() {
  const [batchSize, setBatchSize] = useState<any>('1000');
  const [seed, setSeed] = useState<any>('2026');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<BatchSimulationReport | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleSimulate = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setErrorMsg(null);
    setResult(null);

    const parsedBatchSize = Number(batchSize) || 1000;
    const parsedSeed = Number(seed) || 2026;

    try {
      const data = await simulateBatch(parsedBatchSize, parsedSeed);
      setResult(data);
    } catch (e: any) {
      if (e instanceof ApiError) {
        setErrorMsg(e.message);
      } else {
        setErrorMsg("An unexpected error occurred during batch simulation.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const formatCurrencyCompact = (amount: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      notation: 'compact',
      maximumFractionDigits: 1
    }).format(amount);
  };
  
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
      minimumFractionDigits: 0,
      maximumFractionDigits: 1
    }).format(val);
  };

  return (
    <div className={styles.container}>
      <div>
        <span className={styles.disclaimer}>Offline synthetic evaluation</span>
        <h2 style={{ fontSize: '1.125rem', fontWeight: 500, margin: 0, color: 'var(--text-main)' }}>Offline Evaluation</h2>
      </div>

      <form className={styles.form} onSubmit={handleSimulate}>
        <div className={styles.field}>
          <label className={styles.label} htmlFor="batch_size">Batch Size</label>
          <input
            className={styles.input}
            type="number"
            id="batch_size"
            value={batchSize}
            onChange={(e) => setBatchSize(e.target.value)}
            min="10"
            max="10000"
          />
        </div>
        <div className={styles.field}>
          <label className={styles.label} htmlFor="seed">Seed</label>
          <input
            className={styles.input}
            type="number"
            id="seed"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
          />
        </div>
        <button type="submit" className={styles.button} disabled={isLoading}>
          {isLoading ? 'Running...' : 'Run Evaluation'}
        </button>
      </form>

      {errorMsg && <div className={styles.error}>{errorMsg}</div>}

      {result && (
        <div className={styles.results}>
          <div className={styles.tableContainer}>
            <table className={styles.comparisonTable}>
              <thead>
                <tr>
                  <th>Metric</th>
                  <th className={styles.primaryCol}>RecoveryOS</th>
                  <th>No-Action</th>
                  <th>Random</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Recovery rate</td>
                  <td className={styles.primaryCol}>{formatPercent(result.recovery_os_metrics.recovery_rate)}</td>
                  <td>{formatPercent(result.no_action_metrics.recovery_rate)}</td>
                  <td>{formatPercent(result.random_metrics.recovery_rate)}</td>
                </tr>
                <tr>
                  <td>Recovered cases</td>
                  <td className={styles.primaryCol}>{result.recovery_os_metrics.recovered_cases}</td>
                  <td>{result.no_action_metrics.recovered_cases}</td>
                  <td>{result.random_metrics.recovered_cases}</td>
                </tr>
                <tr>
                  <td>Simulated recovered</td>
                  <td className={styles.primaryCol}>{formatCurrencyCompact(result.recovery_os_metrics.total_recovered_amount)}</td>
                  <td>{formatCurrencyCompact(result.no_action_metrics.total_recovered_amount)}</td>
                  <td>{formatCurrencyCompact(result.random_metrics.total_recovered_amount)}</td>
                </tr>
                <tr>
                  <td>Intervention cost</td>
                  <td className={styles.primaryCol}>{formatCurrencyCompact(result.recovery_os_metrics.total_intervention_cost)}</td>
                  <td>{formatCurrencyCompact(result.no_action_metrics.total_intervention_cost)}</td>
                  <td>{formatCurrencyCompact(result.random_metrics.total_intervention_cost)}</td>
                </tr>
                <tr>
                  <td>Net recovery</td>
                  <td className={styles.primaryCol}>{formatCurrencyCompact(result.recovery_os_metrics.total_net_recovery)}</td>
                  <td>{formatCurrencyCompact(result.no_action_metrics.total_net_recovery)}</td>
                  <td>{formatCurrencyCompact(result.random_metrics.total_net_recovery)}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className={styles.deltas}>
            <div className={styles.deltaItem}>
              <div className={styles.deltaLabel}>Recovery rate lift</div>
              <div className={`${styles.deltaValue} ${result.comparison.recovery_rate_lift_vs_no_action > 0 ? styles.highlight : ''}`}>
                {result.comparison.recovery_rate_lift_vs_no_action > 0 ? '+' : ''}
                {(result.comparison.recovery_rate_lift_vs_no_action * 100).toFixed(1)} pts
              </div>
            </div>
            <div className={styles.deltaItem}>
              <div className={styles.deltaLabel}>Incremental net recovery</div>
              <div className={`${styles.deltaValue} ${result.comparison.incremental_net_recovery_vs_no_action > 0 ? styles.highlight : ''}`}>
                {result.comparison.incremental_net_recovery_vs_no_action > 0 ? '+' : ''}
                {formatCurrency(result.comparison.incremental_net_recovery_vs_no_action)}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
