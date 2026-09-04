'use client';

import React, { useState, useEffect } from 'react';
import styles from './page.module.css';
import Header from './components/Header';
import DecisionForm from './components/DecisionForm';
import DecisionResult from './components/DecisionResult';
import InterventionTable from './components/InterventionTable';
import GuardrailPanel from './components/GuardrailPanel';
import AuditTrail from './components/AuditTrail';
import BatchSimulation from './components/BatchSimulation';
import { checkHealth, evaluateDecision, executeRecovery, getAuditTrail, ApiError } from './lib/api';
import { DecideRequest, DecideResponse, ExecutionResult, AuditEvent } from './lib/types';

export default function Dashboard() {
  const [apiStatus, setApiStatus] = useState<'checking' | 'ok' | 'error'>('checking');
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isExecuting, setIsExecuting] = useState<boolean>(false);
  
  const [currentRequest, setCurrentRequest] = useState<DecideRequest | null>(null);
  const [decision, setDecision] = useState<DecideResponse | null>(null);
  const [executionResult, setExecutionResult] = useState<ExecutionResult | null>(null);
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([]);
  
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    checkHealth()
      .then((data) => {
        setApiStatus('ok');
        setModelLoaded(data.model_loaded);
        refreshAuditTrail();
      })
      .catch(() => {
        setApiStatus('error');
      });
  }, []);

  const refreshAuditTrail = async () => {
    try {
      const events = await getAuditTrail();
      setAuditEvents(events);
    } catch (e) {
      console.error("Failed to fetch audit trail", e);
    }
  };

  const handleEvaluate = async (request: DecideRequest) => {
    setIsLoading(true);
    setErrorMsg(null);
    setDecision(null);
    setExecutionResult(null);
    setCurrentRequest(request);
    
    try {
      const result = await evaluateDecision(request);
      setDecision(result);
    } catch (e: any) {
      if (e instanceof ApiError) {
        setErrorMsg(e.message);
      } else {
        setErrorMsg("An unexpected error occurred while communicating with the backend.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleExecute = async () => {
    if (!currentRequest) return;
    
    setIsExecuting(true);
    setErrorMsg(null);
    
    try {
      const result = await executeRecovery(currentRequest);
      setDecision(result.decision);
      setExecutionResult(result.execution_result);
      await refreshAuditTrail();
    } catch (e: any) {
      if (e instanceof ApiError) {
        setErrorMsg(e.message);
      } else {
        setErrorMsg("An unexpected error occurred during execution.");
      }
    } finally {
      setIsExecuting(false);
    }
  };

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
    <main className={styles.container}>
      <Header apiStatus={apiStatus} modelLoaded={modelLoaded} />
      
      <div className={styles.workflow}>
        
        {/* HERO */}
        <div className={styles.hero}>
          <h1 className={styles.heroTitle}>Recover a failed payment</h1>
          <p className={styles.heroSubtitle}>
            Enter the payment context and let Revora determine the highest-value eligible recovery action.
          </p>
        </div>
        
        {/* PAYMENT CONTEXT */}
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Payment Context</h2>
          <DecisionForm onSubmit={handleEvaluate} isLoading={isLoading} />
          {errorMsg && <div className={styles.errorBox}>{errorMsg}</div>}
        </section>

        {/* DECISION & REASONING */}
        {decision && (
          <>
            <section className={styles.section}>
              <DecisionResult decision={decision} />
            </section>
            
            <section className={styles.section}>
              {Object.keys(decision.guardrail_reasons).length > 0 && (
                <GuardrailPanel guardrailReasons={decision.guardrail_reasons} />
              )}
              <InterventionTable 
                economics={decision.all_economics} 
                selectedIntervention={decision.selected_intervention} 
              />
            </section>

            {/* EXECUTION */}
            <section className={styles.section}>
              {!executionResult ? (
                <div className={styles.executionBlock}>
                  <div className={styles.executeHeader}>Ready to execute</div>
                  <h3 className={styles.executeActionName}>{formatAction(decision.selected_intervention)}</h3>
                  
                  <div className={styles.executeMetrics}>
                    <div className={styles.executeMetric}>
                      <span className={styles.executeMetricLabel}>Expected net recovery</span>
                      <span className={styles.executeMetricValue}>{formatCurrency(decision.selected_expected_net_recovery)}</span>
                    </div>
                    <div className={styles.executeMetric}>
                      <span className={styles.executeMetricLabel}>Incremental value</span>
                      <span className={`${styles.executeMetricValue} ${styles.highlight}`}>
                        {decision.selected_incremental_expected_net_recovery > 0 ? '+' : ''}{formatCurrency(decision.selected_incremental_expected_net_recovery)}
                      </span>
                    </div>
                  </div>

                  <button 
                    className={styles.executeBtn} 
                    onClick={handleExecute} 
                    disabled={isExecuting || decision.selected_intervention === 'no_action' || decision.selected_incremental_expected_net_recovery <= 0}
                  >
                    {isExecuting ? 'Executing...' : 'Execute Recovery'}
                  </button>
                  
                  <div className={styles.simulatedMeta}>
                    <span>SIMULATED</span>
                    <span>·</span>
                    <span>IN-MEMORY</span>
                  </div>
                </div>
              ) : (
                <div className={styles.executionResult}>
                  <h3 className={styles.executionResultTitle}>
                    {executionResult.status === 'executed' ? '✓ Recovery executed' : '⚠ Execution Blocked'}
                  </h3>
                  <p className={styles.executionResultMessage}>{executionResult.message}</p>
                  <div className={styles.executionResultId}>
                    Execution ID: {executionResult.execution_id}
                  </div>
                  <div className={styles.simulatedMeta}>
                    <span>SIMULATED</span>
                    <span>·</span>
                    <span>IN-MEMORY</span>
                  </div>
                </div>
              )}
            </section>
          </>
        )}

        {/* AUDIT TRAIL */}
        {auditEvents.length > 0 && (
          <section className={styles.section}>
            <AuditTrail events={auditEvents} />
          </section>
        )}

        {/* MEASUREMENT */}
        <section className={styles.section} style={{ marginTop: 'var(--space-12)' }}>
          <BatchSimulation />
        </section>
        
      </div>
    </main>
  );
}
