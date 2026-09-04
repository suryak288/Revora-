import React, { useState } from 'react';
import styles from './DecisionForm.module.css';
import { DecideRequest } from '../lib/types';

interface DecisionFormProps {
  onSubmit: (request: DecideRequest) => void;
  isLoading: boolean;
}

export default function DecisionForm({ onSubmit, isLoading }: DecisionFormProps) {
  const [formData, setFormData] = useState<any>({
    payment_amount: '1500',
    currency: "INR",
    payment_method: "card",
    payment_method_category: "card",
    customer_tenure_days: '180',
    previous_successful_payments: '5',
    previous_failed_payments: '0',
    failure_reason: "insufficient_funds",
    time_since_failure_hours: '2',
    retry_count: '0',
    is_recurring: true,
    merchant_segment: "small_business"
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    let parsedValue: any = value;
    
    if (type === 'number') {
      parsedValue = value === '' ? '' : value;
    } else if (type === 'checkbox') {
      parsedValue = (e.target as HTMLInputElement).checked;
    }
    
    setFormData((prev: any) => ({
      ...prev,
      [name]: parsedValue
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const request: DecideRequest = {
      ...formData,
      payment_amount: Number(formData.payment_amount) || 0,
      customer_tenure_days: Number(formData.customer_tenure_days) || 0,
      previous_successful_payments: Number(formData.previous_successful_payments) || 0,
      previous_failed_payments: Number(formData.previous_failed_payments) || 0,
      time_since_failure_hours: Number(formData.time_since_failure_hours) || 0,
      retry_count: Number(formData.retry_count) || 0,
    };
    onSubmit(request);
  };

  return (
    <form className={styles.form} onSubmit={handleSubmit}>
      
      <fieldset className={styles.fieldset}>
        <legend className={styles.legend}>Payment</legend>
        <div className={styles.grid}>
          <div className={styles.field}>
            <label htmlFor="payment_amount">Amount</label>
            <div className={styles.inputGroup}>
              <span className={styles.prefix}>{formData.currency}</span>
              <input 
                type="number" 
                id="payment_amount" 
                name="payment_amount" 
                value={formData.payment_amount} 
                onChange={handleChange} 
                min="1" 
                step="0.01"
                required 
              />
            </div>
          </div>
          
          <div className={styles.field}>
            <label htmlFor="payment_method">Method</label>
            <select id="payment_method" name="payment_method" value={formData.payment_method} onChange={handleChange}>
              <option value="card">Card</option>
              <option value="upi">UPI</option>
              <option value="netbanking">Netbanking</option>
              <option value="wallet">Wallet</option>
              <option value="emi">EMI</option>
            </select>
          </div>

          <div className={styles.field}>
            <div className={styles.checkboxContainer}>
              <input type="checkbox" id="is_recurring" name="is_recurring" checked={formData.is_recurring} onChange={handleChange} />
              <label htmlFor="is_recurring">Recurring Payment</label>
            </div>
          </div>
        </div>
      </fieldset>

      <fieldset className={styles.fieldset}>
        <legend className={styles.legend}>Failure</legend>
        <div className={styles.grid}>
          <div className={styles.field}>
            <label htmlFor="failure_reason">Reason</label>
            <select id="failure_reason" name="failure_reason" value={formData.failure_reason} onChange={handleChange}>
              <option value="insufficient_funds">Insufficient Funds</option>
              <option value="card_declined">Card Declined</option>
              <option value="network_error">Network Error</option>
              <option value="expired_card">Expired Card</option>
              <option value="authentication_failed">Authentication Failed</option>
              <option value="bank_unavailable">Bank Unavailable</option>
              <option value="payment_method_error">Payment Method Error</option>
            </select>
          </div>
          
          <div className={styles.field}>
            <label htmlFor="time_since_failure_hours">Time Since Failure (Hours)</label>
            <input type="number" id="time_since_failure_hours" name="time_since_failure_hours" value={formData.time_since_failure_hours} onChange={handleChange} min="0" />
          </div>

          <div className={styles.field}>
            <label htmlFor="retry_count">Retries</label>
            <input type="number" id="retry_count" name="retry_count" value={formData.retry_count} onChange={handleChange} min="0" />
          </div>
        </div>
      </fieldset>

      <fieldset className={styles.fieldset}>
        <legend className={styles.legend}>Customer</legend>
        <div className={styles.grid}>
          <div className={styles.field}>
            <label htmlFor="customer_tenure_days">Tenure (Days)</label>
            <input type="number" id="customer_tenure_days" name="customer_tenure_days" value={formData.customer_tenure_days} onChange={handleChange} min="0" />
          </div>

          <div className={styles.field}>
            <label htmlFor="previous_successful_payments">Previous Successes</label>
            <input type="number" id="previous_successful_payments" name="previous_successful_payments" value={formData.previous_successful_payments} onChange={handleChange} min="0" />
          </div>

          <div className={styles.field}>
            <label htmlFor="previous_failed_payments">Previous Failures</label>
            <input type="number" id="previous_failed_payments" name="previous_failed_payments" value={formData.previous_failed_payments} onChange={handleChange} min="0" />
          </div>
        </div>
      </fieldset>

      <fieldset className={styles.fieldset}>
        <legend className={styles.legend}>Merchant</legend>
        <div className={styles.grid}>
          <div className={styles.field}>
            <label htmlFor="merchant_segment">Segment</label>
            <select id="merchant_segment" name="merchant_segment" value={formData.merchant_segment} onChange={handleChange}>
              <option value="startup">Startup</option>
              <option value="small_business">Small Business</option>
              <option value="mid_market">Mid Market</option>
              <option value="enterprise">Enterprise</option>
            </select>
          </div>
        </div>
      </fieldset>
      
      <div className={styles.actions}>
        <button type="submit" className={styles.submitBtn} disabled={isLoading}>
          {isLoading ? 'Evaluating...' : 'Evaluate Recovery'}
        </button>
      </div>
    </form>
  );
}
