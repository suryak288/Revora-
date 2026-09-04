import React from 'react';
import styles from './Header.module.css';

interface HeaderProps {
  apiStatus: 'checking' | 'ok' | 'error';
  modelLoaded: boolean;
}

export default function Header({ apiStatus, modelLoaded }: HeaderProps) {
  return (
    <header className={styles.header}>
      <div className={styles.logoSection}>
        <h1 className={styles.title}>Revora</h1>
        <span className={styles.subtitle}>AI Revenue Recovery</span>
      </div>
      
      <div className={styles.statusSection}>
        {apiStatus === 'ok' ? (
          <div className={`${styles.statusBadge} ${styles.success}`}>
            API Connected
          </div>
        ) : apiStatus === 'error' ? (
           <div className={`${styles.statusBadge} ${styles.error}`}>
            API Offline
          </div>
        ) : (
           <div className={`${styles.statusBadge} ${styles.warning}`}>
            Connecting...
          </div>
        )}
      </div>
    </header>
  );
}
