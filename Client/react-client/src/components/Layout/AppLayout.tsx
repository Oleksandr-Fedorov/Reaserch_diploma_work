/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import './styles.scss'; 

interface AppLayoutProps {
  children: React.ReactNode;
  navSlot?: React.ReactNode;
  onReset?: () => void;
  hasActiveAnalysis?: boolean;
}

export default function AppLayout({ children, navSlot, onReset, hasActiveAnalysis }: AppLayoutProps) {
  return (
    <div className="aegis-layout">
      {/* Zone A: Top Navigation Bar */}
      <header className="aegis-layout__header">
        <div className="aegis-layout__brand">
          <div className="aegis-layout__logo-mark">
            <span>Æ</span>
          </div>
          <div className="aegis-layout__logo-text">
            <span 
              className="aegis-layout__title"
              onClick={onReset}
            >
              AEGIS // VISION
            </span>
            <span className="aegis-layout__subtitle">
              Forensic Image Neural Analyzer
            </span>
          </div>
        </div>

        {/* Center: Navigation options */}
        <nav className="aegis-layout__nav">
          <span className="aegis-layout__nav-item aegis-layout__nav-item--active">
            Workspace
          </span>
          <span 
            className="aegis-layout__nav-item"
            onClick={onReset}
          >
            Archive / Samples
          </span>
          <span className="aegis-layout__nav-item">
            System Status
          </span>
        </nav>

        {/* Dynamic header widgets */}
        <div className="aegis-layout__widgets">
          {hasActiveAnalysis && (
            <div className="aegis-layout__status-badge">
              <span className="aegis-layout__status-dot" />
              <span>SESSION ACTIVE</span>
            </div>
          )}
          {navSlot}
        </div>
      </header>

      {/* Main View Area */}
      <div className="aegis-layout__main">
        {children}
      </div>
    </div>
  );
}