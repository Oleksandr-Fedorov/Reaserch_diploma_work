/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect } from 'react';
import { Terminal as TerminalIcon, FileText, CheckCircle,ShieldAlert } from 'lucide-react';
import type { AnalysisResults } from '../../../types/types';

import './styles.scss';

interface AnalyticsPanelProps {
  results: AnalysisResults | null;
  isAnalyzing: boolean;
  onExportReport: () => void;
}

export default function AnalyticsPanel({ results, isAnalyzing, onExportReport }: AnalyticsPanelProps) {
  const [animatedScore, setAnimatedScore] = useState<number>(results ? results.probability : 0);

  // Linear count-up animation over exactly 600ms
  useEffect(() => {
    if (!results) {
      return;
    }

    const start = 0;
    const end = results.probability;
    const duration = 600; // ms
    const startTime = performance.now();

    let animationFrameId: number;

    const animate = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Linear calculation
      const currentVal = start + (end - start) * progress;
      setAnimatedScore(Number(currentVal.toFixed(1)));

      if (progress < 1) {
        animationFrameId = requestAnimationFrame(animate);
      } else {
        setAnimatedScore(end);
      }
    };

    animationFrameId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [results]);

  if (isAnalyzing) {
    return (
      <aside className="aegis-analytics">
        <div className="aegis-analytics__state-overlay">
          <TerminalIcon className="aegis-analytics__icon-pulse" />
          <div className="aegis-analytics__state-text">
            <span className="aegis-analytics__code-highlight">// RECALCULATING DATA MATRIX</span>
            <span className="aegis-analytics__subtext">Parsing headers and checking hash pools</span>
          </div>
        </div>
      </aside>
    );
  }

  if (!results) {
    return (
      <aside className="aegis-analytics">
        <div className="aegis-analytics__state-overlay">
          <div className="aegis-analytics__icon-circle">
            <TerminalIcon size={20} />
          </div>
          <div className="aegis-analytics__state-text">
            <span className="aegis-analytics__label">// NO MODEL LOADED</span>
            <p className="aegis-analytics__description">
              Load an image target from the dropzone or select a sample sandbox target to populate the forensic dashboard.
            </p>
          </div>
        </div>
      </aside>
    );
  }

  const isManipulated = results.verdict === 'MANIPULATION_DETECTED';
  const statusModifier = isManipulated ? 'fake' : 'auth';

  return (
    <aside className="aegis-analytics">
      
      {/* Scrollable analysis metrics */}
      <div className="aegis-analytics__content">
        
        {/* Block 1: System Verdict */}
        <div className="aegis-verdict">
          <span className="aegis-section-title">System Verdict</span>
          
          <div className="aegis-verdict__header">
            {isManipulated ? (
              <ShieldAlert className="aegis-text-fake" size={20} />
            ) : (
              <CheckCircle className="aegis-text-auth" size={20} />
            )}
            <h2 className={`aegis-verdict__title aegis-text-${statusModifier}`}>
              {isManipulated ? 'Manipulation Detected' : 'Authentic / Unaltered'}
            </h2>
          </div>
          
          <div className="aegis-verdict__score-block">
            <div className={`aegis-verdict__score aegis-text-${statusModifier}`}>
              {animatedScore}% 
              <span className="aegis-verdict__score-label">
                {isManipulated ? 'Probability of tampering' : 'Confidence authentic'}
              </span>
            </div>
            <span className="aegis-verdict__hash">
              HASH: {results.fileHash}
            </span>
          </div>
        </div>

        {/* Block 2: AI Confidence Metrics */}
        <div className="aegis-metrics">
          <span className="aegis-section-title">AI Confidence Metrics</span>

          <div className="aegis-metrics__list">
            {/* Metric 1 */}
            <div className="aegis-metrics__item">
              <div className="aegis-metrics__label-row">
                <span>ELA Anomaly.....................</span>
                <span className={`aegis-metrics__value aegis-text-${statusModifier} ${isManipulated ? 'aegis-font-bold' : ''}`}>
                  [{results.metrics.elaAnomaly.toFixed(1)}%]
                </span>
              </div>
              <div className="aegis-metrics__track">
                <div 
                  className={`aegis-metrics__fill aegis-bg-${statusModifier}`}
                  style={{ width: `${results.metrics.elaAnomaly}%` }}
                />
              </div>
            </div>

            {/* Metric 2 */}
            <div className="aegis-metrics__item">
              <div className="aegis-metrics__label-row">
                <span>Noise Inconsistency...........</span>
                <span className={`aegis-metrics__value aegis-text-${statusModifier} ${isManipulated ? 'aegis-font-bold' : ''}`}>
                  [{results.metrics.noiseInconsistency.toFixed(1)}%]
                </span>
              </div>
              <div className="aegis-metrics__track">
                <div 
                  className={`aegis-metrics__fill aegis-bg-${statusModifier}`}
                  style={{ width: `${results.metrics.noiseInconsistency}%` }}
                />
              </div>
            </div>

            {/* Metric 3 */}
            <div className="aegis-metrics__item">
              <div className="aegis-metrics__label-row">
                <span>Edge Artifacts.................</span>
                <span className={`aegis-metrics__value aegis-text-${statusModifier} ${isManipulated ? 'aegis-font-bold' : ''}`}>
                  [{results.metrics.edgeArtifacts.toFixed(1)}%]
                </span>
              </div>
              <div className="aegis-metrics__track">
                <div 
                  className={`aegis-metrics__fill aegis-bg-${statusModifier}`}
                  style={{ width: `${results.metrics.edgeArtifacts}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Block 3: Metadata & EXIF Analysis Terminal */}
        <div className="aegis-terminal">
          <div className="aegis-terminal__header">
            <span className="aegis-section-title">Metadata & EXIF Analysis</span>
            <span className="aegis-terminal__badge">HEX READER ACTIVE</span>
          </div>

          <div className="aegis-terminal__window">
            <div className="aegis-terminal__row aegis-terminal__row--header">
              <span>PARAMETER</span>
              <span>METRICS VALUE</span>
            </div>

            <div className="aegis-terminal__row">
              <span className="aegis-terminal__label">Camera Make:</span>
              <span className="aegis-terminal__value">{results.exif.make || 'GENERIC'}</span>
            </div>

            <div className="aegis-terminal__row">
              <span className="aegis-terminal__label">Sensor Model:</span>
              <span className="aegis-terminal__value">{results.exif.model || 'UNKNOWN'}</span>
            </div>

            <div className="aegis-terminal__row">
              <span className="aegis-terminal__label">Software Codec:</span>
              <span className={`aegis-terminal__value ${
                isManipulated && (results.exif.software.toLowerCase().includes('photoshop') || results.exif.software.toLowerCase().includes('gimp') || results.exif.software.toLowerCase().includes('paint'))
                  ? 'aegis-text-fake aegis-font-bold'
                  : ''
              }`}>
                {results.exif.software || 'NATIVE ENCODER'}
              </span>
            </div>

            <div className="aegis-terminal__row">
              <span className="aegis-terminal__label">File Size:</span>
              <span className="aegis-terminal__value">{results.exif.fileSize}</span>
            </div>

            <div className="aegis-terminal__row">
              <span className="aegis-terminal__label">Dimensions:</span>
              <span className="aegis-terminal__value">{results.exif.resolution}</span>
            </div>

            <div className="aegis-terminal__row">
              <span className="aegis-terminal__label">Color Space:</span>
              <span className="aegis-terminal__value">{results.exif.colorSpace}</span>
            </div>

            <div className="aegis-terminal__row">
              <span className="aegis-terminal__label">Original UTC:</span>
              <span className="aegis-terminal__value">{results.exif.originalTime}</span>
            </div>

            <div className="aegis-terminal__row">
              <span className="aegis-terminal__label">Modification UTC:</span>
              <span className={`aegis-terminal__value ${
                isManipulated && results.exif.originalTime !== results.exif.modTime
                  ? 'aegis-text-fake aegis-font-bold'
                  : ''
              }`}>
                {results.exif.modTime}
              </span>
            </div>

            <div className="aegis-terminal__hash-block">
              <span className="aegis-terminal__hash-label">INTEGRITY SHA-256 CHECK:</span>
              <span className="aegis-terminal__hash-value">{results.fileHash}</span>
            </div>
          </div>
        </div>

      </div>

      {/* Block 4: Action Footer */}
      <button
        onClick={onExportReport}
        className="aegis-analytics__export-btn"
        title="Export Report File"
      >
        <FileText size={13} className="aegis-text-accent" />
        Export Forensic Report (.PDF)
      </button>

    </aside>
  );
}