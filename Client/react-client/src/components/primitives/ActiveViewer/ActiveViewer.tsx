import React, { useState, useRef, useEffect } from 'react';
import { ZoomIn, ZoomOut, Eye, Layers, Flame, Split, Sliders } from 'lucide-react';
import type { ViewMode, AnalysisResults } from '../../../types/types';
import './styles.scss';

interface WorkspaceActiveViewerProps {
  results: AnalysisResults;
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
  isAnalyzing: boolean;
}

export default function WorkspaceActiveViewer({
  results,
  viewMode,
  setViewMode,
  isAnalyzing
}: WorkspaceActiveViewerProps) {
  const [zoom, setZoom] = useState<number>(100);
  const [opacity, setOpacity] = useState<number>(100);
  const [splitRatio, setSplitRatio] = useState<number>(50);
  const [isDraggingSplit, setIsDraggingSplit] = useState<boolean>(false);
  const [tooltip, setTooltip] = useState({
    x: 0, y: 0, imgX: 0, imgY: 0, level: 'NORMAL', visible: false
  });

  const containerRef = useRef<HTMLDivElement>(null);
  const splitContainerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  const handleZoomIn = () => setZoom((prev) => Math.min(prev + 25, 400));
  const handleZoomOut = () => setZoom((prev) => Math.max(prev - 25, 50));

  const handleSplitMove = (clientX: number) => {
    if (!splitContainerRef.current) return;
    const rect = splitContainerRef.current.getBoundingClientRect();
    const relativeX = clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (relativeX / rect.width) * 100));
    setSplitRatio(percentage);
  };

  const handleMouseDownSplit = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDraggingSplit(true);
  };

  useEffect(() => {
    const handleMouseMoveGlobal = (e: MouseEvent) => {
      if (isDraggingSplit) handleSplitMove(e.clientX);
    };
    const handleMouseUpGlobal = () => setIsDraggingSplit(false);

    if (isDraggingSplit) {
      window.addEventListener('mousemove', handleMouseMoveGlobal);
      window.addEventListener('mouseup', handleMouseUpGlobal);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMoveGlobal);
      window.removeEventListener('mouseup', handleMouseUpGlobal);
    };
  }, [isDraggingSplit]);

  const handleMouseMoveImage = (e: React.MouseEvent<HTMLDivElement>) => {
    if (viewMode !== 'AI_HEATMAP' || !imageRef.current) {
      setTooltip((t) => ({ ...t, visible: false }));
      return;
    }

    const rect = imageRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {
      const naturalWidth = imageRef.current.naturalWidth || parseInt(results.exif.resolution.split(' x ')[0]);
      const naturalHeight = imageRef.current.naturalHeight || parseInt(results.exif.resolution.split(' x ')[1]);
      
      const imgX = Math.round((x / rect.width) * naturalWidth);
      const imgY = Math.round((y / rect.height) * naturalHeight);

      let anomalyLevel = 'NORMAL';
      if (results.verdict === 'MANIPULATION_DETECTED') {
        if (results.filename.includes('invoice')) {
          if (imgX > 400 && imgY > 300) anomalyLevel = 'HIGH';
          else if (imgX > 90 && imgX < 230 && imgY > 300 && imgY < 360) anomalyLevel = 'HIGH';
        } else if (results.filename.includes('passport')) {
          if (imgX > 320 && imgX < 380 && imgY > 220 && imgY < 265) anomalyLevel = 'HIGH';
        } else {
          if ((imgX > 200 && imgX < 400 && imgY > 150 && imgY < 300)) anomalyLevel = 'HIGH';
        }
      }

      setTooltip({ x: e.clientX, y: e.clientY, imgX, imgY, level: anomalyLevel, visible: true });
    } else {
      setTooltip((t) => ({ ...t, visible: false }));
    }
  };

  const handleMouseLeaveImage = () => setTooltip((t) => ({ ...t, visible: false }));

  return (
    <div ref={containerRef} className="workspace-viewer">
      
      {/* 5.3 Floating Toolbar - Top Center Pill */}
      <div className="workspace-viewer__toolbar">
        <button
          onClick={() => setViewMode('ORIGINAL')}
          className={`workspace-viewer__toolbar-btn ${viewMode === 'ORIGINAL' ? 'workspace-viewer__toolbar-btn--active' : ''}`}
        >
          <Eye /> Original
        </button>
        <button
          onClick={() => setViewMode('ELA_MAP')}
          className={`workspace-viewer__toolbar-btn ${viewMode === 'ELA_MAP' ? 'workspace-viewer__toolbar-btn--active' : ''}`}
        >
          <Layers /> ELA Map
        </button>
        <button
          onClick={() => setViewMode('AI_HEATMAP')}
          className={`workspace-viewer__toolbar-btn ${viewMode === 'AI_HEATMAP' ? 'workspace-viewer__toolbar-btn--active' : ''}`}
        >
          <Flame /> AI Heatmap
        </button>
        <button
          onClick={() => setViewMode('SPLIT_COMPARE')}
          className={`workspace-viewer__toolbar-btn ${viewMode === 'SPLIT_COMPARE' ? 'workspace-viewer__toolbar-btn--active' : ''}`}
        >
          <Split /> Split Compare
        </button>
      </div>

      {/* НОВОЕ: Контейнер для центрирования холста */}
      <div className="workspace-viewer__canvas-container">
        {isAnalyzing ? (
          <div className="workspace-viewer__loading">
            <div className="workspace-viewer__spinner" />
            <span className="workspace-viewer__loading-text">// ALIGNING LAYERS...</span>
          </div>
        ) : (
          <div 
            className="workspace-viewer__canvas"
            style={{ transform: `scale(${zoom / 100})`, transition: 'transform 100ms ease' }}
            onMouseMove={handleMouseMoveImage}
            onMouseLeave={handleMouseLeaveImage}
          >
            {viewMode === 'SPLIT_COMPARE' ? (
              /* Шторка: Draggable Split Compare Window */
              <div ref={splitContainerRef} className="split-view" style={{ width: '100%', height: '100%' }}>
                <img src={results.imageUrl} alt="Original" className="split-view__base" />

                <div className="split-view__overlay-container" style={{ left: `${splitRatio}%` }}>
                  <img src={results.elaImageUrl} alt="ELA Map" className="split-view__overlay-img" style={{ width: '640px', height: '480px' }} />
                </div>

                <div className="split-view__divider" style={{ left: `${splitRatio}%` }} onMouseDown={handleMouseDownSplit}>
                  <div className="split-view__handle"><Split /></div>
                </div>
              </div>
            ) : (
              /* Стандартный режим наложения */
              <div className="overlay-view">
                <img ref={imageRef} src={results.imageUrl} alt={results.filename} className="overlay-view__base" />

                {viewMode === 'ELA_MAP' && results.elaImageUrl && (
                  <img src={results.elaImageUrl} alt="ELA Overlay" className="overlay-view__layer" style={{ opacity: opacity / 100 }} />
                )}
                {viewMode === 'AI_HEATMAP' && results.heatmapImageUrl && (
                  <img src={results.heatmapImageUrl} alt="AI Heatmap" className="overlay-view__layer" style={{ opacity: opacity / 100 }} />
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* НОВОЕ: Нижняя статус-панель (заменяет старый HUD) */}
      <div className="workspace-viewer__statusbar">
        <div className="statusbar-left">
          <span className="statusbar-item statusbar-filename" title={results.filename}>
            {results.filename}
          </span>
          <span className="statusbar-divider" />
          <span className="statusbar-item">RES: {results.exif.resolution}</span>
        </div>

        <div className="statusbar-right">
          {(viewMode === 'ELA_MAP' || viewMode === 'AI_HEATMAP') && (
            <>
              <div className="statusbar-opacity">
                <Sliders size={12} />
                <span>BLEND</span>
                <input 
                  type="range" 
                  min="0" 
                  max="100" 
                  value={opacity} 
                  onChange={(e) => setOpacity(Number(e.target.value))} 
                />
                <span className="statusbar-value">{opacity}%</span>
              </div>
              <span className="statusbar-divider" />
            </>
          )}
          
          <div className="statusbar-zoom">
            <button onClick={handleZoomOut} title="Zoom Out"><ZoomOut size={14} /></button>
            <span onClick={() => setZoom(100)} className="statusbar-value zoom-value" title="Reset Zoom">
              {zoom}%
            </span>
            <button onClick={handleZoomIn} title="Zoom In"><ZoomIn size={14} /></button>
          </div>
        </div>
      </div>

      {/* Floating Cursor Coordinate Tooltip (Без изменений) */}
      {viewMode === 'AI_HEATMAP' && tooltip.visible && (
        <div className="workspace-viewer__tooltip" style={{ left: `${tooltip.x + 12}px`, top: `${tooltip.y + 12}px` }}>
          <div className="workspace-viewer__tooltip-row">
            <span>COORD:</span>
            <span className="workspace-viewer__tooltip-val--default">{tooltip.imgX}x, {tooltip.imgY}y</span>
          </div>
          <div className="workspace-viewer__tooltip-row">
            <span>ANOMALY:</span>
            <span className={`workspace-viewer__tooltip-val--${tooltip.level === 'HIGH' ? 'high' : 'normal'}`}>
              {tooltip.level}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}