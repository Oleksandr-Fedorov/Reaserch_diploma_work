import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import type { Accept } from 'react-dropzone';
import { Upload, FileWarning } from 'lucide-react';
import type { SampleTarget } from '../../../types/types';

import './styles.scss';

interface WorkspaceDropzoneProps {
  onFileSelect: (file: File) => void;
  onSampleSelect: (sample: SampleTarget) => void;
  isAnalyzing: boolean;
  acceptTypes?: Accept;
  maxFiles?: number;
}

export default function WorkspaceDropzone({ 
  onFileSelect, 
  isAnalyzing,
  acceptTypes = {
    'image/jpeg': ['.jpeg', '.jpg'],
    'image/png': ['.png'],
    'image/webp': ['.webp']
  },
  maxFiles = 1
}: WorkspaceDropzoneProps) {
  
  // Логика из вашего кастомного компонента
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0]); // Берем первый файл, так как анализируем по одному
    }
  }, [onFileSelect]);

  const { 
    getRootProps, 
    getInputProps, 
    isDragActive, 
    isDragReject 
  } = useDropzone({
    onDrop,
    accept: acceptTypes,
    maxFiles,
    disabled: isAnalyzing // Блокируем дропзону во время анализа
  });

  // Формируем BEM классы для обертки дропзоны
  const dropAreaClassName = `workspace-dropzone__area ${
    isDragActive && !isDragReject ? 'workspace-dropzone__area--active' : ''
  } ${
    isDragReject ? 'workspace-dropzone__area--reject' : ''
  }`;

  const iconWrapperClassName = `workspace-dropzone__icon-wrapper ${
    isDragActive && !isDragReject ? 'workspace-dropzone__icon-wrapper--active' : ''
  } ${
    isDragReject ? 'workspace-dropzone__icon-wrapper--reject' : ''
  }`;

  return (
    <div className="workspace-dropzone">
      {/* Главная зона загрузки */}
      <div {...getRootProps({ className: dropAreaClassName })}>
        <input {...getInputProps()} />

        {isAnalyzing ? (
          <div className="workspace-dropzone__loading">
            <div className="workspace-dropzone__spinner" />
            <div className="workspace-dropzone__loading-text">
              // MOUNTING AND ANALYZING IMAGE MATRIX...
            </div>
            <p className="workspace-dropzone__subtitle">
              Running dynamic ELA quality pass & error density mapping
            </p>
          </div>
        ) : (
          <div className="workspace-dropzone__content">
            <div className={iconWrapperClassName}>
              {isDragReject ? <FileWarning size={32} /> : <Upload size={32} />}
            </div>
            
            <div className="workspace-dropzone__text-group">
              <h2 className="workspace-dropzone__title">
                {isDragReject ? 'Недопустимый формат файла' : 'Drag Target Image Here'}
              </h2>
              <p className="workspace-dropzone__subtitle">
                or click to mount a local forensic file (JPG, PNG, WEBP)
              </p>
            </div>

            <div className="workspace-dropzone__tags">
              <span>MAX SIZE: 15MB</span>
              <span>LOCAL PROCESSING (100% PRIVATE)</span>
              <span>JPEG DIFFERENCES</span>
            </div>
          </div>
        )}
      </div>

      {/* Секция с заготовленными примерами (Sandbox) */}
      <div className="samples">
        <div className="samples__header">
          <div className="samples__header-title">
            <span className="badge">Sandbox</span>
            <h3>Select Preloaded Forensic Samples for Quick Test</h3>
          </div>
          <span className="samples__header-count">3 DEMOS LOADED</span>
        </div>
      </div>
    </div>
  );
}