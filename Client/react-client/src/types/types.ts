/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type ViewMode = 'ORIGINAL' | 'ELA_MAP' | 'AI_HEATMAP' | 'SPLIT_COMPARE';

export interface ExifData {
  make: string;
  model: string;
  software: string;
  originalTime: string;
  modTime: string;
  fileSize: string;
  resolution: string;
  colorSpace: string;
}

export interface ConfidenceMetrics {
  elaAnomaly: number;
  noiseInconsistency: number;
  edgeArtifacts: number;
}

export interface AnalysisResults {
  filename: string;
  verdict: 'MANIPULATION_DETECTED' | 'AUTHENTIC';
  probability: number;
  metrics: ConfidenceMetrics;
  exif: ExifData;
  fileHash: string;
  imageUrl: string; // original image
  elaImageUrl?: string; // generated ELA image
  heatmapImageUrl?: string; // generated Heatmap image
}

export interface SampleTarget {
  id: string;
  name: string;
  description: string;
  verdict: 'MANIPULATION_DETECTED' | 'AUTHENTIC';
  probability: number;
  metrics: ConfidenceMetrics;
  exif: ExifData;
  // Canvas generator function to generate a real image for ELA
  generateCanvas: (ctx: CanvasRenderingContext2D, width: number, height: number) => void;
}
