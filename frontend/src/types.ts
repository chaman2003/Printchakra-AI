/**
 * Application Types
 * Common type definitions for the application
 */

export interface FileInfo {
  filename: string;
  size: number;
  created: string;
  has_text: boolean;
  has_ocr?: boolean;
  page_count?: number;
  mime_type?: string;
  processing?: boolean;
  processing_step?: number;
  processing_total?: number;
  processing_stage?: string;
  processing_eta?: number;
  processing_progress?: number;
  processing_error?: string;
  thumbnail?: string;
  thumbnailUrl?: string; // Keep for compatibility if needed
}

export interface DocumentPage {
  pageNumber: number;
  thumbnailUrl?: string;
  fullUrl?: string;
}

export interface Document {
  filename: string;
  pages?: DocumentPage[];
  thumbnailUrl?: string;
  fullUrl?: string;
  size?: number;
  type?: string;
  fileObject?: File;
}

export interface ScanConfig {
  resolution: number;
  colorMode: 'color';
  paperSize: string;
  orientation: 'portrait' | 'landscape';
  autoDetect: boolean;
  enhance: boolean;
  ocr: boolean;
}

export interface PrintConfig {
  printerName?: string;
  copies: number;
  paperSize: string;
  orientation: 'portrait' | 'landscape';
  colorMode: 'color';
  quality: 'draft' | 'normal' | 'high';
  printResolution: number;
  duplex: boolean;
  collate: boolean;
  pages?: number[];
  scale: number;
}

export interface PreviewSettings {
  layout?: 'portrait' | 'landscape';
  scale?: number;
  colorMode?: 'color';
  paperSize?: string;
}

export interface ConversionOptions {
  format: 'pdf' | 'jpg' | 'png';
  quality?: number;
  dpi?: number;
}

// OCR Types
export interface OCRBoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  points?: number[][];
}

export interface OCRRawResult {
  text: string;
  confidence: number;
  bbox: OCRBoundingBox;
}

export interface OCRStructuredUnit {
  text: string;
  type: 'title' | 'heading' | 'paragraph' | 'list_item' | 'table_cell' | 'footer' | 'other';
  bbox: OCRBoundingBox;
  confidence: number;
  word_indices: number[];
}

export interface OCRResult {
  raw_results: OCRRawResult[];
  structured_units: OCRStructuredUnit[];
  full_text: string;
  derived_title: string;
  confidence_avg: number;
  word_count: number;
  timestamp: string;
  processing_time_ms: number;
  image_dimensions: [number, number];
}

export interface OCRResponse {
  success: boolean;
  filename: string;
  original_filename?: string;  // Present if file was renamed
  ocr_result: OCRResult | null;
  ocr_ready: boolean;
  error?: string;
}

