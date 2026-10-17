export * from './offlineExporter.js';
export * from './realtimeExporter.js';

import type { ExportFormat, OfflineExportOptions, ExportResult } from './offlineExporter.js';
import { exportOfflineGPUVideo } from './offlineExporter.js';
import { startRealtimeGPUVideo, type RealtimeExportSession } from './realtimeExporter.js';
import type { WebGLParallaxViewer } from '../renderer/webglViewer.js';

export type ExportMode = 'offline' | 'realtime';

export async function exportParallaxVideo(
  format: ExportFormat,
  mode: ExportMode,
  viewer: WebGLParallaxViewer,
  sourceVideo: HTMLVideoElement,
  options?: OfflineExportOptions
): Promise<ExportResult | RealtimeExportSession> {
  if (mode === 'offline') {
    return exportOfflineGPUVideo(format, viewer, sourceVideo, options);
  } else {
    return startRealtimeGPUVideo(format, viewer.getCanvas(), sourceVideo, options);
  }
}
