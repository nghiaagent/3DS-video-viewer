export interface DcimItem {
  name: string;
  relativePath: string;
  folder: string;
  sizeBytes: number;
  mtime: number;
}

export interface BundledSample {
  id: string;
  name: string;
  type: string;
  url: string;
  format: 'sbs_mp4' | 'raw_avi';
  sizeBytes?: number;
  width?: number;
  height?: number;
  description?: string;
}

export interface ManifestData {
  samples: BundledSample[];
}

export async function fetchDcimList(): Promise<DcimItem[]> {
  try {
    const res = await fetch('/api/dcim/list');
    if (!res.ok) {
      return [];
    }
    const data = await res.json();
    if (Array.isArray(data)) {
      return data;
    }
    return [];
  } catch {
    // Expected on static GitHub Pages deployment where Vite middleware is not running
    return [];
  }
}

export async function fetchBundledManifest(): Promise<BundledSample[]> {
  try {
    const res = await fetch('samples/manifest.json');
    if (!res.ok) {
      return [];
    }
    const data: ManifestData = await res.json();
    return data.samples || [];
  } catch (err) {
    console.warn('Failed to load bundled manifest:', err);
    return [];
  }
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
