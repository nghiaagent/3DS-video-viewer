/**
 * The grain pattern is re-seeded at the footage's own frame rate (about 20 fps for 3DS video), which looks like film and
 * is cheaper to encode than noise that changes every rendered frame. The live loop and the exporter share this helper so
 * a given moment of the video always gets the same grain.
 */
export const SOURCE_FPS = 20;

export function grainSeedForTime(tSec: number, sourceFps: number = SOURCE_FPS): number {
  return Math.round(tSec * sourceFps);
}
