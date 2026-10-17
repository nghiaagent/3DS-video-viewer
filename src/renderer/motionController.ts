export interface MotionOptions {
  frequencyHz?: number;
  amplitude?: number;
  dampingFactor?: number; // 0..1, e.g. 0.15 for smooth interpolation
  gyroSensitivity?: number; // tilt degrees for full shift, e.g. 20.0
}

/**
 * Pure calculation of perspective shift.
 * When mouse/touch position is provided (mouseXNormalized !== null), uses it directly with clamping.
 * Otherwise, oscillates automatically with a smooth sine wave (0.28Hz, amplitude 0.42).
 */
export function getPerspectiveShift(
  timestampMs: number,
  mouseXNormalized: number | null = null,
  options?: MotionOptions
): number {
  if (mouseXNormalized !== null) {
    return Math.max(0.0, Math.min(1.0, mouseXNormalized));
  }

  const freq = options?.frequencyHz ?? 0.28;
  const amp = options?.amplitude ?? 0.42;

  const rawSine = Math.sin((timestampMs / 1000.0) * freq * Math.PI * 2.0);
  const shift = 0.5 + amp * rawSine;
  return Math.max(0.0, Math.min(1.0, shift));
}

export class MotionController {
  private element: HTMLElement | null = null;
  private mouseXNormalized: number | null = null;
  private currentShift: number = 0.5;
  private targetShift: number = 0.5;
  private isHovered: boolean = false;
  private options: Required<MotionOptions>;

  private onMouseMoveHandler = (e: MouseEvent) => this.onMouseMove(e);
  private onMouseLeaveHandler = () => this.onMouseLeave();
  private onTouchMoveHandler = (e: TouchEvent) => this.onTouchMove(e);
  private onTouchEndHandler = () => this.onTouchEnd();
  private onDeviceOrientationHandler = (e: DeviceOrientationEvent) => this.onDeviceOrientation(e);

  constructor(element?: HTMLElement, options?: MotionOptions) {
    this.options = {
      frequencyHz: options?.frequencyHz ?? 0.28,
      amplitude: options?.amplitude ?? 0.42,
      dampingFactor: options?.dampingFactor ?? 0.18,
      gyroSensitivity: options?.gyroSensitivity ?? 20.0,
    };

    if (element) {
      this.attach(element);
    }
  }

  public attach(element: HTMLElement): void {
    this.detach();
    this.element = element;

    element.addEventListener('mousemove', this.onMouseMoveHandler);
    element.addEventListener('mouseleave', this.onMouseLeaveHandler);
    element.addEventListener('touchmove', this.onTouchMoveHandler, { passive: true });
    element.addEventListener('touchend', this.onTouchEndHandler);
    element.addEventListener('touchcancel', this.onTouchEndHandler);

    if (typeof window !== 'undefined' && 'DeviceOrientationEvent' in window) {
      window.addEventListener('deviceorientation', this.onDeviceOrientationHandler);
    }
  }

  public detach(): void {
    if (this.element) {
      this.element.removeEventListener('mousemove', this.onMouseMoveHandler);
      this.element.removeEventListener('mouseleave', this.onMouseLeaveHandler);
      this.element.removeEventListener('touchmove', this.onTouchMoveHandler);
      this.element.removeEventListener('touchend', this.onTouchEndHandler);
      this.element.removeEventListener('touchcancel', this.onTouchEndHandler);
      this.element = null;
    }

    if (typeof window !== 'undefined') {
      window.removeEventListener('deviceorientation', this.onDeviceOrientationHandler);
    }
  }

  public destroy(): void {
    this.detach();
  }

  private onMouseMove(e: MouseEvent): void {
    if (!this.element) return;
    const rect = this.element.getBoundingClientRect();
    if (rect.width <= 0) return;

    this.isHovered = true;
    const normX = (e.clientX - rect.left) / rect.width;
    this.mouseXNormalized = Math.max(0.0, Math.min(1.0, normX));
  }

  private onMouseLeave(): void {
    this.isHovered = false;
    this.mouseXNormalized = null;
  }

  private onTouchMove(e: TouchEvent): void {
    if (!this.element || e.touches.length === 0) return;
    const rect = this.element.getBoundingClientRect();
    if (rect.width <= 0) return;

    this.isHovered = true;
    const touch = e.touches[0];
    const normX = (touch.clientX - rect.left) / rect.width;
    this.mouseXNormalized = Math.max(0.0, Math.min(1.0, normX));
  }

  private onTouchEnd(): void {
    this.isHovered = false;
    this.mouseXNormalized = null;
  }

  private onDeviceOrientation(e: DeviceOrientationEvent): void {
    // Only use gyro if not hovering with mouse/touch
    if (this.isHovered || e.gamma === null) return;

    // gamma represents left-to-right tilt in degrees [-90, 90]
    const clampedGamma = Math.max(-this.options.gyroSensitivity, Math.min(this.options.gyroSensitivity, e.gamma));
    // Map [-sensitivity, +sensitivity] to [0.0, 1.0]
    const gyroShift = (clampedGamma / (2.0 * this.options.gyroSensitivity)) + 0.5;
    this.targetShift = Math.max(0.0, Math.min(1.0, gyroShift));
  }

  /**
   * Advances the motion state and returns the current perspective shift in [0.0, 1.0].
   */
  public update(timestampMs: number): number {
    if (this.mouseXNormalized !== null) {
      this.targetShift = this.mouseXNormalized;
    } else if (!this.isHovered) {
      this.targetShift = getPerspectiveShift(timestampMs, null, this.options);
    }

    // Exponential smoothing for buttery-smooth 60fps parallax
    const factor = this.options.dampingFactor;
    this.currentShift += (this.targetShift - this.currentShift) * factor;

    return Math.max(0.0, Math.min(1.0, this.currentShift));
  }

  public getShift(): number {
    return this.currentShift;
  }

  public setManualShift(shift: number): void {
    this.currentShift = Math.max(0.0, Math.min(1.0, shift));
    this.targetShift = this.currentShift;
    this.mouseXNormalized = this.currentShift;
  }
}
