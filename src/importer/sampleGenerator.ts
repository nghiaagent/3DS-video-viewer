/**
 * Procedural stereoscopic 3D Side-by-Side (SBS) canvas generator.
 * Used as an instant offline/zero-asset fallback producing animated 3D geometries
 * at distinct stereoscopic depth planes.
 */
export class ProceduralStereoGenerator {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private animId: number | null = null;
  private isRunning: boolean = false;

  constructor(width: number = 960, height: number = 240) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = width;
    this.canvas.height = height;
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get 2D context for ProceduralStereoGenerator');
    this.ctx = ctx;
  }

  public getCanvas(): HTMLCanvasElement {
    return this.canvas;
  }

  public start(): void {
    if (this.isRunning) return;
    this.isRunning = true;
    const render = (time: number) => {
      if (!this.isRunning) return;
      this.drawFrame(time);
      this.animId = requestAnimationFrame(render);
    };
    this.animId = requestAnimationFrame(render);
  }

  public stop(): void {
    this.isRunning = false;
    if (this.animId !== null) {
      cancelAnimationFrame(this.animId);
      this.animId = null;
    }
  }

  public drawFrame(time: number): void {
    const ctx = this.ctx;
    const sbsW = this.canvas.width;
    const eyeW = sbsW / 2;
    const h = this.canvas.height;
    const t = time / 1000.0;

    // Clear background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, sbsW, h);

    // Grid floor
    this.drawGrid(ctx, 0, eyeW, h);
    this.drawGrid(ctx, eyeW, eyeW, h);

    // Objects at 3 distinct depth planes:
    // Background object: disparity d = -2 (behind screen)
    // Screen-plane object: disparity d = 0 (on screen)
    // Foreground object: disparity d = +14 (popping out)
    const objects = [
      {
        baseX: eyeW * 0.3,
        y: h * 0.35 + Math.sin(t * 1.5) * 15,
        radius: 28,
        color: '#38bdf8',
        label: 'Background (d=-2)',
        disp: -2,
      },
      {
        baseX: eyeW * 0.5 + Math.cos(t * 1.2) * 25,
        y: h * 0.55,
        radius: 36,
        color: '#a855f7',
        label: 'Screen Plane (d=0)',
        disp: 0,
      },
      {
        baseX: eyeW * 0.72,
        y: h * 0.65 + Math.sin(t * 2.0) * 18,
        radius: 44,
        color: '#f43f5e',
        label: 'Foreground (d=+14)',
        disp: 14,
      },
    ];

    for (const obj of objects) {
      // Left Eye (x_left = baseX)
      const leftX = obj.baseX;
      // Right Eye (x_right = baseX - disp)
      const rightX = eyeW + (obj.baseX - obj.disp);

      // Draw Left Eye instance
      this.drawObject(ctx, leftX, obj.y, obj.radius, obj.color, obj.label);
      // Draw Right Eye instance
      this.drawObject(ctx, rightX, obj.y, obj.radius, obj.color, obj.label);
    }

    // Divider line between Left and Right eyes
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(eyeW, 0);
    ctx.lineTo(eyeW, h);
    ctx.stroke();
  }

  private drawGrid(ctx: CanvasRenderingContext2D, startX: number, w: number, h: number): void {
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let x = startX; x < startX + w; x += 30) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += 30) {
      ctx.beginPath();
      ctx.moveTo(startX, y);
      ctx.lineTo(startX + w, y);
      ctx.stroke();
    }
  }

  private drawObject(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    radius: number,
    color: string,
    label: string
  ): void {
    ctx.save();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(label, x, y + radius + 14);
    ctx.restore();
  }
}
