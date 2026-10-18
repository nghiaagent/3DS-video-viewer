import { test, expect } from '@playwright/test';

test.describe('Depth and Focus controls', () => {
  test('the Depth slider sets the parallax strength and the Focus slider sets the focus plane', async ({ page }) => {
    await page.goto('/');
    await page.waitForFunction(() => !!(window as any).__viewerApp?.viewer);

    // defaults: a little stronger parallax than before (0.34), focus plane 0.35
    const defaults = await page.evaluate(() => {
      const v = (window as any).__viewerApp.viewer;
      return { sensitivity: v.getSensitivity(), focus: v.getFocus() };
    });
    expect(defaults.sensitivity).toBeCloseTo(0.34, 5);
    expect(defaults.focus).toBeCloseTo(0.35, 5);

    // both controls sit in the dock (no pop-up panel to open)
    await expect(page.locator('#maxdisp-slider')).toBeVisible();
    await expect(page.locator('#focus-slider')).toBeVisible();

    await page.locator('#maxdisp-slider').fill('0.5');
    await page.locator('#focus-slider').fill('0.6');
    const after = await page.evaluate(() => {
      const v = (window as any).__viewerApp.viewer;
      return { sensitivity: v.getSensitivity(), focus: v.getFocus() };
    });
    expect(after.sensitivity).toBeCloseTo(0.5, 5);
    expect(after.focus).toBeCloseTo(0.6, 5);
  });
});
