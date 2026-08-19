/**
 * The back-buffer sizing rule.
 *
 * This exists because of a real failure: the sidebar's content made the CSS
 * grid row grow past the window, the canvas box grew with it, and the render
 * ended up more than twice the height of the screen. Nothing looked wrong in
 * the numbers — the picture simply ran off the bottom while the shading pass,
 * which costs per pixel, paid for all of it.
 */
import { describe, expect, it } from 'vitest';
import { backBufferSize } from '../src/ui/layout.ts';

describe('back-buffer sizing', () => {
  it('uses the canvas box when it fits inside the window', () => {
    expect(backBufferSize(1040, 800, 1360, 800, 1)).toEqual({ w: 1040, h: 800 });
  });

  it('never exceeds the window, however tall the canvas box grows', () => {
    // The reported failure: a 1360x2174 buffer in a window 1000 px tall.
    expect(backBufferSize(1360, 2174, 1360, 1000, 1)).toEqual({ w: 1360, h: 1000 });
  });

  it('applies the render scale after clamping', () => {
    expect(backBufferSize(1360, 2174, 1360, 1000, 0.75)).toEqual({ w: 1020, h: 750 });
    expect(backBufferSize(800, 600, 1600, 1200, 1.5)).toEqual({ w: 1200, h: 900 });
  });

  it('keeps at least one pixel in each axis', () => {
    expect(backBufferSize(0, 0, 0, 0, 1)).toEqual({ w: 1, h: 1 });
    expect(backBufferSize(100, 100, 1000, 1000, 0.001)).toEqual({ w: 1, h: 1 });
  });

  it('falls back to a sane scale rather than producing a zero-sized target', () => {
    expect(backBufferSize(800, 600, 1600, 1200, Number.NaN)).toEqual({ w: 800, h: 600 });
    expect(backBufferSize(800, 600, 1600, 1200, -2)).toEqual({ w: 800, h: 600 });
  });

  it('ignores a missing window size instead of collapsing to one pixel', () => {
    // `innerWidth` is 0 in some embedding contexts; that must not blank the view.
    expect(backBufferSize(640, 480, 0, 0, 1)).toEqual({ w: 640, h: 480 });
  });

  // 2174 px of buffer for a 1000 px window is 2.17x the shading work for
  // pixels that were never on screen.
  it('cuts the reported case to well under half the pixels', () => {
    const bad = 1360 * 2174;
    const fixed = backBufferSize(1360, 2174, 1360, 1000, 1);
    expect((fixed.w * fixed.h) / bad).toBeLessThan(0.47);
  });
});
