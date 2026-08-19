/**
 * How an external frame is fitted to the network's square input, and when
 * inference is allowed to be skipped.
 *
 * The original camera path copied a 448×448 *window* out of the frame, because
 * `copyExternalImageToTexture` cannot scale. On a 1280×720 webcam that is a
 * heavily zoomed crop of about a third of the picture — which is what made the
 * camera look broken rather than merely mis-framed. These tests pin down the
 * replacement: a centre crop to square covering the full short axis, scaled.
 */
import { describe, expect, it } from 'vitest';
import { cropTransform } from '../src/scene/blit.ts';

describe('frame fitting', () => {
  it('maps the full short axis and a centred square of the long axis', () => {
    // 1280×720 landscape: the whole height, the middle 720 px of the width.
    const [sx, sy, ox, oy] = cropTransform(1280, 720);
    expect(sy).toBe(1);
    expect(oy).toBe(0);
    expect(sx).toBeCloseTo(720 / 1280, 10);
    expect(ox).toBeCloseTo((1 - 720 / 1280) / 2, 10);
    // The sampled span must be centred and exactly square.
    expect(ox + sx / 2).toBeCloseTo(0.5, 10);
    // A 448-pixel window out of 1280 would cover 0.35 of the width; a correct
    // fit covers 0.5625 — the whole short axis.
    expect(sx).toBeGreaterThan(448 / 1280);
  });

  it('handles portrait by cropping the height instead', () => {
    const [sx, sy, ox, oy] = cropTransform(720, 1280);
    expect(sx).toBe(1);
    expect(ox).toBe(0);
    expect(sy).toBeCloseTo(720 / 1280, 10);
    expect(oy + sy / 2).toBeCloseTo(0.5, 10);
  });

  it('is the identity for a square frame', () => {
    expect(cropTransform(512, 512)).toEqual([1, 1, 0, 0]);
  });

  it('degrades safely for an empty frame', () => {
    expect(cropTransform(0, 0)).toEqual([1, 1, 0, 0]);
    expect(cropTransform(-4, 10)).toEqual([1, 1, 0, 0]);
  });
});
