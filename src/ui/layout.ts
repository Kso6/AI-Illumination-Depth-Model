/**
 * How large the back buffer should be.
 *
 * Separated from the application shell because it is the one piece of the
 * layout with a performance cliff behind it, and because the bug it now guards
 * against was invisible: a CSS grid whose row grew to the sidebar's content
 * height made the canvas box more than twice the height of the window, so the
 * picture ran off the bottom of the screen while the shading pass — which costs
 * per pixel — paid for all of it.
 */
export interface Viewport {
  readonly w: number;
  readonly h: number;
}

/**
 * `scale` is the render scale; the client size is what CSS gave the canvas and
 * the window size is the hard ceiling. A back buffer larger than the window
 * cannot be displayed, so producing one is always waste.
 */
export function backBufferSize(
  clientWidth: number,
  clientHeight: number,
  windowWidth: number,
  windowHeight: number,
  scale: number,
): Viewport {
  const s = Number.isFinite(scale) && scale > 0 ? scale : 1;
  const cw = Math.min(orZero(clientWidth), orZero(windowWidth) || orZero(clientWidth));
  const ch = Math.min(orZero(clientHeight), orZero(windowHeight) || orZero(clientHeight));
  return {
    w: Math.max(1, Math.round(cw * s)),
    h: Math.max(1, Math.round(ch * s)),
  };
}

function orZero(v: number): number {
  return Number.isFinite(v) && v > 0 ? v : 0;
}
