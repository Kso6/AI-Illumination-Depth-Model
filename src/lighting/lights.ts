/**
 * Light definitions shared by the CPU-side scene and the shading kernel.
 *
 * Light positions are uploaded already transformed into **view space**, which is
 * the space the depth map reconstructs into. Doing the transform on the CPU
 * costs nothing (there are a handful of lights) and keeps the shader free of
 * matrix maths.
 */
import * as d from 'typegpu/data';

export const LightKind = {
  point: 0,
  spot: 1,
  directional: 2,
} as const;

export type LightKindName = keyof typeof LightKind;

export const Light = d.struct({
  /** `xyz` = view-space position (or direction, for a directional light). */
  position: d.vec4f,
  /** `rgb` = linear colour, `a` = intensity in arbitrary radiometric units. */
  color: d.vec4f,
  /** `xyz` = view-space spot axis, `w` = cosine of the outer cone angle. */
  direction: d.vec4f,
  /**
   * `x` = kind (see `LightKind`), `y` = cosine of the inner cone angle,
   * `z` = influence radius, `w` = contact-shadow strength in `[0, 1]`.
   */
  params: d.vec4f,
});

/** Maximum lights the shading kernel is compiled for. */
export const MAX_LIGHTS = 8;

export const LightBuffer = d.arrayOf(Light, MAX_LIGHTS);

export interface LightDescription {
  readonly kind: LightKindName;
  /** World-space position (metres), or direction for a directional light. */
  readonly position: readonly [number, number, number];
  readonly color: readonly [number, number, number];
  readonly intensity: number;
  readonly radius?: number;
  /** Spot cone half-angles in radians. */
  readonly innerAngle?: number;
  readonly outerAngle?: number;
  readonly shadow?: number;
  readonly direction?: readonly [number, number, number];
}

function normalise(v: readonly [number, number, number]): [number, number, number] {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

/**
 * Packs light descriptions into the GPU struct array.
 *
 * The "view transform" here is deliberately just a translation of the camera
 * origin: the scene the depth model reconstructs is always seen from the camera
 * that produced the image, so view space *is* the reconstruction space and the
 * lights are authored directly in it.
 */
export function packLights(lights: readonly LightDescription[]): Array<d.Infer<typeof Light>> {
  if (lights.length > MAX_LIGHTS) {
    throw new Error(`packLights: ${lights.length} lights exceeds the maximum of ${MAX_LIGHTS}`);
  }
  const out: Array<d.Infer<typeof Light>> = [];
  for (let i = 0; i < MAX_LIGHTS; i++) {
    const l = lights[i];
    if (!l) {
      // Zero intensity — the shader skips it without branching on a count.
      out.push({
        position: d.vec4f(0, 0, 0, 0),
        color: d.vec4f(0, 0, 0, 0),
        direction: d.vec4f(0, 0, -1, -1),
        params: d.vec4f(0, 0, 0, 0),
      });
      continue;
    }
    const dir = normalise(l.direction ?? [0, 0, -1]);
    const outer = Math.cos(l.outerAngle ?? Math.PI / 5);
    const inner = Math.cos(Math.min(l.innerAngle ?? Math.PI / 8, l.outerAngle ?? Math.PI / 5));
    out.push({
      position: d.vec4f(l.position[0], l.position[1], l.position[2], 1),
      color: d.vec4f(l.color[0], l.color[1], l.color[2], l.intensity),
      direction: d.vec4f(dir[0], dir[1], dir[2], outer),
      params: d.vec4f(LightKind[l.kind], inner, l.radius ?? 8, l.shadow ?? 1),
    });
  }
  return out;
}
