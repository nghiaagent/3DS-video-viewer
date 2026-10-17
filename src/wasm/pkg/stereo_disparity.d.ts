/* tslint:disable */
/* eslint-disable */

export class StereoDisparity {
    free(): void;
    [Symbol.dispose](): void;
    base_disp(): number;
    /**
     * Computes disparity map into internal buffer and returns a cloned Vec<u8>.
     */
    compute_disparity(sbs_rgba: Uint8Array): Uint8Array;
    /**
     * Zero-copy in-place computation for high-speed WebGL2 texture upload.
     * Returns raw pointer to the internal disparity buffer.
     */
    compute_disparity_inplace(sbs_rgba: Uint8Array): number;
    disparity_len(): number;
    disparity_ptr(): number;
    /**
     * Computes disparity map into internal buffer and returns a cloned Vec<u8>.
     * Estimates the global horizontal stereo baseline shift between Left and Right eyes.
     * Returns the optimal horizontal disparity in pixels within [min_shift, max_shift].
     */
    estimate_global_disparity(sbs_rgba: Uint8Array, min_shift: number, max_shift: number): number;
    height(): number;
    max_disp(): number;
    constructor(eye_width: number, eye_height: number, max_disp: number);
    set_base_disp(base: number): void;
    width(): number;
}

/**
 * Helper function to 2x2 downsample a Side-by-Side RGBA frame.
 * Inputs: full_sbs_width (e.g. 960), full_height (e.g. 240).
 * Output: RGBA buffer of size (full_sbs_width / 2) * (full_height / 2) * 4.
 */
export function downsample_sbs_rgba_half(sbs_rgba: Uint8Array, full_sbs_width: number, full_height: number): Uint8Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_stereodisparity_free: (a: number, b: number) => void;
    readonly downsample_sbs_rgba_half: (a: number, b: number, c: number, d: number) => [number, number];
    readonly stereodisparity_base_disp: (a: number) => number;
    readonly stereodisparity_compute_disparity: (a: number, b: number, c: number) => [number, number];
    readonly stereodisparity_compute_disparity_inplace: (a: number, b: number, c: number) => number;
    readonly stereodisparity_disparity_len: (a: number) => number;
    readonly stereodisparity_disparity_ptr: (a: number) => number;
    readonly stereodisparity_estimate_global_disparity: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly stereodisparity_height: (a: number) => number;
    readonly stereodisparity_max_disp: (a: number) => number;
    readonly stereodisparity_new: (a: number, b: number, c: number) => number;
    readonly stereodisparity_set_base_disp: (a: number, b: number) => void;
    readonly stereodisparity_width: (a: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
