use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct StereoDisparity {
    eye_width: usize,
    eye_height: usize,
    max_disp: i32,
    base_disp: i32,
    left_luma: Vec<u8>,
    right_luma: Vec<u8>,
    disparity_map: Vec<u8>,
}

#[wasm_bindgen]
impl StereoDisparity {
    #[wasm_bindgen(constructor)]
    pub fn new(eye_width: usize, eye_height: usize, max_disp: i32) -> Self {
        let pixel_count = eye_width * eye_height;
        Self {
            eye_width,
            eye_height,
            max_disp,
            base_disp: 0,
            left_luma: vec![0u8; pixel_count],
            right_luma: vec![0u8; pixel_count],
            disparity_map: vec![0u8; pixel_count],
        }
    }

    pub fn width(&self) -> usize {
        self.eye_width
    }

    pub fn height(&self) -> usize {
        self.eye_height
    }

    pub fn max_disp(&self) -> i32 {
        self.max_disp
    }

    pub fn base_disp(&self) -> i32 {
        self.base_disp
    }

    pub fn set_base_disp(&mut self, base: i32) {
        self.base_disp = base;
    }

    pub fn disparity_ptr(&self) -> *const u8 {
        self.disparity_map.as_ptr()
    }

    pub fn disparity_len(&self) -> usize {
        self.disparity_map.len()
    }

    /// Extract grayscale luma from Side-by-Side RGBA buffer.
    /// Supports exact matching resolution (sbs_width = eye_width * 2)
    /// or 2x full-res input (sbs_width = eye_width * 4) with on-the-fly 2x2 box averaging.
    fn extract_luma(&mut self, sbs_rgba: &[u8]) {
        let w = self.eye_width;
        let h = self.eye_height;
        let exact_sbs_w = w * 2;
        let exact_len = exact_sbs_w * h * 4;
        let double_sbs_w = w * 4;
        let double_len = double_sbs_w * (h * 2) * 4;

        if sbs_rgba.len() == double_len {
            // Full-res 2x downsampling directly into half-res luma buffers
            let full_stride = double_sbs_w * 4;
            let full_eye_w = w * 2;

            for y in 0..h {
                let fy0 = y * 2;
                let fy1 = fy0 + 1;
                let row0_offset = fy0 * full_stride;
                let row1_offset = fy1 * full_stride;
                let dest_row_offset = y * w;

                for x in 0..w {
                    let fx0 = x * 2;
                    let fx1 = fx0 + 1;

                    // Left eye (fx0, fx1 in 0..full_eye_w)
                    let l00 = row0_offset + fx0 * 4;
                    let l01 = row0_offset + fx1 * 4;
                    let l10 = row1_offset + fx0 * 4;
                    let l11 = row1_offset + fx1 * 4;

                    let lr = sbs_rgba[l00] as u32 + sbs_rgba[l01] as u32 + sbs_rgba[l10] as u32 + sbs_rgba[l11] as u32;
                    let lg = sbs_rgba[l00 + 1] as u32 + sbs_rgba[l01 + 1] as u32 + sbs_rgba[l10 + 1] as u32 + sbs_rgba[l11 + 1] as u32;
                    let lb = sbs_rgba[l00 + 2] as u32 + sbs_rgba[l01 + 2] as u32 + sbs_rgba[l10 + 2] as u32 + sbs_rgba[l11 + 2] as u32;
                    self.left_luma[dest_row_offset + x] = ((77 * lr + 150 * lg + 29 * lb + 512) >> 10) as u8;

                    // Right eye (fx0 + full_eye_w, fx1 + full_eye_w)
                    let r00 = row0_offset + (fx0 + full_eye_w) * 4;
                    let r01 = row0_offset + (fx1 + full_eye_w) * 4;
                    let r10 = row1_offset + (fx0 + full_eye_w) * 4;
                    let r11 = row1_offset + (fx1 + full_eye_w) * 4;

                    let rr = sbs_rgba[r00] as u32 + sbs_rgba[r01] as u32 + sbs_rgba[r10] as u32 + sbs_rgba[r11] as u32;
                    let rg = sbs_rgba[r00 + 1] as u32 + sbs_rgba[r01 + 1] as u32 + sbs_rgba[r10 + 1] as u32 + sbs_rgba[r11 + 1] as u32;
                    let rb = sbs_rgba[r00 + 2] as u32 + sbs_rgba[r01 + 2] as u32 + sbs_rgba[r10 + 2] as u32 + sbs_rgba[r11 + 2] as u32;
                    self.right_luma[dest_row_offset + x] = ((77 * rr + 150 * rg + 29 * rb + 512) >> 10) as u8;
                }
            }
        } else if sbs_rgba.len() >= exact_len {
            // 1:1 direct luma extraction
            let sbs_stride = exact_sbs_w * 4;
            for y in 0..h {
                let row_offset = y * sbs_stride;
                let dest_row_offset = y * w;
                for x in 0..w {
                    let left_idx = row_offset + x * 4;
                    let right_idx = row_offset + (w + x) * 4;

                    let l_luma = (77 * sbs_rgba[left_idx] as u32
                        + 150 * sbs_rgba[left_idx + 1] as u32
                        + 29 * sbs_rgba[left_idx + 2] as u32)
                        >> 8;
                    self.left_luma[dest_row_offset + x] = l_luma as u8;

                    let r_luma = (77 * sbs_rgba[right_idx] as u32
                        + 150 * sbs_rgba[right_idx + 1] as u32
                        + 29 * sbs_rgba[right_idx + 2] as u32)
                        >> 8;
                    self.right_luma[dest_row_offset + x] = r_luma as u8;
                }
            }
        }
    }

    /// Computes disparity map into internal buffer and returns a cloned Vec<u8>.
    /// Estimates the global horizontal stereo baseline shift between Left and Right eyes.
    /// Returns the optimal horizontal disparity in pixels within [min_shift, max_shift].
    pub fn estimate_global_disparity(&mut self, sbs_rgba: &[u8], min_shift: i32, max_shift: i32) -> i32 {
        self.extract_luma(sbs_rgba);
        let w = self.eye_width as i32;
        let h = self.eye_height as i32;

        let y_start = (h / 6).max(2);
        let y_end = (5 * h / 6).min(h - 2);
        let x_start = (w / 6).max(4);
        let x_end = (5 * w / 6).min(w - 4);

        let left_ptr = self.left_luma.as_ptr();
        let right_ptr = self.right_luma.as_ptr();
        let w_usize = self.eye_width;

        let mut best_coarse_d = min_shift;
        let mut min_coarse_sad = u64::MAX;

        // Coarse pass: step 4
        let mut d = min_shift;
        while d <= max_shift {
            let mut sad_sum = 0u64;
            let mut count = 0u64;

            for y in (y_start..y_end).step_by(4) {
                let row_offset = (y as usize) * w_usize;
                for x in (x_start..x_end).step_by(4) {
                    let rx = x - d;
                    if rx >= 0 && rx < w {
                        unsafe {
                            let l_val = *left_ptr.add(row_offset + x as usize);
                            let r_val = *right_ptr.add(row_offset + rx as usize);
                            sad_sum += l_val.abs_diff(r_val) as u64;
                            count += 1;
                        }
                    }
                }
            }

            if count > 0 {
                let mean_sad = (sad_sum * 1024) / count;
                if mean_sad < min_coarse_sad {
                    min_coarse_sad = mean_sad;
                    best_coarse_d = d;
                }
            }
            d += 4;
        }

        // Fine pass: step 1 around best_coarse_d
        let mut best_d = best_coarse_d;
        let mut min_fine_sad = u64::MAX;
        let fine_min = (best_coarse_d - 3).max(min_shift);
        let fine_max = (best_coarse_d + 3).min(max_shift);

        for d in fine_min..=fine_max {
            let mut sad_sum = 0u64;
            let mut count = 0u64;

            for y in (y_start..y_end).step_by(2) {
                let row_offset = (y as usize) * w_usize;
                for x in (x_start..x_end).step_by(2) {
                    let rx = x - d;
                    if rx >= 0 && rx < w {
                        unsafe {
                            let l_val = *left_ptr.add(row_offset + x as usize);
                            let r_val = *right_ptr.add(row_offset + rx as usize);
                            sad_sum += l_val.abs_diff(r_val) as u64;
                            count += 1;
                        }
                    }
                }
            }

            if count > 0 {
                let mean_sad = (sad_sum * 1024) / count;
                if mean_sad < min_fine_sad {
                    min_fine_sad = mean_sad;
                    best_d = d;
                }
            }
        }

        self.base_disp = best_d;
        best_d
    }

    /// Computes disparity map into internal buffer and returns a cloned Vec<u8>.
    pub fn compute_disparity(&mut self, sbs_rgba: &[u8]) -> Vec<u8> {
        self.compute_internal(sbs_rgba);
        self.disparity_map.clone()
    }

    /// Zero-copy in-place computation for high-speed WebGL2 texture upload.
    /// Returns raw pointer to the internal disparity buffer.
    pub fn compute_disparity_inplace(&mut self, sbs_rgba: &[u8]) -> *const u8 {
        self.compute_internal(sbs_rgba);
        self.disparity_map.as_ptr()
    }

    /// Core block matching algorithm with stepped 5x5 SAD and border replication.
    fn compute_internal(&mut self, sbs_rgba: &[u8]) {
        self.extract_luma(sbs_rgba);

        let w = self.eye_width;
        let h = self.eye_height;
        let (min_d, max_d) = if self.base_disp == 0 {
            (-4i32, self.max_disp)
        } else {
            (self.base_disp - self.max_disp, self.base_disp + self.max_disp)
        };
        let d_range = (max_d - min_d).max(1) as u32;

        const RADIUS: usize = 4;
        const STEPS: [i32; 5] = [-4, -2, 0, 2, 4];
        // Center row (dy = 0) evaluated first for maximum early exit pruning
        const ROW_ORDER: [usize; 5] = [2, 1, 3, 0, 4];

        let y_start = RADIUS;
        let y_end = h.saturating_sub(RADIUS);

        let x_start = (max_d + 4).max(4).max(0) as usize;
        let x_end = ((w as i32 - 4 + min_d.min(0)).min(w as i32 - 4)).max(x_start as i32) as usize;

        if y_end <= y_start || x_end <= x_start {
            return;
        }

        let left_ptr = self.left_luma.as_ptr();
        let right_ptr = self.right_luma.as_ptr();
        let disp_ptr = self.disparity_map.as_mut_ptr();

        // Interior SAD calculation
        for y in y_start..y_end {
            let row_offset = y * w;

            // Pre-calculate row offsets for each dy
            let mut left_row_offsets = [0usize; 5];
            let mut right_row_offsets = [0usize; 5];
            for (i, &dy) in STEPS.iter().enumerate() {
                let sy = (y as i32 + dy) as usize;
                left_row_offsets[i] = sy * w;
                right_row_offsets[i] = sy * w;
            }

            let mut prev_best_d = self.base_disp;

            for x in x_start..x_end {
                // Preload the 25 left samples (fixed for this x, y across all d)
                let mut l_samples = [[0u8; 5]; 5];
                unsafe {
                    for i in 0..5 {
                        let l_row = left_row_offsets[i] + x;
                        l_samples[i][0] = *left_ptr.add(l_row - 4);
                        l_samples[i][1] = *left_ptr.add(l_row - 2);
                        l_samples[i][2] = *left_ptr.add(l_row);
                        l_samples[i][3] = *left_ptr.add(l_row + 2);
                        l_samples[i][4] = *left_ptr.add(l_row + 4);
                    }
                }

                // First evaluate prev_best_d to establish a tight best_sad threshold
                let start_d = if prev_best_d >= min_d && prev_best_d <= max_d {
                    prev_best_d
                } else {
                    self.base_disp
                };

                let mut best_sad = 0u32;
                let mut best_d = start_d;
                unsafe {
                    let r_base = (x as i32 - start_d) as usize;
                    for &i in &ROW_ORDER {
                        let r_row = right_row_offsets[i] + r_base;
                        let r0 = *right_ptr.add(r_row - 4);
                        let r1 = *right_ptr.add(r_row - 2);
                        let r2 = *right_ptr.add(r_row);
                        let r3 = *right_ptr.add(r_row + 2);
                        let r4 = *right_ptr.add(r_row + 4);

                        best_sad += l_samples[i][0].abs_diff(r0) as u32
                            + l_samples[i][1].abs_diff(r1) as u32
                            + l_samples[i][2].abs_diff(r2) as u32
                            + l_samples[i][3].abs_diff(r3) as u32
                            + l_samples[i][4].abs_diff(r4) as u32;
                    }
                }

                // Match all other d in min_d..=max_d
                for d in min_d..=max_d {
                    if d == start_d {
                        continue;
                    }
                    let mut sad = 0u32;
                    let mut early_exit = false;
                    let r_base = (x as i32 - d) as usize;

                    unsafe {
                        for &i in &ROW_ORDER {
                            let r_row = right_row_offsets[i] + r_base;
                            let r0 = *right_ptr.add(r_row - 4);
                            let r1 = *right_ptr.add(r_row - 2);
                            let r2 = *right_ptr.add(r_row);
                            let r3 = *right_ptr.add(r_row + 2);
                            let r4 = *right_ptr.add(r_row + 4);

                            sad += l_samples[i][0].abs_diff(r0) as u32
                                + l_samples[i][1].abs_diff(r1) as u32
                                + l_samples[i][2].abs_diff(r2) as u32
                                + l_samples[i][3].abs_diff(r3) as u32
                                + l_samples[i][4].abs_diff(r4) as u32;

                            if sad >= best_sad {
                                early_exit = true;
                                break;
                            }
                        }
                    }

                    if !early_exit && sad < best_sad {
                        best_sad = sad;
                        best_d = d;
                    }
                }

                prev_best_d = best_d;

                // Normalize best_d from [-4, max_disp] to [0, 255]
                // formula: ((best_d + 4) * 255) / (max_disp + 4)
                let normalized = (((best_d - min_d) as u32 * 255) / d_range) as u8;
                unsafe {
                    *disp_ptr.add(row_offset + x) = normalized;
                }
            }
        }

        // Replicate nearest valid inner pixels onto border regions
        // Top and bottom borders (y < y_start || y >= y_end)
        // Left and right borders (x < x_start || x >= x_end)
        for y in 0..h {
            let cy = y.clamp(y_start, y_end - 1);
            let row_offset = y * w;
            let src_row_offset = cy * w;

            for x in 0..w {
                if y < y_start || y >= y_end || x < x_start || x >= x_end {
                    let cx = x.clamp(x_start, x_end - 1);
                    unsafe {
                        *disp_ptr.add(row_offset + x) = *disp_ptr.add(src_row_offset + cx);
                    }
                }
            }
        }
    }
}

/// Helper function to 2x2 downsample a Side-by-Side RGBA frame.
/// Inputs: full_sbs_width (e.g. 960), full_height (e.g. 240).
/// Output: RGBA buffer of size (full_sbs_width / 2) * (full_height / 2) * 4.
#[wasm_bindgen]
pub fn downsample_sbs_rgba_half(
    sbs_rgba: &[u8],
    full_sbs_width: usize,
    full_height: usize,
) -> Vec<u8> {
    let out_w = full_sbs_width / 2;
    let out_h = full_height / 2;
    let mut out = vec![0u8; out_w * out_h * 4];

    let in_stride = full_sbs_width * 4;
    let out_stride = out_w * 4;

    for y in 0..out_h {
        let in_y0 = y * 2;
        let in_y1 = in_y0 + 1;
        let in_row0 = in_y0 * in_stride;
        let in_row1 = in_y1 * in_stride;
        let out_row = y * out_stride;

        for x in 0..out_w {
            let in_x0 = x * 2;
            let in_x1 = in_x0 + 1;

            let p00 = in_row0 + in_x0 * 4;
            let p01 = in_row0 + in_x1 * 4;
            let p10 = in_row1 + in_x0 * 4;
            let p11 = in_row1 + in_x1 * 4;

            let out_idx = out_row + x * 4;

            for c in 0..4 {
                let sum = sbs_rgba[p00 + c] as u32
                    + sbs_rgba[p01 + c] as u32
                    + sbs_rgba[p10 + c] as u32
                    + sbs_rgba[p11 + c] as u32;
                out[out_idx + c] = ((sum + 2) >> 2) as u8;
            }
        }
    }

    out
}
