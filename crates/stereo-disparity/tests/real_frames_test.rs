use std::fs;
use std::path::Path;
use std::time::Instant;
use stereo_disparity::{downsample_sbs_rgba_half, StereoDisparity};

/// Helper to generate pseudo-random hash texture for synthetic disparity tests
fn hash_texture(x: i32, y: i32) -> u8 {
    let mut n = (x as u32).wrapping_mul(374761393) ^ (y as u32).wrapping_mul(668265263);
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    (n ^ (n >> 16)) as u8
}

#[test]
fn test_synthetic_known_shifts() {
    let eye_w = 480usize;
    let eye_h = 240usize;
    let sbs_w = eye_w * 2;
    let max_disp = 24i32;
    let min_disp = -4i32;

    let test_shifts = [0i32, 8i32, 16i32, -2i32];

    for &shift in &test_shifts {
        let mut sbs_rgba = vec![0u8; sbs_w * eye_h * 4];

        for y in 0..eye_h {
            let row_offset = y * sbs_w * 4;
            for x in 0..eye_w {
                // Left eye
                let val_left = hash_texture(x as i32, y as i32);
                let l_idx = row_offset + x * 4;
                sbs_rgba[l_idx] = val_left;
                sbs_rgba[l_idx + 1] = val_left;
                sbs_rgba[l_idx + 2] = val_left;
                sbs_rgba[l_idx + 3] = 255;

                // Right eye shifted by `shift`: right(x, y) = left(x + shift, y)
                let val_right = hash_texture(x as i32 + shift, y as i32);
                let r_idx = row_offset + (eye_w + x) * 4;
                sbs_rgba[r_idx] = val_right;
                sbs_rgba[r_idx + 1] = val_right;
                sbs_rgba[r_idx + 2] = val_right;
                sbs_rgba[r_idx + 3] = 255;
            }
        }

        let mut engine = StereoDisparity::new(eye_w, eye_h, max_disp);
        let disp_map = engine.compute_disparity(&sbs_rgba);

        assert_eq!(disp_map.len(), eye_w * eye_h);

        let expected_norm = (((shift - min_disp) as u32 * 255) / (max_disp - min_disp) as u32) as u8;

        // Verify across interior region
        let x_start = (max_disp + 4) as usize + 2;
        let x_end = eye_w - 12;
        let y_start = 8;
        let y_end = eye_h - 8;

        let mut match_count = 0;
        let mut total_interior = 0;

        for y in y_start..y_end {
            for x in x_start..x_end {
                let actual = disp_map[y * eye_w + x];
                if actual == expected_norm {
                    match_count += 1;
                }
                total_interior += 1;
            }
        }

        let accuracy = match_count as f64 / total_interior as f64;
        println!(
            "Shift d = {:2} -> expected norm = {:3}, interior accuracy: {:.2}% ({}/{})",
            shift,
            expected_norm,
            accuracy * 100.0,
            match_count,
            total_interior
        );

        assert!(
            accuracy > 0.99,
            "Accuracy for shift {} was {:.2}%, expected > 99%",
            shift,
            accuracy * 100.0
        );
    }
}

fn calculate_stats(data: &[u8]) -> (f64, f64, usize) {
    let sum: f64 = data.iter().map(|&v| v as f64).sum();
    let mean = sum / data.len() as f64;
    let variance: f64 = data
        .iter()
        .map(|&v| {
            let diff = v as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / data.len() as f64;
    let std_dev = variance.sqrt();

    let mut distinct = std::collections::HashSet::new();
    for &v in data {
        distinct.insert(v);
    }

    (mean, std_dev, distinct.len())
}

#[test]
fn test_real_dcim_frames() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixtures_dir = manifest_dir.join("../../tests/fixtures");

    let fixture_files = [
        ("hni_0060_frame10.rgba", fixtures_dir.join("hni_0060_frame10.rgba")),
        ("hni_0070_frame05.rgba", fixtures_dir.join("hni_0070_frame05.rgba")),
    ];

    if !fixture_files.iter().all(|(_, p)| p.exists()) {
        println!("Private DCIM fixtures not present in repo. Skipping real frame benchmarks.");
        return;
    }

    for (name, path) in &fixture_files {
        println!("\n=======================================================");
        println!("Testing real 3DS DCIM frame: {}", name);

        let raw_rgba = fs::read(path).expect("Failed to read fixture");
        assert_eq!(
            raw_rgba.len(),
            960 * 240 * 4,
            "Expected 960x240x4 = 921,600 bytes"
        );

        // 1. Full Resolution Benchmark (480x240 per eye, max_disp = 24)
        {
            let mut full_engine = StereoDisparity::new(480, 240, 24);

            // Warm-up
            let _ = full_engine.compute_disparity(&raw_rgba);

            let iterations = 10;
            let start = Instant::now();
            let mut disp = Vec::new();
            for _ in 0..iterations {
                disp = full_engine.compute_disparity(&raw_rgba);
            }
            let elapsed = start.elapsed();
            let per_frame_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

            let (mean, std_dev, distinct_count) = calculate_stats(&disp);
            println!(
                "Full Res (480x240, max_disp=24): {:.2} ms/frame | Mean: {:.2}, StdDev: {:.2}, Distinct levels: {}",
                per_frame_ms, mean, std_dev, distinct_count
            );

            assert!(
                std_dev > 5.0,
                "Expected standard deviation > 5.0, got {:.2}",
                std_dev
            );
            assert!(
                distinct_count >= 10,
                "Expected >= 10 distinct depth levels, got {}",
                distinct_count
            );
        }

        // 2. Half Resolution Benchmark (240x120 per eye, max_disp = 12)
        {
            let mut half_engine = StereoDisparity::new(240, 120, 12);

            // Direct pass of full-res buffer with internal on-the-fly 2x2 downsampling
            // Warm-up
            let _ = half_engine.compute_disparity(&raw_rgba);

            let iterations = 20;
            let start = Instant::now();
            let mut half_disp = Vec::new();
            for _ in 0..iterations {
                half_disp = half_engine.compute_disparity(&raw_rgba);
            }
            let elapsed = start.elapsed();
            let per_frame_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

            let (mean, std_dev, distinct_count) = calculate_stats(&half_disp);
            println!(
                "Half Res (240x120, max_disp=12): {:.2} ms/frame | Mean: {:.2}, StdDev: {:.2}, Distinct levels: {}",
                per_frame_ms, mean, std_dev, distinct_count
            );

            assert!(
                std_dev > 5.0,
                "Expected half-res standard deviation > 5.0, got {:.2}",
                std_dev
            );
            assert!(
                distinct_count >= 8,
                "Expected >= 8 distinct depth levels, got {}",
                distinct_count
            );

            // Also test helper function downsample_sbs_rgba_half and measure direct half-res input speed
            let pre_downsampled = downsample_sbs_rgba_half(&raw_rgba, 960, 240);
            assert_eq!(pre_downsampled.len(), 480 * 120 * 4);
            let disp2 = half_engine.compute_disparity(&pre_downsampled);
            assert_eq!(disp2.len(), 240 * 120);

            let bench_start = Instant::now();
            let bench_iters = 50;
            for _ in 0..bench_iters {
                let _ = half_engine.compute_disparity_inplace(&pre_downsampled);
            }
            let bench_ms = bench_start.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;
            println!("Half Res direct input (240x120, inplace): {:.2} ms/frame", bench_ms);
            assert!(bench_ms < 3.0, "Expected bench_ms < 3.0, got {:.2}", bench_ms);
        }
    }
}
