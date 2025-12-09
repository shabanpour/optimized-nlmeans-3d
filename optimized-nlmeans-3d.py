import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import time
import pandas as pd
from pathlib import Path
from scipy.ndimage import convolve

# =============================================================================
# 1. HELPER FUNCTIONS: NOISE & SCALING
# =============================================================================
def scale_to_255(volume):
    """
    Scales the volume intensity to range [0, 255].
    """
    min_val = volume.min()
    max_val = volume.max()
    if max_val - min_val == 0:
        return np.zeros_like(volume)
    # Min-Max Scaling
    return (volume - min_val) / (max_val - min_val) * 255.0

def add_gaussian_noise(volume, noise_percentage):
    """
    Adds Gaussian noise to a clean 3D volume.
    """
    # Robustly find the brightest tissue value
    non_zero = volume[volume > 0]
    brightest_tissue = np.percentile(non_zero, 99)

    # Calculate standard deviation sigma based on the scaled intensity
    sigma = (noise_percentage / 100.0) * brightest_tissue

    # Generate Gaussian noise
    noise = np.random.normal(0, sigma, volume.shape)

    # Add to volume and clip
    noisy_volume = volume + noise
    noisy_volume[noisy_volume < 0] = 0

    return noisy_volume, sigma

def get_beta_for_noise(noise_pct):
    """
    Returns the optimal Beta value based on the paper's findings.
    """
    if noise_pct <= 3:
        return 0.5  # Low noise -> Gentle smoothing
    elif noise_pct <= 9:
        return 0.8  # Medium noise -> Moderate smoothing (Paper says 0.7-1.0)
    else:
        return 1.0  # High noise -> Strong smoothing

# =============================================================================
# 2. CORE CLASS: NL-MEANS
# =============================================================================
class OptimizedBlockwiseNLMeans:
    """
    Optimized Blockwise NL-Means for 3D MRI Denoising
    Paper: Coupé et al., IEEE TMI 2008
    """

    def __init__(self, beta=1.0, f=1, M=5, n=3,
                 alpha_mean=0.80, alpha_var=0.30,
                 max_candidates=100):
        self.beta = beta
        self.f = f
        self.M = M
        self.n = n
        self.alpha_mean = alpha_mean
        self.alpha_var = alpha_var
        self.max_candidates = max_candidates

    def compute_h(self, sigma_hat):
        patch_volume = (2 * self.f + 1) ** 3
        h = self.beta * sigma_hat * np.sqrt(patch_volume)
        return h

    def precompute_statistics(self, volume_padded):
        X, Y, Z = volume_padded.shape
        f = self.f
        mean_map = np.zeros_like(volume_padded, dtype=np.float32)
        var_map = np.zeros_like(volume_padded, dtype=np.float32)

        for i in range(f, X-f):
            for j in range(f, Y-f):
                for k in range(f, Z-f):
                    patch = volume_padded[i-f:i+f+1, j-f:j+f+1, k-f:k+f+1]
                    mean_map[i, j, k] = np.mean(patch)
                    var_map[i, j, k] = np.var(patch)
        return mean_map, var_map

    def select_candidates(self, mean_ref, var_ref, mean_map, var_map, search_bounds):
        selected = []
        (x_min, x_max, y_min, y_max, z_min, z_max) = search_bounds
        mean_thresh = self.alpha_mean * np.sqrt(var_ref + 1e-10)
        var_thresh = self.alpha_var * (var_ref + 1e-10)

        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                for z in range(z_min, z_max + 1):
                    mean_comp = mean_map[x, y, z]
                    var_comp = var_map[x, y, z]
                    if abs(mean_ref - mean_comp) > mean_thresh: continue
                    if abs(var_ref - var_comp) > var_thresh: continue
                    selected.append((x, y, z))
                    if len(selected) >= self.max_candidates: return selected
        return selected

    def compute_block_weights_vectorized(self, ref_block, candidate_blocks, h):
        block_size = ref_block.shape[0]
        patch_vol = block_size ** 3
        diffs = candidate_blocks - ref_block[np.newaxis, ...]
        d2_all = np.sum(diffs ** 2, axis=(1, 2, 3)) / patch_vol
        weights = np.exp(-d2_all / (h * h))
        return weights

    def denoise(self, noisy_volume, known_sigma):
        start_time = time.time()
        X, Y, Z = noisy_volume.shape

        sigma_hat = known_sigma
        h = self.compute_h(sigma_hat)

        pad_size = self.f + self.M
        noisy_padded = np.pad(noisy_volume, pad_size, mode='symmetric')

        print(f"      Precomputing statistics...")
        mean_map, var_map = self.precompute_statistics(noisy_padded)

        denoised = np.zeros((X, Y, Z), dtype=np.float32)
        count = np.zeros((X, Y, Z), dtype=np.float32)

        block_size = 2 * self.f + 1
        processed = 0
        total_candidates = 0

        # Iterate through volume
        for i in range(0, X, self.n):
            for j in range(0, Y, self.n):
                for k in range(0, Z, self.n):
                    processed += 1
                    ix, jy, kz = i + pad_size, j + pad_size, k + pad_size

                    ref_block = noisy_padded[ix-self.f:ix+self.f+1, jy-self.f:jy+self.f+1, kz-self.f:kz+self.f+1]
                    mean_ref = mean_map[ix, jy, kz]
                    var_ref = var_map[ix, jy, kz]

                    search_bounds = (
                        max(ix - self.M, pad_size), min(ix + self.M, X + pad_size - 1),
                        max(jy - self.M, pad_size), min(jy + self.M, Y + pad_size - 1),
                        max(kz - self.M, pad_size), min(kz + self.M, Z + pad_size - 1)
                    )

                    candidates = self.select_candidates(mean_ref, var_ref, mean_map, var_map, search_bounds)
                    total_candidates += len(candidates)

                    if not candidates:
                        # Fallback copy
                        for bi in range(block_size):
                             for bj in range(block_size):
                                 for bk in range(block_size):
                                     if (0 <= i+bi-self.f < X and 0 <= j+bj-self.f < Y and 0 <= k+bk-self.f < Z):
                                        denoised[i+bi-self.f, j+bj-self.f, k+bk-self.f] += ref_block[bi, bj, bk]
                                        count[i+bi-self.f, j+bj-self.f, k+bk-self.f] += 1
                        continue

                    candidate_blocks = np.array([
                        noisy_padded[x-self.f:x+self.f+1, y-self.f:y+self.f+1, z-self.f:z+self.f+1]
                        for (x, y, z) in candidates
                    ], dtype=np.float32)

                    weights = self.compute_block_weights_vectorized(ref_block, candidate_blocks, h)
                    Z_norm = np.sum(weights)

                    if Z_norm > 0:
                        restored_block = np.sum(weights[:, None, None, None] * candidate_blocks, axis=0) / Z_norm
                    else:
                        restored_block = ref_block

                    for bi in range(block_size):
                        for bj in range(block_size):
                            for bk in range(block_size):
                                out_i, out_j, out_k = i + bi - self.f, j + bj - self.f, k + bk - self.f
                                if (0 <= out_i < X and 0 <= out_j < Y and 0 <= out_k < Z):
                                    denoised[out_i, out_j, out_k] += restored_block[bi, bj, bk]
                                    count[out_i, out_j, out_k] += 1

        mask = count > 0
        denoised[mask] /= count[mask]
        denoised[~mask] = noisy_volume[~mask]

        info = {
            'sigma': sigma_hat, 'h': h,
            'time': time.time() - start_time,
            'candidates': total_candidates / processed
        }
        return denoised, info

# =============================================================================
# 3. UTILITY & VISUALIZATION
# =============================================================================
def load_brainweb_minc(minc_file):
    print(f"Loading clean file: {minc_file}")
    img = nib.load(minc_file)
    volume = img.get_fdata()
    return volume, img.affine

def compute_psnr(ground_truth, comparison, max_val=255.0):
    mse = np.mean((ground_truth - comparison) ** 2)
    if mse == 0: return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))

def visualize_results(clean_volume, noisy_volume, denoised_volume, info,
                     noise_pct, psnr_noisy, psnr_denoised, save_dir):
    """
    Visualizes results for a specific experiment and saves them.
    """
    X, Y, Z = clean_volume.shape
    slice_idx = Z // 2

    clean_slice = clean_volume[:, :, slice_idx]
    noisy_slice = noisy_volume[:, :, slice_idx]
    denoised_slice = denoised_volume[:, :, slice_idx]

    # Setup Figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    vmin, vmax = clean_slice.min(), clean_slice.max()

    # Row 1: Full slices
    axes[0, 0].imshow(clean_slice.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 0].set_title(f'Clean Ground Truth', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_slice.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 1].set_title(f'Noisy Input ({noise_pct}%)\nPSNR: {psnr_noisy:.2f} dB', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(denoised_slice.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 2].set_title(f'Denoised Result\nPSNR: {psnr_denoised:.2f} dB', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2: Zoomed regions
    crop = min(80, X // 3, Y // 3)
    cx, cy = X // 2, Y // 2

    clean_crop = clean_slice[cx-crop:cx+crop, cy-crop:cy+crop]
    noisy_crop = noisy_slice[cx-crop:cx+crop, cy-crop:cy+crop]
    denoised_crop = denoised_slice[cx-crop:cx+crop, cy-crop:cy+crop]

    axes[1, 0].imshow(clean_crop.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[1, 0].set_title('Zoom: Clean', fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(noisy_crop.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[1, 1].set_title('Zoom: Noisy', fontsize=12)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(denoised_crop.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[1, 2].set_title('Zoom: Denoised', fontsize=12)
    axes[1, 2].axis('off')

    # Experiment Info
    info_text = (
        f"Noise Level: {noise_pct}%\n"
        f"Beta (β): {info['beta_used']} | Sigma (σ): {info['sigma']:.2f}\n"
        f"Processing Time: {info['time']:.1f}s\n"
        f"PSNR Improvement: +{psnr_denoised - psnr_noisy:.2f} dB"
    )

    plt.subplots_adjust(hspace=0.4, wspace=0.1)

    fig.suptitle(f'Experiment Result: {noise_pct}% Noise', fontsize=16, fontweight='bold', y=0.96)
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    save_path = save_dir / f"experiment_results_{noise_pct}pct.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close() # Close figure to free memory
    print(f"      ✓ Plot saved to: {save_path.name}")

def visualize_histogram_comparison(clean_volume, noisy_volume, denoised_volume, noise_pct, save_dir):
    """
    Visualizes and saves the intensity histogram comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Exclude background for histogram (assumes 0 is background or very low values)
    mask = clean_volume > 5

    bins = 100
    ax.hist(clean_volume[mask].flatten(), bins=bins, alpha=0.5, label='Clean', color='green', density=True, histtype='step', linewidth=2)
    ax.hist(noisy_volume[mask].flatten(), bins=bins, alpha=0.5, label='Noisy', color='red', density=True, histtype='step')
    ax.hist(denoised_volume[mask].flatten(), bins=bins, alpha=0.5, label='Denoised', color='blue', density=True, histtype='step', linewidth=2)

    ax.set_xlabel('Intensity (0-255)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Intensity Histogram Comparison ({noise_pct}% Noise)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    save_path = save_dir / f"histogram_comparison_{noise_pct}pct.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      ✓ Histogram saved to: {save_path.name}")

# =============================================================================
# 4. MAIN EXPERIMENT LOOP
# =============================================================================
def run_experiments(input_file):

    # Load and scale clean image once
    clean_vol_raw, affine = load_brainweb_minc(input_file)
    clean_vol = scale_to_255(clean_vol_raw)

    noise_levels = [3, 9, 15, 21]
    results = []

    output_dir = Path("denoising_experiments")
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*80)
    print(f"STARTING COMPREHENSIVE EXPERIMENTS ({len(noise_levels)} levels)")
    print("="*80)

    for noise_pct in noise_levels:
        print(f"\n>> Running Experiment: {noise_pct}% Noise")

        # 1. Determine Dynamic Beta
        beta = get_beta_for_noise(noise_pct)
        print(f"   Dynamic Beta Selected: {beta}")

        # 2. Add Noise
        noisy_vol, true_sigma = add_gaussian_noise(clean_vol, noise_pct)

        # 3. Calculate Baseline PSNR
        psnr_noisy = compute_psnr(clean_vol, noisy_vol)
        print(f"   Baseline PSNR (Noisy): {psnr_noisy:.2f} dB")

        # 4. Configure Filter with Dynamic Beta
        filter_obj = OptimizedBlockwiseNLMeans(
            beta=beta, f=1, M=5, n=2,
            alpha_mean=0.70, alpha_var=0.25,
            max_candidates=80
        )

        # 5. Run Denoising
        denoised_vol, info = filter_obj.denoise(noisy_vol, known_sigma=true_sigma)

        # 6. Calculate Result Metrics
        psnr_denoised = compute_psnr(clean_vol, denoised_vol)
        improvement = psnr_denoised - psnr_noisy

        print(f"   Result PSNR (Denoised): {psnr_denoised:.2f} dB")
        print(f"   Improvement: +{improvement:.2f} dB")

        # 7. Add extra info for logging
        info['beta_used'] = beta

        # 8. Visualizations
        visualize_results(clean_vol, noisy_vol, denoised_vol, info,
                         noise_pct, psnr_noisy, psnr_denoised, output_dir)
        visualize_histogram_comparison(clean_vol, noisy_vol, denoised_vol, noise_pct, output_dir)

        # 9. Store Data
        results.append({
            'Noise Level (%)': noise_pct,
            'Beta': beta,
            'Sigma': true_sigma,
            'PSNR Noisy (dB)': psnr_noisy,
            'PSNR Denoised (dB)': psnr_denoised,
            'Improvement (dB)': improvement,
            'Time (s)': info['time']
        })

    # =============================================================================
    # 5. FINAL SUMMARY
    # =============================================================================
    print("\n" + "="*80)
    print("FINAL EXPERIMENT SUMMARY")
    print("="*80)

    df = pd.DataFrame(results)

    # Format the table for nice printing
    print(df[['Noise Level (%)', 'Beta', 'PSNR Noisy (dB)', 'PSNR Denoised (dB)', 'Improvement (dB)', 'Time (s)']].to_string(index=False, float_format="%.2f"))

    # Save CSV
    csv_path = output_dir / "summary_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Summary CSV saved to: {csv_path}")
    print(f"✓ All plots saved in: {output_dir.absolute()}")

if __name__ == "__main__":
    # Point this to your CLEAN file
    input_file = '/content/t1_icbm_normal_1mm_pn0_rf0.mnc.gz'

    run_experiments(input_file)