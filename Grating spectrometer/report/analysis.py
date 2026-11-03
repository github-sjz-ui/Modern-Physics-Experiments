from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "lines.linewidth": 1.0,
    "savefig.bbox": "tight",
})


def load(relative_path):
    return np.loadtxt(ROOT / relative_path)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=170)
    plt.close(fig)


def gaussian(x, center, amplitude, sigma):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def two_gaussians(x, c1, a1, s1, c2, a2, s2):
    return gaussian(x, c1, a1, s1) + gaussian(x, c2, a2, s2)


def plateau_center(x, y, left, right):
    mask = (x >= left) & (x <= right)
    xx, yy = x[mask], y[mask]
    top = yy == yy.max()
    return float(np.mean(xx[top])), float(yy.max())


hg = load("Hg-all/0.35mm.txt")
na_all = load("Na-all/0.35mm.txt")
laser = load("LASER/green.txt")

reference_lines = [
    (404.656, 402.0, 403.5),
    (407.783, 405.5, 406.7),
    (435.833, 433.8, 435.2),
    (546.074, 544.0, 547.5),
    (576.960, 576.4, 577.7),
    (579.067, 578.6, 579.8),
]
calibration = []
for reference, left, right in reference_lines:
    measured, intensity = plateau_center(hg[:, 0], hg[:, 1], left, right)
    calibration.append((reference, measured, intensity))
calibration = np.array(calibration)
reference = calibration[:, 0]
measured = calibration[:, 1]
slope, intercept = np.polyfit(measured, reference, 1)
calibrated = slope * measured + intercept
residuals = reference - calibrated
rmse = float(np.sqrt(np.mean(residuals**2)))
r_squared = 1 - np.sum(residuals**2) / np.sum((reference - reference.mean()) ** 2)

fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.6), constrained_layout=True)
axes[0].plot(hg[:, 0], hg[:, 1], color="#174a7e")
axes[0].set(xlabel="Read wavelength (nm)", ylabel="Intensity (a.u.)", xlim=(250, 660), ylim=(0, 1060))
axes[0].set_title("Mercury spectrum, 0.35 mm slit")
axes[1].plot(hg[:, 0], hg[:, 1], color="#174a7e")
for ref, raw, _ in calibration:
    axes[1].axvline(raw, color="#c44e52", alpha=0.55, linestyle="--", linewidth=0.8)
    axes[1].annotate(f"{ref:.3f}", (raw, 1020), rotation=90, ha="right", va="top", fontsize=7)
axes[1].set(xlabel="Read wavelength (nm)", ylabel="Intensity (a.u.)", xlim=(395, 590), ylim=(0, 1060))
axes[1].set_title("Visible lines used for calibration (labels: reference wavelengths)")
save(fig, "hg_spectrum")

fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}, constrained_layout=True)
xx = np.linspace(measured.min() - 3, measured.max() + 3, 300)
axes[0].plot(xx, slope * xx + intercept, color="#174a7e", label=f"$\\lambda_{{ref}}={slope:.6f}\\lambda_{{read}}+{intercept:.6f}$")
axes[0].scatter(measured, reference, color="#c44e52", zorder=3, label="Hg lines")
axes[0].set(ylabel="Reference wavelength (nm)")
axes[0].legend(loc="upper left")
axes[0].text(0.98, 0.06, f"$R^2={r_squared:.7f}$\nRMSE$={rmse:.4f}$ nm", transform=axes[0].transAxes, ha="right", va="bottom")
axes[1].axhline(0, color="black", linewidth=0.7)
axes[1].scatter(measured, residuals, color="#174a7e")
axes[1].set(xlabel="Read wavelength (nm)", ylabel="Residual (nm)", ylim=(-0.075, 0.075))
save(fig, "hg_calibration")

fig, axes = plt.subplots(2, 1, figsize=(7.1, 5.4), constrained_layout=True)
axes[0].plot(na_all[:, 0], na_all[:, 1], color="#b07503", linewidth=0.8)
axes[0].set(xlabel="Read wavelength (nm)", ylabel="Intensity (a.u.)", xlim=(450, 650), ylim=(0, 1060))
axes[0].set_title("Sodium spectrum, 0.35 mm slit")
mask = (na_all[:, 0] >= 584) & (na_all[:, 0] <= 594)
axes[1].plot(na_all[mask, 0], na_all[mask, 1], color="#b07503")
axes[1].axhline(996.3, color="#c44e52", linestyle="--", label="Saturation level")
axes[1].set(xlabel="Read wavelength (nm)", ylabel="Intensity (a.u.)", xlim=(584, 594), ylim=(0, 1060))
axes[1].set_title("Sodium yellow region: saturated and unsuitable for peak fitting")
axes[1].legend()
save(fig, "na_overview")

mask = (laser[:, 0] >= 480) & (laser[:, 0] <= 570)
lx, ly = laser[mask].T
fig, ax = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)
ax.plot(lx, ly, marker="o", markersize=2.5, color="#2a8a57")
expected_green_raw = (532.0 - intercept) / slope
ax.axvline(expected_green_raw, color="#c44e52", linestyle="--", label="Expected 532 nm position")
peak_index = np.argmax(ly)
ax.annotate(f"Maximum: {lx[peak_index]:.1f} nm, {ly[peak_index]:.1f} a.u.", (lx[peak_index], ly[peak_index]), xytext=(15, 22), textcoords="offset points", arrowprops={"arrowstyle": "->"})
ax.set(xlabel="Read wavelength (nm)", ylabel="Intensity (a.u.)", xlim=(480, 570), ylim=(0, 20))
ax.set_title("Green-laser file, valid scan segment")
ax.legend()
save(fig, "laser_check")

slit_results = []
fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.5), sharex=True, constrained_layout=True)
for index, (slit, filename) in enumerate([
    (0.5, "0.5mm.txt"),
    (1.0, "1.0.txt"),
    (1.5, "1.5mm.txt"),
    (2.0, "2.0mm.txt"),
    (2.5, "2.5mm.txt"),
]):
    data = load(f"Na-589/{filename}")
    mask = data[:, 0] >= 588
    x, y = data[mask].T
    ax = axes.flat[index]
    ax.plot(x, y, color="#174a7e", linewidth=0.9, label="Data")
    result = {"slit_mm": slit, "file": filename}
    if slit <= 1.5:
        p0 = [589.22, y.max() * 0.8, 0.17, 589.82, y.max() * 0.7, 0.18]
        bounds = ([589.0, 0, 0.02, 589.5, 0, 0.02], [589.5, 2000, 0.6, 590.1, 2000, 0.6])
        params, covariance = curve_fit(two_gaussians, x, y, p0=p0, bounds=bounds, maxfev=100000)
        fitted = two_gaussians(x, *params)
        c1, a1, s1, c2, a2, s2 = params
        result.update({
            "fit": "two_gaussians",
            "raw_D2_nm": float(c1),
            "raw_D1_nm": float(c2),
            "calibrated_D2_nm": float(slope * c1 + intercept),
            "calibrated_D1_nm": float(slope * c2 + intercept),
            "separation_nm": float(slope * (c2 - c1)),
            "fwhm_D2_nm": float(2 * np.sqrt(2 * np.log(2)) * s1),
            "fwhm_D1_nm": float(2 * np.sqrt(2 * np.log(2)) * s2),
            "fwhm_mean_nm": float(np.sqrt(2 * np.log(2)) * (s1 + s2)),
            "resolved": True,
        })
        ax.plot(x, fitted, "--", color="#c44e52", label="Two-Gaussian fit")
    else:
        p0 = [589.45, y.max(), 0.45]
        bounds = ([589.0, 0, 0.02], [590.5, 2000, 1.5])
        params, covariance = curve_fit(gaussian, x, y, p0=p0, bounds=bounds, maxfev=100000)
        fitted = gaussian(x, *params)
        center, amplitude, sigma = params
        width = float(2 * np.sqrt(2 * np.log(2)) * sigma)
        result.update({
            "fit": "single_gaussian_band",
            "raw_band_center_nm": float(center),
            "fwhm_mean_nm": width,
            "resolved": False,
        })
        ax.plot(x, fitted, "--", color="#c44e52", label="Single-band fit")
    result["fit_rmse_au"] = float(np.sqrt(np.mean((y - fitted) ** 2)))
    result["resolution"] = float(589.3 / result["fwhm_mean_nm"])
    slit_results.append(result)
    for ref in (588.9950, 589.5924):
        raw_position = (ref - intercept) / slope
        ax.axvline(raw_position, color="#555555", linestyle=":", linewidth=0.8)
    ax.set_title(f"Slit {slit:.1f} mm; FWHM = {result['fwhm_mean_nm']:.3f} nm")
    ax.set(ylabel="Intensity (a.u.)", xlim=(588, 590), ylim=(0, max(110, y.max() * 1.15)), xticks=[588, 588.5, 589, 589.5, 590])
    ax.legend(loc="upper left")
axes.flat[4].set_xlabel("Read wavelength (nm)")
axes.flat[5].axis("off")
axes.flat[5].text(0.03, 0.82, "Dotted lines: expected Na D positions\nafter inverse wavelength calibration", transform=axes.flat[5].transAxes, fontsize=9)
axes.flat[5].text(0.03, 0.57, "0.5--1.5 mm: two components resolved\n2.0--2.5 mm: fitted as one band", transform=axes.flat[5].transAxes, fontsize=9)
save(fig, "na_slits")

resolved = [r for r in slit_results if r["resolved"]]
mean_d2 = float(np.mean([r["calibrated_D2_nm"] for r in resolved]))
mean_d1 = float(np.mean([r["calibrated_D1_nm"] for r in resolved]))
mean_separation = float(np.mean([r["separation_nm"] for r in resolved]))
std_separation = float(np.std([r["separation_nm"] for r in resolved], ddof=1))
reference_separation = 589.5924 - 588.9950

fig, ax = plt.subplots(figsize=(6.8, 3.7), constrained_layout=True)
slits = [r["slit_mm"] for r in slit_results]
widths = [r["fwhm_mean_nm"] for r in slit_results]
colors = ["#174a7e" if r["resolved"] else "#c44e52" for r in slit_results]
ax.scatter(slits[:3], widths[:3], color=colors[:3], s=38, label="Mean of two components", zorder=3)
ax.scatter(slits[3:], widths[3:], color=colors[3:], s=38, marker="s", label="Whole-band fit", zorder=3)
ax.plot(slits, widths, color="#777777", linewidth=0.9)
ax.set(xlabel="Slit width (mm)", ylabel="FWHM (nm)", xlim=(0.35, 2.65), ylim=(0, 1.2), xticks=slits)
ax.set_title("Observed line width versus slit width")
ax.legend()
save(fig, "na_width")

results = {
    "calibration": {
        "slope": float(slope),
        "intercept": float(intercept),
        "rmse_nm": rmse,
        "r_squared": float(r_squared),
        "lines": [
            {"reference_nm": float(ref), "read_nm": float(raw), "intensity_au": float(intensity), "residual_nm": float(ref - (slope * raw + intercept))}
            for ref, raw, intensity in calibration
        ],
    },
    "na_doublet": {
        "mean_D2_nm": mean_d2,
        "mean_D1_nm": mean_d1,
        "mean_separation_nm": mean_separation,
        "std_separation_nm": std_separation,
        "reference_D2_nm": 588.9950,
        "reference_D1_nm": 589.5924,
        "reference_separation_nm": reference_separation,
        "D2_error_nm": mean_d2 - 588.9950,
        "D1_error_nm": mean_d1 - 589.5924,
        "separation_relative_error_percent": 100 * (mean_separation - reference_separation) / reference_separation,
    },
    "slits": slit_results,
    "laser": {
        "expected_532_raw_nm": float(expected_green_raw),
        "peak_in_valid_segment_raw_nm": float(lx[peak_index]),
        "peak_in_valid_segment_intensity_au": float(ly[peak_index]),
    },
}
with open(Path(__file__).resolve().parent / "analysis_results.json", "w", encoding="utf-8") as handle:
    json.dump(results, handle, ensure_ascii=False, indent=2)

print(json.dumps(results, ensure_ascii=False, indent=2))
