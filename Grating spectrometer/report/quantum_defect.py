from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

with open(HERE / "analysis_results.json", encoding="utf-8") as handle:
    calibration = json.load(handle)["calibration"]
SLOPE = calibration["slope"]
INTERCEPT = calibration["intercept"]
CALIBRATION_RMSE = calibration["rmse_nm"]

R_INFINITY = 109737.315685
NA_MASS_U = 22.9897692820
ELECTRON_PROTON_MASS_RATIO = 1 / 1836.15267343
R_NA = R_INFINITY / (1 + ELECTRON_PROTON_MASS_RATIO / NA_MASS_U)
IONIZATION_LIMIT = 41449.451
E_3P_12 = 16956.172
E_3P_32 = 16973.368
E_3P_MEAN = (E_3P_12 + E_3P_32) / 2

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "savefig.bbox": "tight",
})


def calibrated(raw_wavelength):
    return SLOPE * raw_wavelength + INTERCEPT


def two_gaussians_with_background(x, c1, a1, s1, c2, a2, s2, background):
    first = a1 * np.exp(-0.5 * ((x - c1) / s1) ** 2)
    second = a2 * np.exp(-0.5 * ((x - c2) / s2) ** 2)
    return background + first + second


def fit_short_component(x, y, lower, upper, p0, bounds):
    mask = (x >= lower) & (x <= upper)
    xx, yy = x[mask], y[mask]
    params, covariance = curve_fit(
        two_gaussians_with_background, xx, yy, p0=p0, bounds=bounds, maxfev=100000
    )
    fitted = two_gaussians_with_background(xx, *params)
    rmse = float(np.sqrt(np.mean((yy - fitted) ** 2)))
    return float(params[0]), float(np.sqrt(covariance[0, 0])), rmse


def plateau_center(x, y, lower, upper):
    mask = (x >= lower) & (x <= upper)
    xx, yy = x[mask], y[mask]
    highest = np.flatnonzero(yy == yy.max())
    return float((xx[highest[0]] + xx[highest[-1]]) / 2)


def delta_from_wavelength(n, wavelength, lower_energy):
    wavenumber = 1e7 / wavelength
    term = IONIZATION_LIMIT - lower_energy - wavenumber
    n_effective = np.sqrt(R_NA / term)
    delta = n - n_effective
    delta_per_nm = n_effective * 1e7 / (2 * term * wavelength**2)
    return wavenumber, term, n_effective, delta, delta_per_nm


def ritz(n, delta_0, coefficient):
    return delta_0 + coefficient / (n - delta_0) ** 2


data = np.loadtxt(ROOT / "Na-all/0.35mm.txt")
x, y = data.T
rows = []

s_measurements = {
    5: (plateau_center(x, y, 615.8, 616.2), 0.0, 0.0, 615.423),
    6: (plateau_center(x, y, 514.5, 514.9), 0.0, 0.0, 515.340),
    7: (*fit_short_component(
        x, y, 473.2, 474.55,
        [473.84, 230, 0.15, 474.05, 200, 0.15, 15],
        ([473.6, 0, 0.03, 473.8, 0, 0.03, 0], [474.1, 600, 0.5, 474.4, 600, 0.5, 100]),
    ), 474.794),
    8: (*fit_short_component(
        x, y, 452.35, 453.55,
        [452.98, 95, 0.15, 453.19, 85, 0.15, 15],
        ([452.7, 0, 0.03, 452.9, 0, 0.03, 0], [453.3, 300, 0.5, 453.5, 300, 0.5, 100]),
    ), 454.163),
}

for n, (raw_center, center_error, fit_rmse, reference_wavelength) in s_measurements.items():
    wavelength = calibrated(raw_center)
    wavenumber, term, n_effective, delta, derivative = delta_from_wavelength(n, wavelength, E_3P_32)
    _, _, _, reference_delta, _ = delta_from_wavelength(n, reference_wavelength, E_3P_32)
    wavelength_uncertainty = np.hypot(CALIBRATION_RMSE, SLOPE * center_error)
    rows.append({
        "series": "s",
        "n": n,
        "transition": f"3p(3/2)-{n}s",
        "raw_wavelength_nm": raw_center,
        "measured_wavelength_nm": wavelength,
        "reference_wavelength_nm": reference_wavelength,
        "wavenumber_cm-1": wavenumber,
        "term_energy_cm-1": term,
        "n_effective": n_effective,
        "quantum_defect": delta,
        "reference_quantum_defect": reference_delta,
        "delta_minus_reference": delta - reference_delta,
        "calibration_uncertainty_delta": derivative * wavelength_uncertainty,
        "fit_rmse_au": fit_rmse,
        "method": "short-wavelength component; plateau center" if n <= 6 else "short-wavelength component; two-Gaussian fit",
    })

for n, lower, upper, reference_short, reference_long in [
    (4, 567.0, 569.9, 568.263, 568.822),
    (5, 495.5, 498.7, 497.854, 498.287),
    (6, 464.5, 466.7, 466.486, 466.859),
]:
    raw_center = plateau_center(x, y, lower, upper)
    wavelength = calibrated(raw_center)
    wavenumber, term, n_effective, delta, derivative = delta_from_wavelength(n, wavelength, E_3P_MEAN)
    reference_wavenumber = (1e7 / reference_short + 1e7 / reference_long) / 2
    reference_wavelength = 1e7 / reference_wavenumber
    _, _, _, reference_delta, _ = delta_from_wavelength(n, reference_wavelength, E_3P_MEAN)
    rows.append({
        "series": "d",
        "n": n,
        "transition": f"3p(mean)-{n}d",
        "raw_wavelength_nm": raw_center,
        "measured_wavelength_nm": wavelength,
        "reference_wavelength_nm": reference_wavelength,
        "wavenumber_cm-1": wavenumber,
        "term_energy_cm-1": term,
        "n_effective": n_effective,
        "quantum_defect": delta,
        "reference_quantum_defect": reference_delta,
        "delta_minus_reference": delta - reference_delta,
        "calibration_uncertainty_delta": derivative * CALIBRATION_RMSE,
        "fit_rmse_au": None,
        "method": "saturated multiplet; plateau midpoint and mean lower level",
    })

s_rows = [row for row in rows if row["series"] == "s"]
d_rows = [row for row in rows if row["series"] == "d"]
n_s = np.array([row["n"] for row in s_rows], dtype=float)
delta_s = np.array([row["quantum_defect"] for row in s_rows])
(delta_0, coefficient), covariance = curve_fit(ritz, n_s, delta_s, p0=[1.30, 0.5], maxfev=100000)
ritz_errors = np.sqrt(np.diag(covariance))

results = {
    "constants": {
        "R_Na_cm-1": R_NA,
        "ionization_limit_cm-1": IONIZATION_LIMIT,
        "E_3p_1/2_cm-1": E_3P_12,
        "E_3p_3/2_cm-1": E_3P_32,
        "E_3p_mean_cm-1": E_3P_MEAN,
        "calibration_rmse_nm": CALIBRATION_RMSE,
    },
    "s_series": {
        "mean_quantum_defect": float(np.mean(delta_s)),
        "std_quantum_defect": float(np.std(delta_s, ddof=1)),
        "ritz_delta_0": float(delta_0),
        "ritz_delta_0_stderr": float(ritz_errors[0]),
        "ritz_coefficient": float(coefficient),
        "ritz_coefficient_stderr": float(ritz_errors[1]),
        "ritz_rmse": float(np.sqrt(np.mean((delta_s - ritz(n_s, delta_0, coefficient)) ** 2))),
    },
    "d_series": {
        "mean_quantum_defect": float(np.mean([row["quantum_defect"] for row in d_rows])),
        "std_quantum_defect": float(np.std([row["quantum_defect"] for row in d_rows], ddof=1)),
    },
    "lines": rows,
}

with open(HERE / "quantum_defect_results.json", "w", encoding="utf-8") as handle:
    json.dump(results, handle, ensure_ascii=False, indent=2)
with open(HERE / "quantum_defect.csv", "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

fig = plt.figure(figsize=(7.2, 6.7), constrained_layout=True)
grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
ax = fig.add_subplot(grid[0, :])
wavelengths = calibrated(x)
ax.plot(wavelengths, y, color="#b07503", linewidth=0.8)
for row in rows:
    position = row["measured_wavelength_nm"]
    label = f"{row['n']}{row['series']}"
    ax.axvline(position, color="#555555", linestyle=":", linewidth=0.7)
    ax.annotate(label, (position, 1030), ha="center", va="top", fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})
ax.set(xlabel="Calibrated wavelength (nm)", ylabel="Intensity (a.u.)", xlim=(450, 625), ylim=(0, 1080))
ax.set_title("Na I Rydberg series used for quantum-defect analysis")

ax_s = fig.add_subplot(grid[1, 0])
n_fine = np.linspace(4.8, 8.2, 300)
ax_s.plot(n_fine, ritz(n_fine, delta_0, coefficient), color="#174a7e", label=f"Ritz fit: $\\delta_0={delta_0:.4f}$")
ax_s.scatter(n_s, delta_s, color="#174a7e", s=30, zorder=3, label="Measured")
ax_s.scatter(n_s, [row["reference_quantum_defect"] for row in s_rows], marker="x", color="#c44e52", s=35, zorder=3, label="Reference")
ax_s.set(xlabel="Principal quantum number $n$", ylabel="$\\delta_s$", xticks=n_s, ylim=(1.29, 1.37))
ax_s.set_title("Sharp series: $3p\\rightarrow ns$")
ax_s.legend(loc="upper right")

ax_d = fig.add_subplot(grid[1, 1])
n_d = np.array([row["n"] for row in d_rows])
delta_d = np.array([row["quantum_defect"] for row in d_rows])
ax_d.scatter(n_d, delta_d, color="#174a7e", s=30, zorder=3, label="Multiplet estimate")
ax_d.scatter(n_d, [row["reference_quantum_defect"] for row in d_rows], marker="x", color="#c44e52", s=35, zorder=3, label="Reference")
ax_d.axhline(delta_d.mean(), color="#555555", linestyle="--", label=f"Mean = {delta_d.mean():.5f}")
ax_d.set(xlabel="Principal quantum number $n$", ylabel="$\\delta_d$", xticks=n_d, ylim=(0, 0.013))
ax_d.set_title("Diffuse series: $3p\\rightarrow nd$")
ax_d.legend(loc="upper right")

fig.savefig(OUT / "na_quantum_defect.pdf")
fig.savefig(OUT / "na_quantum_defect.png", dpi=180)
plt.close(fig)

print(json.dumps(results, ensure_ascii=False, indent=2))
