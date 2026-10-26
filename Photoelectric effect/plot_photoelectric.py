from pathlib import Path
import csv

import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
VOLTAGE = [-1, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 50]

SERIES = [
    ("波长扫描：365.0 nm，Φ4", "365.0", 4, 1, [0.01, 0.89, 1.88, 2.71, 3.45, 4.00, 4.51, 4.98, 5.36, 5.67, 5.94, 6.19, 6.40, 6.55], "#6a3d9a", "-", "o"),
    ("波长扫描：404.7 nm，Φ4", "404.7", 4, 1, [-0.02, 0.27, 0.49, 0.70, 0.85, 0.96, 1.07, 1.17, 1.24, 1.30, 1.35, 1.39, 1.44, 1.46], "#1f78b4", "-", "s"),
    ("波长扫描：577.0 nm，Φ4", "577.0", 4, 1, [-0.03, 0.11, 0.19, 0.27, 0.31, 0.35, 0.38, 0.41, 0.43, 0.44, 0.45, 0.47, 0.48, 0.48], "#b89b00", "-", "^"),
    ("波长扫描：546.1 nm，Φ4", "546.1", 4, 1, [-0.03, 0.42, 0.72, 1.00, 1.20, 1.35, 1.47, 1.57, 1.64, 1.70, 1.76, 1.81, 1.85, 1.87], "#33a02c", "-", "D"),
    ("孔径扫描：404.7 nm，Φ2", "404.7", 2, 2, [-0.03, 0.06, 0.12, 0.19, 0.23, 0.26, 0.30, 0.33, 0.35, 0.36, 0.37, 0.39, 0.40, 0.41], "#555555", "--", "o"),
    ("孔径扫描：404.7 nm，Φ4", "404.7", 4, 2, [-0.03, 0.28, 0.52, 0.74, 0.90, 1.02, 1.14, 1.24, 1.32, 1.38, 1.43, 1.49, 1.53, 1.56], "#1f78b4", "--", "s"),
    ("孔径扫描：404.7 nm，Φ8", "404.7", 8, 2, [-0.02, 0.95, 1.93, 2.79, 3.52, 4.05, 4.52, 4.95, 5.29, 5.57, 5.80, 6.00, 6.18, 6.31], "#e31a1c", "--", "^"),
]


def configure_fonts():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def export_data():
    with (ROOT / "photoelectric_data.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["组别", "波长_nm", "光阑", "测量轮次", "电压_V", "电流_仪器示数"])
        for label, wavelength, aperture, run, currents, *_ in SERIES:
            for voltage, current in zip(VOLTAGE, currents):
                writer.writerow([label, wavelength, f"Φ{aperture}", run, voltage, f"{current:.2f}"])


def plot_data():
    configure_fonts()
    fig, ax = plt.subplots(figsize=(11, 7.2), constrained_layout=True)
    for label, _, _, _, currents, color, linestyle, marker in SERIES:
        ax.plot(
            VOLTAGE,
            currents,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.0,
            linewidth=1.7,
        )
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.45)
    ax.axvline(0, color="black", linewidth=0.7, alpha=0.45)
    ax.set_xlabel("电压 $U$ / V", fontsize=12)
    ax.set_ylabel("电流 $I$（仪器示数）", fontsize=12)
    ax.set_title("光电效应伏安特性：波长与光阑孔径的影响", fontsize=15)
    ax.set_xlim(-3, 52)
    ax.set_ylim(-0.25, 7.0)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95, ncol=1)
    fig.savefig(ROOT / "photoelectric_all_data.png", dpi=300, bbox_inches="tight")
    fig.savefig(ROOT / "photoelectric_all_data.pdf", bbox_inches="tight")


if __name__ == "__main__":
    export_data()
    plot_data()
    i2, i4, i8 = (SERIES[i][4][-1] for i in (4, 5, 6))
    print(f"U=50 V: Phi4/Phi2={i4/i2:.3f}, Phi8/Phi2={i8/i2:.3f}")
    print(f"Deviation from Phi^2: Phi4={(i4/(4*i2)-1)*100:.2f}%, Phi8={(i8/(16*i2)-1)*100:.2f}%")
