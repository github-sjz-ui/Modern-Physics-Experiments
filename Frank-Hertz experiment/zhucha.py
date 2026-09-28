# -*- coding: utf-8 -*-
"""逐差法计算氩原子第一激发电位 + 峰位线性拟合图"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import find_peaks, savgol_filter

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

PARAMS = {  # 组名: (源文件, VG1K, VG2A)
    'A': ('2.CVS', 1, 7), 'B': ('3.CVS', 1, 8),
    'C': ('1.CVS', 2, 7), 'D': ('4.CVS', 2, 8),
    'E': ('6.CVS', 3, 7), 'F': ('5.CVS', 3, 8),
}

def load(fname):
    with open(fname, encoding='gbk') as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if '导出实验数据' in l)
    v, i = [], []
    for l in lines[start + 2:]:
        p = l.split()
        if len(p) >= 2:
            try:
                v.append(float(p[0])); i.append(float(p[1]))
            except ValueError:
                pass
    return np.array(v), np.array(i)

def peaks_of(v, i):
    i_s = savgol_filter(i, 11, 3)
    pk, _ = find_peaks(i_s, prominence=5, distance=50)
    return v[pk]

def successive_diff(p):
    """对 n 个峰位做逐差: 隔 k 项差分 (p_{i+k}-p_i)/k, 取所有 k>=2 的估计"""
    est = []
    for k in range(2, len(p)):
        for j in range(len(p) - k):
            est.append((p[j + k] - p[j]) / k)
    return np.array(est)

print('========== 各曲线峰位 (V) ==========')
all_peaks = {}
for k, (fname, vg1, vg2a) in PARAMS.items():
    v, i = load(fname)
    p = peaks_of(v, i)
    all_peaks[k] = p
    est = successive_diff(p)
    print(f'{k}组 VG1K={vg1}V VG2A={vg2a}V: {np.round(p, 2)}')
    print(f'    逐差估计({len(est)}个): {np.round(est, 3)}  均值={est.mean():.3f}  标准差={est.std(ddof=1):.3f}')

# 六条曲线峰位取平均后再逐差
pm = np.mean([all_peaks[k] for k in PARAMS], axis=0)
print(f'\n平均峰位: {np.round(pm, 3)}')
est_m = successive_diff(pm)
print(f'平均峰位逐差: 均值={est_m.mean():.4f} V')

# 全部 30 个峰位统一逐差（合并 6 条曲线）
est_all = np.concatenate([successive_diff(all_peaks[k]) for k in PARAMS])
U = est_all.mean()
s = est_all.std(ddof=1)
ua = s / np.sqrt(len(est_all))
print(f'\n========== 合并逐差结果 ==========')
print(f'估计值个数 n = {len(est_all)}')
print(f'ΔU = {U:.4f} V, 样本标准差 s = {s:.4f} V')
print(f'A类不确定度 u_A = s/√n = {ua:.4f} V')
print(f'相对误差 (对 11.55 eV): {abs(U - 11.55) / 11.55 * 100:.2f}%')
print(f'结果: ΔU = {U:.2f} ± {ua:.2f} V')

# 线性拟合图: 峰序号 n vs 峰位电压, 斜率即 ΔU (黑白打印: 线型+标记区分)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
LSTYLES = ['-', '--', '-.', ':', (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]
MARKERS = ['o', 's', '^', 'D', 'v', 'p']
COLORS = ['0.0', '0.25', '0.45', '0.6', '0.3', '0.15']
ns = np.arange(1, 6)
slopes = []
for idx, k in enumerate(PARAMS):
    p = all_peaks[k]
    a, b = np.polyfit(ns, p, 1)
    slopes.append(a)
    vg1, vg2a = PARAMS[k][1], PARAMS[k][2]
    ax1.plot(ns, p, marker=MARKERS[idx], mfc='white', mec=COLORS[idx],
             color=COLORS[idx], ms=6, lw=1.2, ls=LSTYLES[idx],
             label=f'{k}组 ({vg1}V,{vg2a}V) 斜率 {a:.2f} V')
ax1.plot(ns, np.polyval(np.polyfit(ns, pm, 1), ns), 'k--', lw=2,
         label=f'平均峰位拟合 ({np.polyfit(ns, pm, 1)[0]:.2f} V)')
ax1.set_xlabel('峰序号 $n$'); ax1.set_ylabel('峰位电压 $U_n$ / V')
ax1.set_title('峰位电压与峰序号的线性关系'); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

res = pm - np.polyval(np.polyfit(ns, pm, 1), ns)
ax2.bar(ns, res, color='0.35', edgecolor='black', width=0.5)
ax2.axhline(0, color='k', lw=0.8)
for n, r in zip(ns, res):
    ax2.text(n, r + (0.03 if r > 0 else -0.08), f'{r:+.2f}', ha='center', fontsize=9)
ax2.set_xlabel('峰序号 $n$'); ax2.set_ylabel('残差 / V')
ax2.set_title('平均峰位拟合残差'); ax2.grid(alpha=0.3, axis='y')

fig.tight_layout()
fig.savefig('frank_hertz_linear_fit.png', dpi=200)
plt.close(fig)
print(f'\n各曲线拟合斜率: {np.round(slopes, 3)}, 平均 {np.mean(slopes):.3f} V')
