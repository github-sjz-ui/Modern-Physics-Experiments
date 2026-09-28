# -*- coding: utf-8 -*-
"""夫兰克-赫兹实验数据处理：读取 1-6.CVS，绘制 Ia-VG2K 曲线，寻峰并计算第一激发电位。"""
import re
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
        parts = l.split()
        if len(parts) >= 2:
            try:
                v.append(float(parts[0])); i.append(float(parts[1]))
            except ValueError:
                pass
    return np.array(v), np.array(i)

results = {}
fig, ax = plt.subplots(figsize=(10, 6))
# 黑白打印区分：线型 + 标记 + 灰度
LSTYLES = ['-', '--', '-.', ':', (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]
MARKERS = ['o', 's', '^', 'D', 'v', 'p']
COLORS = ['0.0', '0.25', '0.45', '0.6', '0.3', '0.15']

summary = []
for idx, (k, (fname, vg1, vg2a)) in enumerate(PARAMS.items()):
    v, i = load(fname)
    i_s = savgol_filter(i, 11, 3)                      # 轻度平滑用于寻峰
    peaks, _ = find_peaks(i_s, prominence=5, distance=50)
    pv, pi = v[peaks], i[peaks]
    results[k] = (v, i, pv, pi)
    dv = np.diff(pv)
    summary.append((k, vg1, vg2a, len(pv), pv, dv, dv.mean() if len(dv) else np.nan))
    ax.plot(v, i, color=COLORS[idx], lw=1.2, ls=LSTYLES[idx],
            label=f'{k}: $V_{{G1K}}$={vg1}V, $V_{{G2A}}$={vg2a}V')
    ax.plot(pv, pi, marker=MARKERS[idx], mfc='white', mec=COLORS[idx],
            color=COLORS[idx], ms=7, ls='none')

ax.set_xlabel('$V_{G2K}$ / V')
ax.set_ylabel('$I_A$ / nA')
ax.set_title('夫兰克-赫兹实验：板极电流-加速电压曲线')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('frank_hertz_curves.png', dpi=200)
plt.close(fig)

# 单图：以 D 组（VG1K=2V, VG2A=8V，峰通常最明显）为例放大标注
k, (v, i, pv, pi) = 'D', results['D']
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(v, i, 'k-', lw=1.2, label='实验数据')
ax.plot(pv, pi, 'kv', ms=9, mfc='white', label='峰位')
for n, (x, y) in enumerate(zip(pv, pi), 1):
    ax.annotate(f'{x:.1f}V', (x, y), textcoords='offset points',
                xytext=(0, 8), ha='center', fontsize=9)
ax.set_xlabel('$V_{G2K}$ / V'); ax.set_ylabel('$I_A$ / nA')
ax.set_title(f'{k} 组曲线峰位标注 ($V_{{G1K}}$=2V, $V_{{G2A}}$=8V)')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('frank_hertz_peaks.png', dpi=200)
plt.close(fig)

# 输出汇总表
with open('results_summary.txt', 'w', encoding='utf-8') as f:
    f.write('夫兰克-赫兹实验数据处理结果\n')
    f.write('条件：VF=2.1V，扫描 0-80V，步距 0.1V，拒斥电压/第一栅压见下表\n\n')
    f.write(f"{'组别':>4} {'VG1K/V':>6} {'VG2A/V':>6} {'峰数':>4}  {'峰位电压/V':<40} {'相邻峰间距/V':<35} {'平均间距/V':>8}\n")
    for k, vg1, vg2a, n, pv, dv, mean in summary:
        pv_s = ', '.join(f'{x:.2f}' for x in pv)
        dv_s = ', '.join(f'{x:.2f}' for x in dv)
        f.write(f'{k:>4} {vg1:>6} {vg2a:>6} {n:>4}  {pv_s:<40} {dv_s:<35} {mean:>8.2f}\n')
    dvs = np.concatenate([s[5] for s in summary if len(s[5])])
    f.write(f'\n全部曲线相邻峰间距总平均: {dvs.mean():.2f} V (std={dvs.std():.2f} V)\n')
    f.write(f'=> 氩原子第一激发电位 ≈ {dvs.mean():.2f} eV\n')
    f.write('(参考值: 氩第一激发态 11.55 eV, 教学仪器典型值 11.5~13 V)\n')

for k, vg1, vg2a, n, pv, dv, mean in summary:
    print(f'{k}组 VG1K={vg1}V VG2A={vg2a}V: {n} 个峰, 平均间距 {mean:.2f} V, 峰位 {np.round(pv, 1)}')
dvs = np.concatenate([s[5] for s in summary if len(s[5])])
print(f'\n总平均峰间距: {dvs.mean():.2f} ± {dvs.std():.2f} V')
