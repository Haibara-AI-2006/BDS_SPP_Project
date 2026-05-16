"""
trajectory_analyzer.py — 精度评估、报表输出与满分图表绘制引擎
==============================================================
核心职责:
  1. 从 SPP 解算结果计算 ENU 误差、RMS/均值/最大误差
  2. 绘制实验大纲满分图表全家桶:
     - ENU 三分量误差时序图 (含 RMS 标注)
     - 水平 (E-N) 误差散点图 + 68%/95% CEP 概率圆
     - DOP 值 (GDOP/PDOP/HDOP/VDOP) 时序图
     - 卫星可用数时序图
     - 3D 定位误差时序图
     - 综合精度评估报告
  3. 提供统一接口, 支持 GUI 画布嵌入与本地文件存储

设计原则:
  - 全部绘图使用 Matplotlib, 支持中文字体
  - 统一接收 DataFrame 或 List[EpochSolution]
  - 高清输出 (dpi=150+)
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.ticker import MaxNLocator
from datetime import datetime
from typing import List, Optional, Tuple, Dict

from rinex_parser import WGS84_A, WGS84_E2, ecef_to_blh

# ============================================================
#  中文字体设置 (兼容多平台)
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                    'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
#  真值锚定
# ============================================================
GT_LAT_DEG = 40.1575
GT_LON_DEG = 116.2885
GT_HEIGHT = 35.0
GT_LAT_RAD = math.radians(GT_LAT_DEG)
GT_LON_RAD = math.radians(GT_LON_DEG)


def _blh_to_ecef(lat_rad: float, lon_rad: float, h: float) -> np.ndarray:
    """WGS84 BLH → ECEF"""
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    return np.array([
        (N + h) * cos_lat * cos_lon,
        (N + h) * cos_lat * sin_lon,
        (N * (1.0 - WGS84_E2) + h) * sin_lat,
    ])


def _ecef_to_enu_matrix(lat_rad: float, lon_rad: float) -> np.ndarray:
    """ECEF → ENU 旋转矩阵"""
    sl, cl = math.sin(lat_rad), math.cos(lat_rad)
    sn, cn = math.sin(lon_rad), math.cos(lon_rad)
    return np.array([
        [-sn,       cn,       0.0],
        [-sl * cn, -sl * sn,  cl ],
        [ cl * cn,  cl * sn,  sl ],
    ])


GT_ECEF = _blh_to_ecef(GT_LAT_RAD, GT_LON_RAD, GT_HEIGHT)
R_ENU = _ecef_to_enu_matrix(GT_LAT_RAD, GT_LON_RAD)


# ============================================================
#  轨迹分析器
# ============================================================
class TrajectoryAnalyzer:
    """
    北斗 SPP 定位精度评估 & 图表绘制引擎

    使用:
        analyzer = TrajectoryAnalyzer(solutions, output_dir='data')
        analyzer.compute_errors()
        analyzer.plot_all()
        report = analyzer.generate_report()
    """

    def __init__(
        self,
        solutions: Optional[List] = None,
        output_dir: str = 'data',
        scene_label: str = '开阔天空',
    ):
        self.solutions = solutions or []
        self.output_dir = output_dir
        self.scene_label = scene_label
        os.makedirs(output_dir, exist_ok=True)

        # 分析结果缓存
        self.df: Optional[pd.DataFrame] = None
        self.enu_errors: Optional[np.ndarray] = None

    # ================================================================
    #  从 solutions 列表构建 DataFrame
    # ================================================================
    def compute_errors(self) -> pd.DataFrame:
        """
        从 EpochSolution 列表中提取数据并计算 ENU 误差

        Returns:
            包含所有分析数据的 DataFrame
        """
        records = []
        enu_list = []

        for sol in self.solutions:
            if not sol.valid:
                continue

            # ECEF 误差
            dx_ecef = sol.pos_ecef - GT_ECEF
            # 转 ENU
            enu = R_ENU @ dx_ecef
            enu_list.append(enu)

            # 3D 误差
            err_3d = np.linalg.norm(dx_ecef)
            # 水平误差
            err_h = math.sqrt(enu[0] ** 2 + enu[1] ** 2)

            records.append({
                'epoch': sol.epoch,
                'x_ecef': sol.pos_ecef[0],
                'y_ecef': sol.pos_ecef[1],
                'z_ecef': sol.pos_ecef[2],
                'lat_deg': sol.lat_deg,
                'lon_deg': sol.lon_deg,
                'height': sol.height,
                'clock_bias': sol.clock_bias,
                'n_sats': sol.n_used,
                'gdop': sol.dop.gdop,
                'pdop': sol.dop.pdop,
                'hdop': sol.dop.hdop,
                'vdop': sol.dop.vdop,
                'tdop': sol.dop.tdop,
                'sigma0': sol.sigma0,
                'err_e': enu[0],
                'err_n': enu[1],
                'err_u': enu[2],
                'err_h': err_h,
                'err_3d': err_3d,
            })

        self.df = pd.DataFrame(records)
        self.enu_errors = np.array(enu_list) if enu_list else np.empty((0, 3))

        print(f"[Analyzer] 有效历元: {len(self.df)}")
        return self.df

    # ================================================================
    #  从外部 DataFrame 加载 (统一接口)
    # ================================================================
    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """从外部已有的 DataFrame 加载分析数据"""
        self.df = df
        if 'err_e' in df.columns:
            self.enu_errors = df[['err_e', 'err_n', 'err_u']].values
        else:
            self.enu_errors = np.empty((0, 3))

    # ================================================================
    #  精度统计报告
    # ================================================================
    def generate_report(self) -> Dict[str, float]:
        """
        生成完整精度评估报告

        Returns:
            {'rms_e', 'rms_n', 'rms_u', 'rms_h', 'rms_3d',
             'mean_e', 'mean_n', 'mean_u',
             'max_e', 'max_n', 'max_u', 'max_3d',
             'cep_68', 'cep_95',
             'mean_n_sats', 'mean_pdop', ...}
        """
        if self.df is None or len(self.df) == 0:
            return {}

        df = self.df
        report = {}

        # ENU RMS
        report['rms_e'] = np.sqrt(np.mean(df['err_e'] ** 2))
        report['rms_n'] = np.sqrt(np.mean(df['err_n'] ** 2))
        report['rms_u'] = np.sqrt(np.mean(df['err_u'] ** 2))
        report['rms_h'] = np.sqrt(np.mean(df['err_h'] ** 2))
        report['rms_3d'] = np.sqrt(np.mean(df['err_3d'] ** 2))

        # 均值误差
        report['mean_e'] = np.mean(df['err_e'])
        report['mean_n'] = np.mean(df['err_n'])
        report['mean_u'] = np.mean(df['err_u'])
        report['mean_3d'] = np.mean(df['err_3d'])

        # 最大误差
        report['max_e'] = np.max(np.abs(df['err_e']))
        report['max_n'] = np.max(np.abs(df['err_n']))
        report['max_u'] = np.max(np.abs(df['err_u']))
        report['max_3d'] = np.max(df['err_3d'])

        # CEP (Circular Error Probable)
        r_horiz = df['err_h'].values
        report['cep_68'] = np.percentile(r_horiz, 68)
        report['cep_95'] = np.percentile(r_horiz, 95)

        # 卫星与 DOP 统计
        report['mean_n_sats'] = np.mean(df['n_sats'])
        report['mean_pdop'] = np.mean(df['pdop'])
        report['mean_gdop'] = np.mean(df['gdop'])
        report['n_epochs'] = len(df)

        # 打印报告
        print(f"\n{'=' * 60}")
        print(f"  北斗 SPP 定位精度评估报告 — {self.scene_label}")
        print(f"{'=' * 60}")
        print(f"  有效历元数:     {report['n_epochs']}")
        print(f"  平均卫星数:     {report['mean_n_sats']:.1f}")
        print(f"  平均 PDOP:      {report['mean_pdop']:.2f}")
        print(f"  平均 GDOP:      {report['mean_gdop']:.2f}")
        print(f"{'─' * 60}")
        print(f"  RMS   E: {report['rms_e']:8.3f} m   "
              f"N: {report['rms_n']:8.3f} m   "
              f"U: {report['rms_u']:8.3f} m")
        print(f"  RMS   水平: {report['rms_h']:8.3f} m   "
              f"3D: {report['rms_3d']:8.3f} m")
        print(f"  均值  E: {report['mean_e']:8.3f} m   "
              f"N: {report['mean_n']:8.3f} m   "
              f"U: {report['mean_u']:8.3f} m")
        print(f"  最大  E: {report['max_e']:8.3f} m   "
              f"N: {report['max_n']:8.3f} m   "
              f"U: {report['max_u']:8.3f} m")
        print(f"  最大 3D: {report['max_3d']:8.3f} m")
        print(f"  68% CEP: {report['cep_68']:8.3f} m")
        print(f"  95% CEP: {report['cep_95']:8.3f} m")
        print(f"{'=' * 60}\n")

        return report

    # ================================================================
    #  一键绘制全部图表
    # ================================================================
    def plot_all(self, prefix: str = '') -> List[str]:
        """
        绘制满分图表全家桶

        Returns:
            生成的图片文件路径列表
        """
        if self.df is None or len(self.df) == 0:
            print("[Analyzer] 无数据可绘制")
            return []

        paths = []
        tag = f"{prefix}_{self.scene_label}" if prefix else self.scene_label

        paths.append(self.plot_enu_timeseries(tag))
        paths.append(self.plot_horizontal_scatter(tag))
        paths.append(self.plot_dop_timeseries(tag))
        paths.append(self.plot_satellite_count(tag))
        paths.append(self.plot_3d_error_timeseries(tag))

        return [p for p in paths if p]

    # ================================================================
    #  图表1: ENU 三分量误差时序图
    # ================================================================
    def plot_enu_timeseries(self, tag: str = '') -> str:
        """绘制 E / N / U 误差随时间变化曲线"""
        df = self.df
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        fig.suptitle(f'ENU 误差时序图 — {tag}', fontsize=14, fontweight='bold')

        components = [('err_e', 'East', '#d62728'),
                      ('err_n', 'North', '#2ca02c'),
                      ('err_u', 'Up', '#1f77b4')]

        for ax, (col, label, color) in zip(axes, components):
            data = df[col].values
            rms = np.sqrt(np.mean(data ** 2))
            mean_val = np.mean(data)

            ax.plot(df['epoch'], data, color=color, linewidth=0.8, alpha=0.8)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.6, alpha=0.4)
            ax.axhline(mean_val, color=color, linestyle=':', linewidth=1.2, alpha=0.6)
            ax.fill_between(df['epoch'], -rms, rms, color=color, alpha=0.08)

            ax.set_ylabel(f'{label} (m)', fontsize=11)
            ax.text(0.02, 0.92,
                    f'RMS={rms:.3f}m  均值={mean_val:.3f}m  最大={np.max(np.abs(data)):.3f}m',
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('时间 (UTC)', fontsize=11)
        fig.autofmt_xdate(rotation=30)
        plt.tight_layout()

        path = os.path.join(self.output_dir, f'{tag}_enu_timeseries.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Analyzer] ENU 时序图: {path}")
        return path

    # ================================================================
    #  图表2: 水平误差散点图 + CEP 圆
    # ================================================================
    def plot_horizontal_scatter(self, tag: str = '') -> str:
        """绘制 E-N 水平误差散点 + 68%/95% CEP 概率圆"""
        df = self.df
        e = df['err_e'].values
        n = df['err_n'].values
        r = np.sqrt(e ** 2 + n ** 2)

        cep_68 = np.percentile(r, 68)
        cep_95 = np.percentile(r, 95)
        rms_h = np.sqrt(np.mean(r ** 2))

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_title(f'水平误差散点图 — {tag}', fontsize=14, fontweight='bold')

        # 散点着色: 按 3D 误差深浅
        sc = ax.scatter(e, n, c=df['err_3d'].values, s=8, alpha=0.6,
                        cmap='RdYlGn_r', edgecolors='none')
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8, label='3D 误差 (m)')

        # 68% CEP 圆
        circle_68 = Circle((0, 0), cep_68, fill=False, color='#ff7f0e',
                           linewidth=2, linestyle='--',
                           label=f'68% CEP = {cep_68:.2f} m')
        ax.add_patch(circle_68)

        # 95% CEP 圆
        circle_95 = Circle((0, 0), cep_95, fill=False, color='#d62728',
                           linewidth=2, linestyle='-',
                           label=f'95% CEP = {cep_95:.2f} m')
        ax.add_patch(circle_95)

        # 均值点
        ax.plot(np.mean(e), np.mean(n), 'k+', markersize=15, markeredgewidth=2,
                label=f'均值 ({np.mean(e):.2f}, {np.mean(n):.2f})')

        # 十字准线
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)

        ax.set_xlabel('East (m)', fontsize=12)
        ax.set_ylabel('North (m)', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # RMS 标注
        ax.text(0.98, 0.02, f'RMS 水平 = {rms_h:.3f} m\n历元数 = {len(df)}',
                transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        path = os.path.join(self.output_dir, f'{tag}_horizontal_scatter.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Analyzer] 水平散点图: {path}")
        return path

    # ================================================================
    #  图表3: DOP 时序图
    # ================================================================
    def plot_dop_timeseries(self, tag: str = '') -> str:
        """绘制 GDOP/PDOP/HDOP/VDOP 随时间变化"""
        df = self.df
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax.set_title(f'DOP 值时序图 — {tag}', fontsize=14, fontweight='bold')

        dop_items = [
            ('gdop', 'GDOP', '#d62728', 1.5),
            ('pdop', 'PDOP', '#ff7f0e', 1.2),
            ('hdop', 'HDOP', '#2ca02c', 1.0),
            ('vdop', 'VDOP', '#1f77b4', 1.0),
        ]

        for col, label, color, lw in dop_items:
            if col in df.columns:
                data = df[col].values
                # 限幅显示 (DOP > 20 视为异常)
                data_clipped = np.minimum(data, 20.0)
                ax.plot(df['epoch'], data_clipped, label=f'{label} (均值={np.mean(data):.2f})',
                        color=color, linewidth=lw, alpha=0.8)

        ax.set_xlabel('时间 (UTC)', fontsize=11)
        ax.set_ylabel('DOP 值', fontsize=11)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.autofmt_xdate(rotation=30)
        plt.tight_layout()

        path = os.path.join(self.output_dir, f'{tag}_dop_timeseries.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Analyzer] DOP 时序图: {path}")
        return path

    # ================================================================
    #  图表4: 卫星可用数时序图
    # ================================================================
    def plot_satellite_count(self, tag: str = '') -> str:
        """绘制可用卫星数随时间变化"""
        df = self.df
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.set_title(f'可用卫星数时序图 — {tag}', fontsize=14, fontweight='bold')

        ax.fill_between(df['epoch'], df['n_sats'], alpha=0.3, color='#1f77b4')
        ax.plot(df['epoch'], df['n_sats'], color='#1f77b4', linewidth=1.0)

        mean_n = np.mean(df['n_sats'])
        ax.axhline(mean_n, color='red', linestyle='--', linewidth=1.2, alpha=0.7,
                   label=f'均值 = {mean_n:.1f}')
        ax.axhline(4, color='orange', linestyle=':', linewidth=1.0, alpha=0.6,
                   label='最低要求 = 4')

        ax.set_xlabel('时间 (UTC)', fontsize=11)
        ax.set_ylabel('卫星数', fontsize=11)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.autofmt_xdate(rotation=30)
        plt.tight_layout()

        path = os.path.join(self.output_dir, f'{tag}_sat_count.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Analyzer] 卫星数图: {path}")
        return path

    # ================================================================
    #  图表5: 3D 定位误差时序图
    # ================================================================
    def plot_3d_error_timeseries(self, tag: str = '') -> str:
        """绘制 3D 定位误差随时间变化"""
        df = self.df
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax.set_title(f'3D 定位误差时序图 — {tag}', fontsize=14, fontweight='bold')

        err_3d = df['err_3d'].values
        rms = np.sqrt(np.mean(err_3d ** 2))

        ax.plot(df['epoch'], err_3d, color='#d62728', linewidth=0.8, alpha=0.7,
                label='3D 误差')
        ax.axhline(rms, color='blue', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'RMS = {rms:.3f} m')
        ax.fill_between(df['epoch'], 0, err_3d, alpha=0.15, color='#d62728')

        ax.set_xlabel('时间 (UTC)', fontsize=11)
        ax.set_ylabel('3D 误差 (m)', fontsize=11)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.autofmt_xdate(rotation=30)
        plt.tight_layout()

        path = os.path.join(self.output_dir, f'{tag}_3d_error.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Analyzer] 3D 误差图: {path}")
        return path

    # ================================================================
    #  AI 补偿前后对比图 (对接 ai_compensator)
    # ================================================================
    def plot_compensation_comparison(
        self,
        solutions_raw: List,
        solutions_comp: List,
        tag: str = 'AI补偿对比',
    ) -> List[str]:
        """
        绘制 AI 补偿前后的 ENU 精度对比

        Returns:
            图片路径列表
        """
        paths = []

        # 提取有效历元 ENU
        enu_raw, enu_comp, epochs = [], [], []
        for sr, sc in zip(solutions_raw, solutions_comp):
            if not sr.valid:
                continue
            enu_raw.append(R_ENU @ (sr.pos_ecef - GT_ECEF))
            enu_comp.append(R_ENU @ (sc.pos_ecef - GT_ECEF))
            epochs.append(sr.epoch)

        if not enu_raw:
            return paths

        enu_raw = np.array(enu_raw)
        enu_comp = np.array(enu_comp)

        # --- 图A: ENU 分量对比 ---
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f'AI 补偿前后 ENU 误差对比 — {tag}', fontsize=14, fontweight='bold')

        labels = ['East (m)', 'North (m)', 'Up (m)']
        for i, ax in enumerate(axes):
            rms_r = np.sqrt(np.mean(enu_raw[:, i] ** 2))
            rms_c = np.sqrt(np.mean(enu_comp[:, i] ** 2))

            ax.plot(epochs, enu_raw[:, i], color='#d62728', linewidth=0.8,
                    alpha=0.6, label=f'补偿前 RMS={rms_r:.3f}m')
            ax.plot(epochs, enu_comp[:, i], color='#1f77b4', linewidth=0.8,
                    alpha=0.8, label=f'补偿后 RMS={rms_c:.3f}m')
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.4)
            ax.set_ylabel(labels[i], fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('时间 (UTC)', fontsize=11)
        fig.autofmt_xdate(rotation=30)
        plt.tight_layout()
        p = os.path.join(self.output_dir, f'{tag}_comp_enu.png')
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        # --- 图B: 水平散点对比 + CEP ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'水平误差散点对比 — {tag}', fontsize=14, fontweight='bold')

        for idx, (enu, title, color, ccolor) in enumerate([
            (enu_raw, '补偿前', '#d62728', 'red'),
            (enu_comp, '补偿后', '#1f77b4', 'blue'),
        ]):
            ax = axes[idx]
            e, n = enu[:, 0], enu[:, 1]
            r = np.sqrt(e ** 2 + n ** 2)
            cep95 = np.percentile(r, 95)
            cep68 = np.percentile(r, 68)

            ax.scatter(e, n, c=color, s=8, alpha=0.5)
            ax.add_patch(Circle((0, 0), cep95, fill=False, color=ccolor,
                                linewidth=2, linestyle='-',
                                label=f'95% CEP={cep95:.2f}m'))
            ax.add_patch(Circle((0, 0), cep68, fill=False, color=ccolor,
                                linewidth=1.5, linestyle='--',
                                label=f'68% CEP={cep68:.2f}m'))
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.axvline(0, color='gray', linewidth=0.5)
            ax.set_xlabel('East (m)')
            ax.set_ylabel('North (m)')
            ax.set_title(title, fontsize=12)
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')

        plt.tight_layout()
        p = os.path.join(self.output_dir, f'{tag}_comp_scatter.png')
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
        # --- 统计汇总 ---
        rms3d_raw = np.sqrt(np.mean(np.sum(enu_raw ** 2, axis=1)))
        rms3d_comp = np.sqrt(np.mean(np.sum(enu_comp ** 2, axis=1)))
        print(f"\n[Analyzer] {tag} 补偿效果:")
        print(f"  补偿前 3D RMS: {rms3d_raw:.3f} m")
        print(f"  补偿后 3D RMS: {rms3d_comp:.3f} m")
        print(f"  精度提升: {(1 - rms3d_comp / max(rms3d_raw, 1e-9)) * 100:.1f}%\n")

        return paths

    # ================================================================
    #  将分析结果导出为 CSV (供 AI 训练或报告附录)
    # ================================================================
    def export_csv(self, filename: Optional[str] = None) -> str:
        """将分析 DataFrame 导出为 CSV 文件"""
        if self.df is None or len(self.df) == 0:
            print("[Analyzer] 无数据可导出")
            return ''

        if filename is None:
            filename = f"{self.scene_label}_analysis.csv"

        path = os.path.join(self.output_dir, filename)
        self.df.to_csv(path, index=False, float_format='%.6f')
        print(f"[Analyzer] 分析数据已导出: {path}")
        return path