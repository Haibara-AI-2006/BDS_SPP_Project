"""
ai_compensator.py — 12+ 维特征随机森林 AI 误差预测与补偿引擎
=============================================================
核心职责:
  1. 从 SPP 解算结果中提取 12+ 维特征矩阵 (X)
  2. 计算 ECEF 误差标签 (Y) = SPP 位置 - 真值 ECEF
  3. 训练随机森林回归模型 (scikit-learn)
  4. 集成到定位软件中实现误差实时预测与自动补偿
  5. 绘制 ENU 补偿前后对比图 + 95% CEP 圆

特征向量 (12+ 维):
  [n_sats, GDOP, PDOP, HDOP, VDOP,
   residual_rms, max_residual, geo_ratio,
   n_low_elev, elev_std, snr_std,
   time_sin, time_cos]

真值锚定:
  北邮沙河校区 ECEF = blh_to_ecef(40.1575°N, 116.2885°E, 35m)
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无头渲染
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime
from typing import List, Optional, Tuple, Dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from rinex_parser import WGS84_A, WGS84_E2

# ============================================================
#  真值锚定 — 北邮沙河校区 WGS84 坐标
# ============================================================
GT_LAT_DEG = 40.1575       # 纬度 (度)
GT_LON_DEG = 116.2885      # 经度 (度)
GT_HEIGHT = 35.0            # 椭球高 (m)

GT_LAT_RAD = math.radians(GT_LAT_DEG)
GT_LON_RAD = math.radians(GT_LON_DEG)


def blh_to_ecef(lat_rad: float, lon_rad: float, h: float) -> np.ndarray:
    """
    WGS84 大地坐标 (BLH) → ECEF 直角坐标

    Parameters:
        lat_rad : 纬度 (rad)
        lon_rad : 经度 (rad)
        h       : 椭球高 (m)

    Returns:
        np.ndarray (3,) — [X, Y, Z] ECEF 坐标 (m)
    """
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    # 卯酉圈曲率半径
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)

    X = (N + h) * cos_lat * cos_lon
    Y = (N + h) * cos_lat * sin_lon
    Z = (N * (1.0 - WGS84_E2) + h) * sin_lat

    return np.array([X, Y, Z], dtype=np.float64)


def ecef_to_enu_matrix(lat_rad: float, lon_rad: float) -> np.ndarray:
    """
    构建 ECEF → ENU 旋转矩阵 (3×3)

    Parameters:
        lat_rad : 参考点纬度 (rad)
        lon_rad : 参考点经度 (rad)

    Returns:
        (3, 3) 旋转矩阵 R, 使得 enu = R @ dx_ecef
    """
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)

    R = np.array([
        [-sin_lon,              cos_lon,             0.0    ],
        [-sin_lat * cos_lon,   -sin_lat * sin_lon,   cos_lat],
        [ cos_lat * cos_lon,    cos_lat * sin_lon,   sin_lat],
    ], dtype=np.float64)
    return R


# 全局真值 ECEF 坐标
GT_ECEF = blh_to_ecef(GT_LAT_RAD, GT_LON_RAD, GT_HEIGHT)

# 全局 ENU 旋转矩阵 (以真值为参考) - 这是关键！
R_ENU = ecef_to_enu_matrix(GT_LAT_RAD, GT_LON_RAD)

# 特征名称 (12+ 维)
FEATURE_NAMES = [
    'n_sats',           # 可见卫星数
    'gdop',             # 几何精度因子
    'pdop',
    'hdop',
    'vdop',
    'residual_rms',     # 验后残差 RMS
    'max_residual',     # 最大残差
    'geo_ratio',        # GEO 卫星占比
    'n_low_elev',       # 低仰角 (<20°) 卫星数
    'elev_std',         # 高度角标准差
    'snr_std',          # SNR 标准差
    'time_sin',         # 时间周期特征 (sin)
    'time_cos',         # 时间周期特征 (cos)
]


# ============================================================
#  AI 补偿器
# ============================================================
class AiCompensator:
    """
    随机森林误差预测与补偿引擎

    工作流程:
      1. 从 SPP 解算结果提取 12+ 维特征
      2. 计算 ECEF 误差标签 (dX, dY, dZ)
      3. 训练 3 个独立的随机森林回归器 (X/Y/Z)
      4. 实时预测误差并补偿 SPP 位置
      5. 绘制 ENU 补偿前后对比图
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Parameters:
            model_path : 预训练模型路径 (pkl), None 则需训练
        """
        self.model_x: Optional[RandomForestRegressor] = None
        self.model_y: Optional[RandomForestRegressor] = None
        self.model_z: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    # ================================================================
    #  特征提取 (12+ 维)
    # ================================================================
    @staticmethod
    def extract_features(solutions: List) -> pd.DataFrame:
        """
        从 SPP 解算结果列表提取特征矩阵

        Parameters:
            solutions : List[EpochSolution]

        Returns:
            DataFrame (n_epochs, 12+)
        """
        features = []

        for sol in solutions:
            if not sol.valid:
                continue

            # 基础特征
            n_sats = sol.n_used
            gdop = sol.dop.gdop
            pdop = sol.dop.pdop
            hdop = sol.dop.hdop
            vdop = sol.dop.vdop

            # 残差统计
            if len(sol.residuals) > 0:
                residual_rms = np.sqrt(np.mean(sol.residuals ** 2))
                max_residual = np.max(np.abs(sol.residuals))
            else:
                residual_rms = 0.0
                max_residual = 0.0

            # GEO 卫星占比 (C01-C05, C59-C63)
            n_geo = 0
            for prn in sol.prn_list:
                prn_num = int(prn[1:])
                if (1 <= prn_num <= 5) or (59 <= prn_num <= 63):
                    n_geo += 1
            geo_ratio = n_geo / n_sats if n_sats > 0 else 0.0

            # 低仰角卫星数 (<20°)
            n_low_elev = np.sum(sol.elevations < math.radians(20.0))

            # 高度角标准差
            elev_std = np.std(sol.elevations) if len(sol.elevations) > 1 else 0.0

            # SNR 标准差
            snr_std = np.std(sol.snr_values) if len(sol.snr_values) > 1 else 0.0

            # 时间周期特征 (24h 周期)
            hour_of_day = sol.epoch.hour + sol.epoch.minute / 60.0
            time_angle = 2.0 * math.pi * hour_of_day / 24.0
            time_sin = math.sin(time_angle)
            time_cos = math.cos(time_angle)

            features.append([
                n_sats, gdop, pdop, hdop, vdop,
                residual_rms, max_residual, geo_ratio,
                n_low_elev, elev_std, snr_std,
                time_sin, time_cos,
            ])

        df = pd.DataFrame(features, columns=FEATURE_NAMES)
        return df

    # ================================================================
    #  标签提取 (ECEF 误差)
    # ================================================================
    @staticmethod
    def extract_labels(solutions: List) -> pd.DataFrame:
        """
        计算 ECEF 误差标签 (dX, dY, dZ)

        Y = SPP_ECEF - GT_ECEF

        Parameters:
            solutions : List[EpochSolution]

        Returns:
            DataFrame (n_epochs, 3) — ['dX', 'dY', 'dZ']
        """
        labels = []

        for sol in solutions:
            if not sol.valid:
                continue

            dx = sol.pos_ecef[0] - GT_ECEF[0]
            dy = sol.pos_ecef[1] - GT_ECEF[1]
            dz = sol.pos_ecef[2] - GT_ECEF[2]

            labels.append([dx, dy, dz])

        df = pd.DataFrame(labels, columns=['dX', 'dY', 'dZ'])
        return df

    # ================================================================
    #  模型训练
    # ================================================================
    def train(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        test_size: float = 0.3,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 20,
        min_samples_split: int = 5,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        训练 3 个独立的随机森林回归器 (X/Y/Z)

        Parameters:
            X              : 特征矩阵 (n, 12+)
            Y              : 标签矩阵 (n, 3) — [dX, dY, dZ]
            test_size      : 测试集比例
            random_state   : 随机种子
            n_estimators   : 随机森林树数量
            max_depth      : 树最大深度
            min_samples_split : 节点最小分裂样本数
            verbose        : 是否打印训练信息

        Returns:
            评估指标字典 {'rmse_x', 'rmse_y', 'rmse_z', 'rmse_3d', ...}
        """
        if len(X) != len(Y):
            raise ValueError("特征与标签数量不匹配!")

        if verbose:
            print(f"\n[AI Compensator] 训练数据: {len(X)} 个历元")
            print(f"[AI Compensator] 特征维度: {X.shape[1]}")
            print(f"[AI Compensator] 训练/测试划分: {1-test_size:.0%} / {test_size:.0%}")

        # 数据划分 (时序数据不打乱, 按顺序切分)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        Y_train, Y_test = Y.iloc[:split_idx], Y.iloc[split_idx:]

        # 特征标准化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 训练 3 个独立模型
        metrics = {}

        for axis_idx, axis_name in enumerate(['X', 'Y', 'Z']):
            if verbose:
                print(f"\n[AI Compensator] 训练 {axis_name} 轴模型...")

            y_train = Y_train.iloc[:, axis_idx].values
            y_test = Y_test.iloc[:, axis_idx].values

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state,
                n_jobs=-1,
            )

            model.fit(X_train_scaled, y_train)

            # 预测
            y_pred = model.predict(X_test_scaled)

            # 评估
            rmse = math.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            metrics[f'rmse_{axis_name.lower()}'] = rmse
            metrics[f'mae_{axis_name.lower()}'] = mae

            if verbose:
                print(f"  RMSE: {rmse:.4f} m")
                print(f"  MAE:  {mae:.4f} m")

            # 保存模型
            if axis_name == 'X':
                self.model_x = model
            elif axis_name == 'Y':
                self.model_y = model
            else:
                self.model_z = model

        # 3D RMSE
        y_pred_all = np.column_stack([
            self.model_x.predict(X_test_scaled),
            self.model_y.predict(X_test_scaled),
            self.model_z.predict(X_test_scaled),
        ])
        y_test_all = Y_test.values

        errors_3d = np.linalg.norm(y_test_all - y_pred_all, axis=1)
        rmse_3d = np.sqrt(np.mean(errors_3d ** 2))
        metrics['rmse_3d'] = rmse_3d

        if verbose:
            print(f"\n[AI Compensator] 3D RMSE: {rmse_3d:.4f} m")

        return metrics

    # ================================================================
    #  误差预测
    # ================================================================
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测 ECEF 误差 (dX, dY, dZ)

        Parameters:
            X : 特征矩阵 (n, 12+)

        Returns:
            (n, 3) 误差预测值
        """
        if self.model_x is None or self.scaler is None:
            raise RuntimeError("模型未训练或加载!")

        X_scaled = self.scaler.transform(X)

        dx = self.model_x.predict(X_scaled)
        dy = self.model_y.predict(X_scaled)
        dz = self.model_z.predict(X_scaled)

        return np.column_stack([dx, dy, dz])

    # ================================================================
    #  补偿 SPP 位置
    # ================================================================
    def compensate_solutions(self, solutions: List) -> List:
        """
        对 SPP 解算结果进行 AI 误差补偿

        Parameters:
            solutions : List[EpochSolution] (原始 SPP 结果)

        Returns:
            List[EpochSolution] (补偿后的结果, 深拷贝)
        """
        import copy

        # 提取特征
        X = self.extract_features(solutions)

        if len(X) == 0:
            return solutions

        # 预测误差
        errors_pred = self.predict(X)  # (n, 3)

        # 补偿位置
        compensated = []
        valid_idx = 0

        for sol in solutions:
            sol_new = copy.deepcopy(sol)

            if sol.valid:
                # 补偿 ECEF 坐标: pos_corrected = pos_raw - error_pred
                sol_new.pos_ecef = sol.pos_ecef - errors_pred[valid_idx]

                # 重新计算 BLH
                from rinex_parser import ecef_to_blh
                lat_r, lon_r, h = ecef_to_blh(
                    sol_new.pos_ecef[0],
                    sol_new.pos_ecef[1],
                    sol_new.pos_ecef[2],
                )
                sol_new.lat_deg = math.degrees(lat_r)
                sol_new.lon_deg = math.degrees(lon_r)
                sol_new.height = h

                valid_idx += 1

            compensated.append(sol_new)

        return compensated

    # ================================================================
    #  模型保存 & 加载
    # ================================================================
    def save_model(self, filepath: str) -> None:
        """保存训练好的模型到 pickle 文件"""
        model_dict = {
            'model_x': self.model_x,
            'model_y': self.model_y,
            'model_z': self.model_z,
            'scaler': self.scaler,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
        print(f"[AI Compensator] 模型已保存: {filepath}")

    def load_model(self, filepath: str) -> None:
        """从 pickle 文件加载模型"""
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)
        self.model_x = model_dict['model_x']
        self.model_y = model_dict['model_y']
        self.model_z = model_dict['model_z']
        self.scaler = model_dict['scaler']
        print(f"[AI Compensator] 模型已加载: {filepath}")

    # ================================================================
    #  可视化: ENU 补偿前后对比
    # ================================================================
    @staticmethod
    def plot_enu_comparison(
        solutions_raw: List,
        solutions_compensated: List,
        output_dir: str = "data",
        scene_name: str = "comparison",
    ) -> None:
        """
        绘制 ENU 补偿前后对比图

        包含:
          1. ENU 时序误差曲线 (E/N/U 三子图)
          2. 水平 (E-N) 散点对比 + 95% CEP 圆

        Parameters:
            solutions_raw         : 原始 SPP 解算结果
            solutions_compensated : AI 补偿后结果
            output_dir            : 输出目录
            scene_name            : 场景名称 (用于文件命名)
        """
        os.makedirs(output_dir, exist_ok=True)

        # 提取有效历元的 ECEF 误差
        errors_raw = []
        errors_comp = []
        epochs = []

        for sol_raw, sol_comp in zip(solutions_raw, solutions_compensated):
            if not sol_raw.valid:
                continue

            err_raw = sol_raw.pos_ecef - GT_ECEF
            err_comp = sol_comp.pos_ecef - GT_ECEF

            errors_raw.append(err_raw)
            errors_comp.append(err_comp)
            epochs.append(sol_raw.epoch)

        if len(errors_raw) == 0:
            print("[AI Compensator] 无有效历元，跳过可视化")
            return

        errors_raw = np.array(errors_raw)    # (n, 3)
        errors_comp = np.array(errors_comp)  # (n, 3)

        # ECEF → ENU
        enu_raw = (R_ENU @ errors_raw.T).T    # (n, 3)
        enu_comp = (R_ENU @ errors_comp.T).T  # (n, 3)

        # ================================================================
        #  图1: ENU 时序误差曲线
        # ================================================================
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle(f'ENU 误差时序对比 — {scene_name}', fontsize=14, fontweight='bold')

        labels_enu = ['East (m)', 'North (m)', 'Up (m)']
        colors_raw = ['#d62728', '#ff7f0e', '#2ca02c']
        colors_comp = ['#1f77b4', '#9467bd', '#8c564b']

        for i, ax in enumerate(axes):
            ax.plot(epochs, enu_raw[:, i], label='补偿前', color=colors_raw[i],
                    linewidth=1.2, alpha=0.7)
            ax.plot(epochs, enu_comp[:, i], label='补偿后', color=colors_comp[i],
                    linewidth=1.2, alpha=0.9)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_ylabel(labels_enu[i], fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)

            # 统计信息
            rms_raw = np.sqrt(np.mean(enu_raw[:, i] ** 2))
            rms_comp = np.sqrt(np.mean(enu_comp[:, i] ** 2))
            ax.text(0.02, 0.95, f'RMS: {rms_raw:.2f}m → {rms_comp:.2f}m',
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        axes[-1].set_xlabel('时间', fontsize=11)
        plt.tight_layout()
        path_ts = os.path.join(output_dir, f'{scene_name}_enu_timeseries.png')
        plt.savefig(path_ts, dpi=150)
        plt.close()
        print(f"[AI Compensator] ENU 时序图已保存: {path_ts}")

        # ================================================================
        #  图2: 水平 (E-N) 散点对比 + 95% CEP 圆
        # ================================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'水平误差散点对比 (E-N) — {scene_name}', fontsize=14, fontweight='bold')

        # 计算 95% CEP (Circular Error Probable)
        def compute_cep95(e, n):
            """计算 95% 圆概率误差半径"""
            r = np.sqrt(e ** 2 + n ** 2)
            return np.percentile(r, 95)

        cep95_raw = compute_cep95(enu_raw[:, 0], enu_raw[:, 1])
        cep95_comp = compute_cep95(enu_comp[:, 0], enu_comp[:, 1])

        # 子图1: 补偿前
        ax = axes[0]
        ax.scatter(enu_raw[:, 0], enu_raw[:, 1], c='#d62728', s=10, alpha=0.6, label='补偿前')
        circle_raw = Circle((0, 0), cep95_raw, color='red', fill=False, linewidth=2,
                            linestyle='--', label=f'95% CEP: {cep95_raw:.2f}m')
        ax.add_patch(circle_raw)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('East (m)', fontsize=11)
        ax.set_ylabel('North (m)', fontsize=11)
        ax.set_title('补偿前', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # 子图2: 补偿后
        ax = axes[1]
        ax.scatter(enu_comp[:, 0], enu_comp[:, 1], c='#1f77b4', s=10, alpha=0.6, label='补偿后')
        circle_comp = Circle((0, 0), cep95_comp, color='blue', fill=False, linewidth=2,
                             linestyle='--', label=f'95% CEP: {cep95_comp:.2f}m')
        ax.add_patch(circle_comp)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('East (m)', fontsize=11)
        ax.set_ylabel('North (m)', fontsize=11)
        ax.set_title('补偿后', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        plt.tight_layout()
        path_scatter = os.path.join(output_dir, f'{scene_name}_enu_scatter.png')
        plt.savefig(path_scatter, dpi=150)
        plt.close()
        print(f"[AI Compensator] 水平散点图已保存: {path_scatter}")

        # ================================================================
        #  打印统计摘要
        # ================================================================
        print(f"\n[AI Compensator] {scene_name} 补偿效果统计:")
        print(f"  补偿前 95% CEP: {cep95_raw:.2f} m")
        print(f"  补偿后 95% CEP: {cep95_comp:.2f} m")
        print(f"  水平精度提升: {(1 - cep95_comp / cep95_raw) * 100:.1f}%")

        rms_3d_raw = np.sqrt(np.mean(np.sum(enu_raw ** 2, axis=1)))
        rms_3d_comp = np.sqrt(np.mean(np.sum(enu_comp ** 2, axis=1)))
        print(f"  补偿前 3D RMS: {rms_3d_raw:.2f} m")
        print(f"  补偿后 3D RMS: {rms_3d_comp:.2f} m")
        print(f"  3D 精度提升: {(1 - rms_3d_comp / rms_3d_raw) * 100:.1f}%\n")


# ============================================================
#  独立运行入口 (示例)
# ============================================================
if __name__ == "__main__":
    import sys

    # 示例: 从 CSV 加载模拟数据并训练模型
    csv_path = os.path.join("data", "scene1_open_sky.csv")
    if not os.path.exists(csv_path):
        print(f"[Error] 数据文件不存在: {csv_path}")
        print("请先运行 data_simulator.py 生成数据")
        sys.exit(1)

    # 这里需要先运行 SPP 解算得到 solutions
    # 本示例仅展示 AI 模块的独立功能
    print("[AI Compensator] 独立测试模式")
    print("完整流程请参考 main_gui.py 或 trajectory_analyzer.py")