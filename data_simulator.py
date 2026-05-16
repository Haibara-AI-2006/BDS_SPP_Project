"""
data_simulator.py — 物理级北斗伪距观测数据生成引擎
===================================================
核心职责:
  1. 读取真实 .nav 星历文件，计算指定时间段内 BDS 卫星的真实轨道位置
  2. 基于北邮沙河真值锚点生成物理级伪距观测值
  3. 注入官方噪声基准 (SISRE + 电离层 + 对流层 + 接收机钟差 + 白噪声)
  4. 低仰角卫星叠加 AR(1) 时间相关彩色多径噪声
  5. 生成标准 RINEX 3.x OBS 文件 & 多场景 CSV 数据集

噪声模型:
  P = ρ_geo + N(0,0.5)_SISRE + N(10,3)_Iono + N(4,1.5)_Tropo
      + (60 + N(0,12))_RClk + N(0,0.3)_obs
  + [低仰角 AR(1) 彩色多径噪声]

真值锚定:
  北京邮电大学沙河校区 WGS84:
    纬度 40.1575° N, 经度 116.2885° E, 高程 35 m
"""

import os
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from rinex_parser import (
    RinexNavParser, BdsEphemeris,
    SPEED_OF_LIGHT, GM_BDS, OMEGA_E_BDS, PI,
    WGS84_A, WGS84_E2, WGS84_F,
    gps_week_seconds_from_datetime, ecef_to_blh, compute_elevation_azimuth,
)
from sat_pos_calculator import SatPosCalculator

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
print(f"[Simulator] 真值 ECEF = {GT_ECEF}")

# 全局 ENU 旋转矩阵 (以真值为参考)
GT_R_ENU = ecef_to_enu_matrix(GT_LAT_RAD, GT_LON_RAD)


# ============================================================
#  AR(1) 彩色多径噪声生成器
# ============================================================
class ColoredMultipathGenerator:
    """
    基于一阶自回归 AR(1) 过程的时间相关彩色多径噪声

    模型:
        ε_t = α · ε_{t-1} + w_t
        w_t ~ N(0, σ_mp)
        σ_mp = A / sin(E)   (低仰角放大)

    其中 α ∈ (0, 1) 控制时间相关性 (越接近 1 越强相关)
    A 为多径噪声基准幅值 (场景相关)
    """

    def __init__(self, alpha: float = 0.85, base_amplitude: float = 2.0):
        """
        Parameters:
            alpha          : AR(1) 衰减系数, 0 < α < 1
            base_amplitude : 基准多径幅值 A (m)
        """
        self.alpha = alpha
        self.base_amp = base_amplitude
        # 每颗卫星维护独立的状态
        self._states: Dict[str, float] = {}

    def reset(self):
        """重置所有卫星的多径状态"""
        self._states.clear()

    def generate(self, prn: str, elev_rad: float) -> float:
        """
        为指定卫星生成当前历元的彩色多径噪声值

        Parameters:
            prn      : 卫星 PRN 标识
            elev_rad : 卫星高度角 (rad)

        Returns:
            多径噪声值 (m)
        """
        sin_e = max(math.sin(elev_rad), 0.05)
        # 低仰角放大: σ_mp = A / sin(E)
        sigma_mp = self.base_amp / sin_e

        # 驱动白噪声
        w_t = random.gauss(0, sigma_mp * math.sqrt(1 - self.alpha ** 2))

        # AR(1) 递推
        prev = self._states.get(prn, 0.0)
        eps_t = self.alpha * prev + w_t
        self._states[prn] = eps_t

        return eps_t


# ============================================================
#  场景配置
# ============================================================
class SceneConfig:
    """模拟场景参数配置"""

    def __init__(
        self,
        name: str,
        label: str,
        elev_mask_deg: float = 10.0,
        multipath_enabled: bool = False,
        multipath_alpha: float = 0.85,
        multipath_amplitude: float = 2.0,
        multipath_elev_threshold_deg: float = 20.0,
        random_sv_dropout: float = 0.0,
        iono_scale: float = 1.0,
        tropo_scale: float = 1.0,
    ):
        self.name = name
        self.label = label
        self.elev_mask_deg = elev_mask_deg
        self.multipath_enabled = multipath_enabled
        self.multipath_alpha = multipath_alpha
        self.multipath_amplitude = multipath_amplitude
        self.multipath_elev_threshold_deg = multipath_elev_threshold_deg
        self.random_sv_dropout = random_sv_dropout
        self.iono_scale = iono_scale
        self.tropo_scale = tropo_scale


# 预定义三大场景
SCENE_OPEN_SKY = SceneConfig(
    name="scene1_open_sky",
    label="开阔天空",
    elev_mask_deg=10.0,
    multipath_enabled=False,
)

SCENE_TREE_CANOPY = SceneConfig(
    name="scene2_tree_canopy",
    label="森林林冠",
    elev_mask_deg=15.0,
    multipath_enabled=True,
    multipath_alpha=0.88,
    multipath_amplitude=3.5,
    multipath_elev_threshold_deg=30.0,
    random_sv_dropout=0.15,
    iono_scale=1.3,
    tropo_scale=1.2,
)

SCENE_URBAN_CANYON = SceneConfig(
    name="scene3_urban_canyon",
    label="城市峡谷",
    elev_mask_deg=20.0,
    multipath_enabled=True,
    multipath_alpha=0.92,
    multipath_amplitude=5.0,
    multipath_elev_threshold_deg=35.0,
    random_sv_dropout=0.25,
    iono_scale=1.5,
    tropo_scale=1.4,
)


# ============================================================
#  伪距观测数据生成器
# ============================================================
class DataSimulator:
    """
    物理级 BDS 伪距观测数据生成引擎

    工作流程:
      1. 加载 .nav 星历 → 初始化卫星位置计算器
      2. 按 30s 采样率遍历 24h 时间窗
      3. 逐历元计算所有可见 BDS 卫星的真实几何距离
      4. 注入分层误差模型 → 生成伪距观测值
      5. 输出 RINEX OBS 文件 + 场景 CSV 文件
    """

    def __init__(self, nav_path: str, output_dir: str = "data"):
        self.nav_path = nav_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 初始化解析器
        self.nav_parser = RinexNavParser()
        self.nav_parser.parse(nav_path)
        self.sat_calc = SatPosCalculator(self.nav_parser)

        # 真值 ECEF
        self.gt_ecef = GT_ECEF.copy()
        self.gt_lat_rad = GT_LAT_RAD
        self.gt_lon_rad = GT_LON_RAD
        self.gt_h = GT_HEIGHT

        # 多径噪声发生器 (按场景动态配置)
        self.multipath_gen = ColoredMultipathGenerator()

    # ================================================================
    #  主生成接口
    # ================================================================
    def generate_obs_file(
        self,
        start_time: datetime,
        duration_hours: float = 24.0,
        interval_sec: float = 30.0,
        scene: SceneConfig = SCENE_OPEN_SKY,
        obs_filename: Optional[str] = None,
        csv_filename: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        生成单场景的 RINEX OBS 文件和 CSV 数据集

        Parameters:
            start_time     : 观测起始时间
            duration_hours : 观测持续时长 (小时)
            interval_sec   : 采样间隔 (秒)
            scene          : 场景配置
            obs_filename   : 输出 OBS 文件名 (None 则自动生成)
            csv_filename   : 输出 CSV 文件名 (None 则自动生成)

        Returns:
            (obs_path, csv_path) 生成文件的完整路径
        """
        if obs_filename is None:
            date_str = start_time.strftime("%Y%m%d")
            obs_filename = f"BUPT_Shahe_{date_str}.obs"
        if csv_filename is None:
            csv_filename = f"{scene.name}.csv"

        obs_path = os.path.join(self.output_dir, obs_filename)
        csv_path = os.path.join(self.output_dir, csv_filename)

        print(f"\n[Simulator] 场景: {scene.label}")
        print(f"[Simulator] 真值 ECEF: {self.gt_ecef}")
        print(f"[Simulator] 时间范围: {start_time} → "
              f"{start_time + timedelta(hours=duration_hours)}")
        print(f"[Simulator] 采样间隔: {interval_sec}s")

        # 重置多径生成器
        self.multipath_gen = ColoredMultipathGenerator(
            alpha=scene.multipath_alpha,
            base_amplitude=scene.multipath_amplitude,
        )

        # 生成所有历元数据
        n_epochs = int(duration_hours * 3600 / interval_sec)
        epoch_records = []     # 用于 CSV 存储
        obs_blocks = []        # 用于 RINEX OBS 输出

        # 固定的接收机钟差偏移 (本次观测会话共用的基准偏移)
        rclk_base = 60.0  # 米

        for ep_idx in range(n_epochs):
            current_time = start_time + timedelta(seconds=ep_idx * interval_sec)
            _, tow = gps_week_seconds_from_datetime(current_time)

            # 本历元接收机钟差: 基准 + 随机游走
            rclk_noise = rclk_base + random.gauss(0, 12.0)

            # 遍历所有有星历的卫星
            epoch_sats = {}

            for prn, eph_list in self.nav_parser.ephemeris_pool.items():
                # 选择最佳星历
                eph = self.nav_parser.select_ephemeris(prn, current_time)
                if eph is None:
                    continue

                # 计算卫星 ECEF 坐标与钟差
                                # 使用粗略的伪距估计（接收机到卫星约 2000-4200万米）
                # --- 卫星精确位置 & 钟差 (严密闭合光行时带来的轨道误差) ---
                # 第 1 次迭代：使用默认的 0.075s 飞行时间算出粗略位置
                res_rough = self.sat_calc.compute_sat_pos_clk(prn, current_time, 0.0)
                if res_rough is None:
                    continue
                sv_pos_rough, _ = res_rough
                
                # 计算出真实场景下，测站到该卫星的物理几何距离
                pr_true = np.linalg.norm(sv_pos_rough - self.gt_ecef)
                
                # 第 2 次迭代：使用真实物理距离，重新精算极致严密的卫星坐标与钟差
                result = self.sat_calc.compute_sat_pos_clk(prn, current_time, pr_true)
                if result is None:
                    continue
                sv_pos, sv_clk_sec = result

                # 检查卫星位置有效性
                if np.linalg.norm(sv_pos) < 1e6:
                    continue

                # 计算高度角和方位角
                elev, az = compute_elevation_azimuth(self.gt_ecef, sv_pos)
                elev_deg = math.degrees(elev)

                # 高度角截止
                if elev_deg < scene.elev_mask_deg:
                    continue

                # 随机卫星丢失 (模拟遮挡)
                if random.random() < scene.random_sv_dropout:
                    continue

# === 1. 计算真实几何距离 (增加 Sagnac 地球自转修正) ===
                # 信号传播时间粗估
                tau = np.linalg.norm(sv_pos - self.gt_ecef) / SPEED_OF_LIGHT
                theta = OMEGA_E_BDS * tau
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                # 模拟地球自转，将发射时刻卫星坐标向前旋转
                R_sagnac = np.array([
                    [ cos_t, sin_t, 0.0],
                    [-sin_t, cos_t, 0.0],
                    [  0.0,   0.0,  1.0]
                ])
                sv_pos_rot = R_sagnac @ sv_pos
                rho_geo = np.linalg.norm(sv_pos_rot - self.gt_ecef)

                # === 2. 分层误差注入 (结合严密物理模型与原有噪声基准) ===
                # 1. SISRE 卫星轨道/钟差残差 (保持不变)
                err_sisre = random.gauss(0, 0.5)

                # 2. 电离层延迟 (带场景缩放): 物理基准模型 + 高斯噪声
                iono_nominal = self.sat_calc.klobuchar_bds(self.gt_lat_rad, self.gt_lon_rad, elev, az, tow)
                err_iono = (iono_nominal + random.gauss(0, 0.5)) * scene.iono_scale

                # 3. 对流层延迟 (带场景缩放): 物理基准模型 + 高斯噪声
                tropo_nominal = self.sat_calc.saastamoinen(elev, self.gt_lat_rad, self.gt_h)
                err_tropo = (tropo_nominal + random.gauss(0, 0.3)) * scene.tropo_scale

                # 4. 接收机钟差 (保持不变)
                err_rclk = rclk_noise

                # 5. 观测白噪声 (保持不变)
                err_obs = random.gauss(0, 0.3)

                # 6. 彩色多径噪声 (仅低仰角 + 场景启用时) - (保持不变)
                err_multipath = 0.0
                if scene.multipath_enabled and elev_deg < scene.multipath_elev_threshold_deg:
                    err_multipath = self.multipath_gen.generate(prn, elev)

                # === 3. 合成伪距 (修正漏扣的真实卫星钟差) ===
                # 物理公式: P = ρ_geo - c*dt_sv + Iono + Tropo + c*dt_rx + 白噪声 + 多径
                c_dt_sv = SPEED_OF_LIGHT * sv_clk_sec
                pseudorange = (rho_geo - c_dt_sv + err_sisre + err_iono + err_tropo
                               + err_rclk + err_obs + err_multipath)
                
                # === 模拟 SNR (与高度角/多径相关) ===
                # 基准 SNR ∝ sin(E), 多径环境降低
                snr_base = 25.0 + 20.0 * math.sin(elev)
                if scene.multipath_enabled and elev_deg < scene.multipath_elev_threshold_deg:
                    snr_base -= abs(err_multipath) * 0.8
                snr = max(snr_base + random.gauss(0, 2.0), 10.0)

                epoch_sats[prn] = {
                    'pseudorange': pseudorange,
                    'snr': snr,
                    'elev_deg': elev_deg,
                    'az_deg': math.degrees(az),
                    'rho_geo': rho_geo,
                    'err_sisre': err_sisre,
                    'err_iono': err_iono,
                    'err_tropo': err_tropo,
                    'err_rclk': err_rclk,
                    'err_obs': err_obs,
                    'err_multipath': err_multipath,
                    'sv_pos': sv_pos.copy(),
                }

            if len(epoch_sats) >= 4:
                obs_blocks.append((current_time, epoch_sats))

                # CSV 记录 (每颗卫星一行)
                for prn, sdata in epoch_sats.items():
                    epoch_records.append({
                        'epoch': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        'epoch_idx': ep_idx,
                        'prn': prn,
                        'pseudorange': sdata['pseudorange'],
                        'snr': sdata['snr'],
                        'elev_deg': sdata['elev_deg'],
                        'az_deg': sdata['az_deg'],
                        'rho_geo': sdata['rho_geo'],
                        'err_total': (sdata['err_sisre'] + sdata['err_iono']
                                      + sdata['err_tropo'] + sdata['err_rclk']
                                      + sdata['err_obs'] + sdata['err_multipath']),
                        'err_multipath': sdata['err_multipath'],
                        'n_sats': len(epoch_sats),
                        'scene': scene.name,
                    })

        print(f"[Simulator] 共生成 {len(obs_blocks)} 个有效历元")

        # 写入 RINEX OBS 文件
        self._write_rinex_obs(obs_path, obs_blocks, start_time, interval_sec)

        # 写入 CSV
        df = pd.DataFrame(epoch_records)
        df.to_csv(csv_path, index=False, float_format='%.6f')
        print(f"[Simulator] OBS 文件已保存: {obs_path}")
        print(f"[Simulator] CSV 文件已保存: {csv_path}")

        return obs_path, csv_path

    def generate_all_scenes(
        self,
        start_time: datetime,
        duration_hours: float = 24.0,
        interval_sec: float = 30.0,
    ) -> Dict[str, str]:
        """
        批量生成三大场景数据

        Returns:
            {scene_name: csv_path} 映射
        """
        results = {}

        scenes = [SCENE_OPEN_SKY, SCENE_TREE_CANOPY, SCENE_URBAN_CANYON]
        date_str = start_time.strftime("%Y%m%d")

        for scene in scenes:
            obs_fn = f"BUPT_Shahe_{date_str}_{scene.name}.obs"
            csv_fn = f"{scene.name}.csv"

            # 只有开阔场景写主 OBS 文件
            if scene.name == "scene1_open_sky":
                obs_fn = f"BUPT_Shahe_{date_str}.obs"

            _, csv_path = self.generate_obs_file(
                start_time=start_time,
                duration_hours=duration_hours,
                interval_sec=interval_sec,
                scene=scene,
                obs_filename=obs_fn,
                csv_filename=csv_fn,
            )
            results[scene.name] = csv_path

        return results

    # ================================================================
    #  RINEX 3.x OBS 文件写入
    # ================================================================
    def _write_rinex_obs(
        self,
        filepath: str,
        obs_blocks: List[Tuple[datetime, dict]],
        start_time: datetime,
        interval: float,
    ) -> None:
        """将生成的观测数据写入 RINEX 3.04 OBS 格式文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            # ---- HEADER ---- (保持不变)
            f.write(f"{'3.04':>9s}{'':11s}{'OBSERVATION DATA':20s}"
                    f"{'C: BDS':20s}RINEX VERSION / TYPE\n")
            f.write(f"{'DataSimulator':20s}{'BUPT-GNSS':20s}"
                    f"{datetime.utcnow().strftime('%Y%m%d %H%M%S'):20s}"
                    f"PGM / RUN BY / DATE\n")
            f.write(f"BDS SPP Simulated Observation Data"
                    f"{'':26s}COMMENT\n")
            f.write(f"BUPT_Shahe{'':10s}{'':20s}{'':20s}MARKER NAME\n")

            # 先验坐标
            f.write(f"{self.gt_ecef[0]:14.4f}{self.gt_ecef[1]:14.4f}"
                    f"{self.gt_ecef[2]:14.4f}{'':18s}APPROX POSITION XYZ\n")

            # 天线偏移
            f.write(f"{'0.0000':14s}{'0.0000':14s}{'0.0000':14s}"
                    f"{'':18s}ANTENNA: DELTA H/E/N\n")

            # BDS 观测类型: C2I, S2I
            f.write(f"C    2 C2I S2I{'':46s}SYS / # / OBS TYPES\n")

            # 时间系统
            f.write(f"{'BDT':>60s}TIME OF FIRST OBS\n")

            # 采样间隔
            f.write(f"{interval:10.3f}{'':50s}INTERVAL\n")

            f.write(f"{'':60s}END OF HEADER\n")

            # ---- DATA BODY ----
            for epoch_time, sats in obs_blocks:
                n_sv = len(sats)
                # 历元头: > YYYY MM DD HH MM SS.SSSSSSS  0  nSV
                f.write(f"> {epoch_time.year:4d} {epoch_time.month:02d} "
                        f"{epoch_time.day:02d} {epoch_time.hour:02d} "
                        f"{epoch_time.minute:02d}{epoch_time.second:11.7f}"
                        f"  0{n_sv:3d}\n")

                # 各卫星观测数据
                for prn in sorted(sats.keys()):
                    sdata = sats[prn]
                    pr = sdata['pseudorange']
                    snr = sdata['snr']
                    
                    # ============================================================
                    # 【关键修复】RINEX 3.x 标准格式:
                    #   PRN(3) + C2I(14.3f) + LLI(1) + SSI(1) + S2I(14.3f) + LLI(1) + SSI(1)
                    #   每个观测值占 16 字符 (14数值 + 2标记)
                    # ============================================================
                    f.write(f"{prn:3s}{pr:14.3f}  {snr:14.3f}  \n")
                    #                      ^^              ^^
                    #                      LLI+SSI 各占1字符，用空格填充


# ============================================================
#  独立运行入口
# ============================================================
if __name__ == "__main__":
    import sys

    nav_file = os.path.join("data", "BUPT_20260510.nav")
    if not os.path.exists(nav_file):
        print(f"[Error] 导航文件不存在: {nav_file}")
        sys.exit(1)

    simulator = DataSimulator(nav_file, output_dir="data")

    # 起始时间: 2026-05-10 00:00:00 BDT
    start = datetime(2026, 5, 10, 0, 0, 0)

    # 生成三大场景 (24 小时, 30 秒采样)
    scene_paths = simulator.generate_all_scenes(
        start_time=start,
        duration_hours=24.0,
        interval_sec=30.0,
    )

    for name, path in scene_paths.items():
        print(f"  {name}: {path}")

    print("\n[Simulator] 全部场景数据生成完毕!")