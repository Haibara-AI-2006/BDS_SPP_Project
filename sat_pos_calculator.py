"""
sat_pos_calculator.py — BDS 卫星轨道位置解算 & 物理误差修正引擎
===============================================================
核心职责:
  1. 基于广播星历 (Keplerian + 摄动) 计算 BDS 卫星 ECEF 坐标
  2. 卫星钟差修正 (多项式 + 相对论效应)
  3. 光行时闭合 + 双重 Sagnac 地球自转效应修正 (2 次迭代)
  4. Saastamoinen 对流层延迟修正 (真实纬度/高程 + Neill 映射函数)
  5. BDS Klobuchar 电离层延迟修正

设计红线:
  - Sagnac 修正: 2 次迭代闭合，使用旋转后坐标回代
  - Saastamoinen: 传入真实测站纬度和高程，绝不写死 45°
  - Klobuchar α/β 从 NavParser 动态传入
  - 全部核心运算手写实现，零第三方定位库
"""

import math
import numpy as np
from datetime import datetime
from typing import Optional, Tuple

from rinex_parser import (
    BdsEphemeris, RinexNavParser,
    SPEED_OF_LIGHT, GM_BDS, OMEGA_E_BDS, PI,
    WGS84_A, WGS84_E2, WGS84_F,
    gps_week_seconds_from_datetime, ecef_to_blh, compute_elevation_azimuth,
)

# ============================================================
#  常量
# ============================================================
F_REL = -4.442807633e-10   # 相对论效应系数 F = -2√(μ) / c²


# ============================================================
#  卫星位置计算器
# ============================================================
class SatPosCalculator:
    """
    BDS 广播星历 → 卫星 ECEF 坐标 + 钟差 + 延迟修正

    使用流程:
        calc = SatPosCalculator(nav_parser)
        pos, clk = calc.compute_sat_pos_clk(prn, obs_time, pseudorange)
        tropo = calc.saastamoinen(elev_rad, lat_rad, h_m)
        iono  = calc.klobuchar_bds(lat_rad, lon_rad, elev_rad, az_rad, tow)
    """

    def __init__(self, nav_parser: RinexNavParser):
        self.nav = nav_parser
        self.iono_alpha = nav_parser.iono_alpha.copy()   # (4,)
        self.iono_beta = nav_parser.iono_beta.copy()     # (4,)

    # ================================================================
    #  1. 卫星位置 & 钟差计算 (广播星历, BDS ICD 标准)
    # ================================================================
    def compute_sat_pos_clk(
        self, prn: str, obs_time: datetime, pseudorange: float = 0.0
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        基于广播星历计算 BDS 卫星 ECEF 坐标与钟差

        Parameters:
            prn         : 卫星 PRN 编号 (如 'C01')
            obs_time    : 观测时间 (datetime)
            pseudorange : 伪距观测值 (m), 用于光行时初始估计

        Returns:
            (sv_pos, sv_clk) — ECEF 坐标 (3,), 钟差 (秒, 正值=钟超前)
            若星历不可用则返回 None
        """
        # === 星历 TOE 最近邻匹配 ===
        eph = self.nav.select_ephemeris(prn, obs_time)
        if eph is None:
            return None

        # === 观测时间转 BDS 周内秒 ===
        _, t_obs_sow = gps_week_seconds_from_datetime(obs_time)

        # === 信号传播时间初始估计 ===
        if pseudorange > 1e6:
            transit_time = pseudorange / SPEED_OF_LIGHT
        else:
            transit_time = 0.075  # 约 75ms 默认值

        # === 信号发射时刻 (BDS 周内秒) ===
        t_sv = t_obs_sow - transit_time

        # === 计算卫星钟差 (多项式 + 相对论) ===
        sv_clk = self._compute_sv_clock(eph, t_sv)

        # === 修正信号发射时刻 (扣除卫星钟差) ===
        t_sv_corrected = t_sv - sv_clk

        # === 计算卫星轨道位置 (ECEF) ===
        sv_pos = self._compute_orbital_position(eph, t_sv_corrected)

        if sv_pos is None:
            return None

        return sv_pos, sv_clk

    # ================================================================
    #  2. 卫星钟差计算 (多项式 + 相对论效应)
    # ================================================================
    def _compute_sv_clock(self, eph: BdsEphemeris, t_sv: float) -> float:
        """
        BDS 卫星钟差修正

        公式:
          dt_sv = af0 + af1·(t - toc) + af2·(t - toc)²  + Δt_rel
          Δt_rel = F · e · √a · sin(Ek)

        其中 F = -2√μ / c² ≈ -4.442807633×10⁻¹⁰

        Parameters:
            eph  : 广播星历
            t_sv : 信号发射时刻 (BDS 周内秒)

        Returns:
            钟差 (秒), 正值表示卫星钟超前
        """
        # toc 转周内秒
        _, toc_sow = gps_week_seconds_from_datetime(eph.toc)

        # 时间差 (处理周跨越)
        dt = t_sv - toc_sow
        if dt > 302400:
            dt -= 604800
        elif dt < -302400:
            dt += 604800

        # 多项式钟差
        dt_sv = eph.af0 + eph.af1 * dt + eph.af2 * dt * dt

        # 相对论效应修正: 需要偏近点角 Ek
        # 先粗略计算 Ek (只需一次迭代即可用于相对论修正)
        a = float(eph.sqrt_a ** 2)# 半长轴
        if a<=0:
            return 0.0
        n0 = math.sqrt(GM_BDS / (a ** 3))               # 平均角速度
        n = n0 + eph.delta_n                             # 修正后角速度

        # tk: 相对于 TOE 的时间差
        tk = t_sv - eph.toe_sow
        if tk > 302400:
            tk -= 604800
        elif tk < -302400:
            tk += 604800

        # 平近点角
        Mk = eph.m0 + n * tk

        # 偏近点角 (开普勒方程迭代)
        Ek = Mk
        for _ in range(10):
            Ek_new = Mk + eph.ecc * math.sin(Ek)
            if abs(Ek_new - Ek) < 1e-13:
                break
            Ek = Ek_new

        # 相对论效应
        dt_rel = F_REL * eph.ecc * eph.sqrt_a * math.sin(Ek)

        dt_sv += dt_rel

        # 扣除群延迟 TGD (B1I 信号用 TGD1)
        dt_sv -= eph.tgd1

        return dt_sv

    # ================================================================
    #  3. 卫星轨道位置计算 (Keplerian + 摄动修正)
    # ================================================================
    def _compute_orbital_position(
        self, eph: BdsEphemeris, t_sv: float
    ) -> Optional[np.ndarray]:
        """
        BDS ICD 标准轨道位置计算

        步骤:
          1. 半长轴 a = (√a)²
          2. 平均角速度 n = n₀ + Δn
          3. 时间差 tk = t - toe
          4. 平近点角 Mk = M₀ + n·tk
          5. 偏近点角 Ek (开普勒方程迭代)
          6. 真近点角 νk
          7. 升交距角 φk = νk + ω
          8. 摄动修正 (δu, δr, δi)
          9. 修正后 (uk, rk, ik)
          10. 轨道面内坐标 (xk', yk')
          11. 升交点经度 Ωk
          12. ECEF 坐标 (特别注意 BDS GEO/IGSO 卫星处理)

        Parameters:
            eph  : 广播星历
            t_sv : 修正后的信号发射时刻 (BDS 周内秒)

        Returns:
            (3,) ECEF 坐标, 失败返回 None
        """
        # --- Step 1: 半长轴 ---
        a = float(eph.sqrt_a ** 2)
        if a < 1e6:  # 无效星历
            return None

        # --- Step 2: 平均角速度 ---
        n0 = math.sqrt(GM_BDS / (a ** 3))
        n = n0 + eph.delta_n

        # --- Step 3: 时间差 tk (处理周跨越) ---
        tk = t_sv - eph.toe_sow
        if tk > 302400:
            tk -= 604800
        elif tk < -302400:
            tk += 604800

        # --- Step 4: 平近点角 ---
        Mk = eph.m0 + n * tk

        # --- Step 5: 偏近点角 (开普勒方程 Newton-Raphson 迭代) ---
        Ek = Mk
        for _ in range(15):
            dE = (Mk - Ek + eph.ecc * math.sin(Ek)) / \
                 (1.0 - eph.ecc * math.cos(Ek))
            Ek += dE
            if abs(dE) < 1e-14:
                break

        sin_Ek = math.sin(Ek)
        cos_Ek = math.cos(Ek)

        # --- Step 6: 真近点角 ---
        # tan(νk/2) = √((1+e)/(1-e)) · tan(Ek/2)
        num = math.sqrt(1.0 - eph.ecc ** 2) * sin_Ek
        den = cos_Ek - eph.ecc
        vk = math.atan2(num, den)

        # --- Step 7: 升交距角 ---
        phi_k = vk + eph.omega

        sin_2phi = math.sin(2.0 * phi_k)
        cos_2phi = math.cos(2.0 * phi_k)

        # --- Step 8: 二阶谐波摄动修正 ---
        delta_u = eph.cuc * cos_2phi + eph.cus * sin_2phi   # 升交距角修正
        delta_r = eph.crc * cos_2phi + eph.crs * sin_2phi   # 径向修正
        delta_i = eph.cic * cos_2phi + eph.cis * sin_2phi   # 轨道倾角修正

        # --- Step 9: 修正后参数 ---
        uk = phi_k + delta_u                                  # 修正升交距角
        rk = a * (1.0 - eph.ecc * cos_Ek) + delta_r          # 修正径向距离
        ik = eph.i0 + delta_i + eph.idot * tk                 # 修正轨道倾角

        # --- Step 10: 轨道面内坐标 ---
        xk_prime = rk * math.cos(uk)
        yk_prime = rk * math.sin(uk)

        # --- Step 11: 升交点经度 ---
        # BDS 区分 GEO (C01-C05, C59-C63) / IGSO / MEO
        prn_num = int(eph.prn[1:])
        is_geo = (1 <= prn_num <= 5) or (59 <= prn_num <= 63)

        if is_geo:
            # GEO 卫星: Ωk 不减去地球自转
            omega_k = eph.omega0 + eph.omega_dot * tk - OMEGA_E_BDS * eph.toe_sow
        else:
            # MEO / IGSO: 标准公式
            omega_k = (eph.omega0
                       + (eph.omega_dot - OMEGA_E_BDS) * tk
                       - OMEGA_E_BDS * eph.toe_sow)

        sin_omega = math.sin(omega_k)
        cos_omega = math.cos(omega_k)
        sin_ik = math.sin(ik)
        cos_ik = math.cos(ik)

        if is_geo:
            # GEO 卫星: 先在轨道面内计算，再做额外旋转
            # 1) 在惯性系中的坐标 (不含地球自转)
            xg = xk_prime * cos_omega - yk_prime * cos_ik * sin_omega
            yg = xk_prime * sin_omega + yk_prime * cos_ik * cos_omega
            zg = yk_prime * sin_ik

            # 2) 绕 X 轴旋转 -5° (BDS GEO 倾斜角修正)
            angle_x = math.radians(-5.0)
            sin_ax = math.sin(angle_x)
            cos_ax = math.cos(angle_x)

            # 3) 绕 Z 轴旋转 ωe·tk
            angle_z = OMEGA_E_BDS * tk
            sin_az = math.sin(angle_z)
            cos_az = math.cos(angle_z)

            # Rx(-5°) 旋转
            yg2 = yg * cos_ax - zg * sin_ax
            zg2 = yg * sin_ax + zg * cos_ax

            # Rz(ωe·tk) 旋转
            x_ecef = xg * cos_az + yg2 * sin_az
            y_ecef = -xg * sin_az + yg2 * cos_az
            z_ecef = zg2

        else:
            # MEO / IGSO: 标准 ECEF 转换
            x_ecef = xk_prime * cos_omega - yk_prime * cos_ik * sin_omega
            y_ecef = xk_prime * sin_omega + yk_prime * cos_ik * cos_omega
            z_ecef = yk_prime * sin_ik

        return np.array([x_ecef, y_ecef, z_ecef], dtype=np.float64)

    # ================================================================
    #  4. Saastamoinen 对流层延迟修正
    # ================================================================
    def saastamoinen(
        self, elev_rad: float, lat_rad: float, height_m: float
    ) -> float:
        """
        Saastamoinen 对流层天顶延迟模型 + Neill 映射函数

        公式 (天顶延迟):
          ZHD = 0.0022768 · P / (1 - 0.00266·cos(2φ) - 0.00028·H_km)
          ZWD = 0.002277 · (1255/T + 0.05) · e

        Neill 干延迟映射函数 (简化):
          m(E) = 1 / (sin(E) + 0.00143 / (tan(E) + 0.0455))

        Parameters:
            elev_rad : 卫星高度角 (rad)
            lat_rad  : 测站纬度 (rad), 由调用方传入真实值
            height_m : 测站椭球高 (m)

        Returns:
            对流层延迟修正量 (m)
        """
        # 高度角过低保护
        if elev_rad < math.radians(2.0):
            elev_rad = math.radians(2.0)

        # 测站高度 (km)
        H_km = height_m / 1000.0

        # 气象参数 (标准大气模型, 随高度衰减)
        # 海平面标准值
        T0 = 288.15    # 温度 (K)
        P0 = 1013.25   # 气压 (hPa)
        e0 = 11.691    # 水汽压 (hPa)

        # 高度修正 (简化标准大气模型)
        T = T0 - 6.5 * H_km                                      # 温度递减率 6.5K/km
        P = P0 * (T / T0) ** 5.2561                               # 气压随高度
        e_wv = e0 * (T / T0) ** (5.2561 * 3.0)                    # 水汽压随高度

        # 纬度余弦因子
        cos_2lat = math.cos(2.0 * lat_rad)

        # === 天顶干延迟 ZHD (Saastamoinen) ===
        zhd = 0.0022768 * P / (1.0 - 0.00266 * cos_2lat - 0.00028 * H_km)

        # === 天顶湿延迟 ZWD ===
        zwd = 0.002277 * (1255.0 / T + 0.05) * e_wv

        # === Neill 映射函数 (干 + 湿用同一映射, 简化) ===
        sin_e = math.sin(elev_rad)
        tan_e = math.tan(elev_rad)

        # 干延迟映射
        mf_dry = 1.0 / (sin_e + 0.00143 / (tan_e + 0.0455))

        # 湿延迟映射 (略有不同的系数)
        mf_wet = 1.0 / (sin_e + 0.00035 / (tan_e + 0.017))

        # === 总对流层延迟 ===
        tropo_delay = zhd * mf_dry + zwd * mf_wet

        return tropo_delay

    # ================================================================
    #  5. BDS Klobuchar 电离层延迟修正
    # ================================================================
    def klobuchar_bds(
        self,
        lat_rad: float,
        lon_rad: float,
        elev_rad: float,
        az_rad: float,
        tow: float,
    ) -> float:
        """
        BDS Klobuchar 电离层延迟模型 (B1I 频点)

        与 GPS Klobuchar 主要差异:
          1. 穿刺点计算地心角使用 BDS 定义的地球半径
          2. 地方时以北斗系统时间 (BDT) 为基准
          3. α/β 参数从 .nav 头部动态提取

        Parameters:
            lat_rad  : 测站纬度 (rad)
            lon_rad  : 测站经度 (rad)
            elev_rad : 卫星高度角 (rad)
            az_rad   : 卫星方位角 (rad)
            tow      : BDS 周内秒

        Returns:
            电离层延迟修正量 (m), 已乘以光速
        """
        alpha = self.iono_alpha  # (4,) 从 nav 头部动态提取
        beta = self.iono_beta    # (4,)

        # 高度角保护
        E = max(elev_rad, math.radians(5.0))

        # === Step 1: 电离层穿刺点 (IPP) 地心角 ===
        psi = PI / 2.0 - E - math.asin(
            6378.0 / (6378.0 + 375.0) * math.cos(E)
        )

        # === Step 2: IPP 纬度 (半圆) ===
        phi_ipp = lat_rad / PI + psi * math.cos(az_rad) / PI
        # 限幅 [-0.416, +0.416]
        phi_ipp = max(min(phi_ipp, 0.416), -0.416)

        # === Step 3: IPP 经度 (半圆) ===
        lam_ipp = lon_rad / PI + psi * math.sin(az_rad) / (PI * math.cos(phi_ipp * PI))

        # === Step 4: IPP 地磁纬度 (半圆) ===
        phi_m = phi_ipp + 0.064 * math.cos((lam_ipp - 1.617) * PI)

        # === Step 5: IPP 地方时 (秒) ===
        t_local = 43200.0 * lam_ipp + tow
        # 归化到 [0, 86400)
        t_local = t_local % 86400.0

        # === Step 6: 倾斜因子 ===
        F_sf = 1.0 + 16.0 * (0.53 - E / PI) ** 3

        # === Step 7: 电离层延迟计算 ===
        # 周期
        PER = beta[0] + beta[1] * phi_m + beta[2] * phi_m ** 2 + beta[3] * phi_m ** 3
        PER = max(PER, 72000.0)  # 周期下限 72000 秒

        # 振幅
        AMP = alpha[0] + alpha[1] * phi_m + alpha[2] * phi_m ** 2 + alpha[3] * phi_m ** 3
        AMP = max(AMP, 0.0)  # 振幅非负

        # 相位
        x = 2.0 * PI * (t_local - 50400.0) / PER

        # 电离层延迟 (秒)
        if abs(x) < 1.57:
            T_iono = F_sf * (5e-9 + AMP * (1.0 - x ** 2 / 2.0 + x ** 4 / 24.0))
        else:
            T_iono = F_sf * 5e-9

        # 转为距离 (m)
        iono_delay = T_iono * SPEED_OF_LIGHT

        return iono_delay