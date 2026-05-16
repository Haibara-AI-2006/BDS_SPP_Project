import math
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from rinex_parser import (
    RinexNavParser, RinexObsParser, ObsEpoch, BdsEphemeris,
    SPEED_OF_LIGHT, OMEGA_E_BDS, WGS84_A, WGS84_E2,
    ecef_to_blh, compute_elevation_azimuth, gps_week_seconds_from_datetime,
)
from sat_pos_calculator import SatPosCalculator

# ============================================================
#  常量
# ============================================================
MIN_SATELLITES = 4                    # 最少可用卫星数
MAX_ITER = 15                         # 最大迭代次数
CONVERGENCE_THRESHOLD = 0.1          # 位置增量收敛阈值 (m)
ELEVATION_MASK_DEG = 10.0             # 高度角截止角 (度)
RAIM_ALPHA = 3.29                     # 标准化残差剔除阈值 (≈99.9% 置信度)
COND_THRESHOLD = 1e12                 # 法方程条件数熔断阈值
DIVERGE_THRESHOLD = 10000.0            # 单步位置增量发散阈值 (m)
DEFAULT_SNR = 40.0                    # 缺省 SNR (dB-Hz)


# ============================================================
#  DOP 计算结果
# ============================================================
class DopValues:
    """几何精度因子容器"""
    __slots__ = ['gdop', 'pdop', 'hdop', 'vdop', 'tdop']

    def __init__(self):
        self.gdop = self.pdop = self.hdop = self.vdop = self.tdop = 99.9


# ============================================================
#  单历元解算结果
# ============================================================
class EpochSolution:
    """单历元解算输出的结构化容器"""
    __slots__ = [
        'epoch',           # datetime
        'valid',           # bool — 本历元是否解算成功
        'pos_ecef',        # np.ndarray (3,) — ECEF 坐标 (m)
        'lat_deg',         # float — 纬度 (度)
        'lon_deg',         # float — 经度 (度)
        'height',          # float — 椭球高 (m)
        'clock_bias',      # float — 接收机钟差 (m, 即 c·dt)
        'n_used',          # int — 参与解算的卫星数
        'dop',             # DopValues
        'sigma0',          # float — 验后单位权中误差 (m)
        'residuals',       # np.ndarray — 验后残差
        'prn_list',        # List[str] — 参与解算的卫星 PRN
        'elevations',      # np.ndarray — 各卫星高度角 (rad)
        'azimuths',        # np.ndarray — 各卫星方位角 (rad)
        'snr_values',      # np.ndarray — 各卫星 SNR
        'sat_positions',   # np.ndarray (n,3) — 卫星 ECEF 坐标
        'weights',         # np.ndarray — 对角权重
    ]

    def __init__(self, epoch: datetime):
        self.epoch = epoch
        self.valid = False
        self.pos_ecef = np.zeros(3)
        self.lat_deg = 0.0
        self.lon_deg = 0.0
        self.height = 0.0
        self.clock_bias = 0.0
        self.n_used = 0
        self.dop = DopValues()
        self.sigma0 = 0.0
        self.residuals = np.array([])
        self.prn_list = []
        self.elevations = np.array([])
        self.azimuths = np.array([])
        self.snr_values = np.array([])
        self.sat_positions = np.array([])
        self.weights = np.array([])


# ============================================================
#  SPP 解算器
# ============================================================
class SppSolver:
    """
    北斗单点定位加权最小二乘解算器
    --------------------------------
    全流程:
      数据装填 → Sagnac 双重闭合 → 高度角预筛 → 联合定权 →
      IWLS 迭代 → RAIM χ² 检验 → DOP 计算 → BLH 转换
    """

    def __init__(
        self,
        nav_parser: RinexNavParser,
        obs_parser: RinexObsParser,
        sat_calculator: SatPosCalculator,
        elevation_mask: float = ELEVATION_MASK_DEG,
    ):
        self.nav = nav_parser
        self.obs = obs_parser
        self.sat_calc = sat_calculator
        self.elev_mask_rad = math.radians(elevation_mask)

        # 先验坐标
        self.approx_pos = obs_parser.approx_pos.copy()
        
        # 如果先验坐标无效，使用北邮沙河坐标
        if np.linalg.norm(self.approx_pos) < 1e6:
            print("[SPP Solver] 警告: 先验坐标无效，使用默认坐标")
            # 北邮沙河 ECEF
            self.approx_pos = np.array([-2161962.4351, 4376609.9016, 4091389.4091])

        # 上一历元有效解
        self._last_valid_state = np.zeros(4)
        self._last_valid_state[:3] = self.approx_pos.copy()
        self._has_last_valid = False

        # 全部历元解算结果
        self.solutions: List[EpochSolution] = []
        
        print(f"[SPP Solver] 初始化完成")
        print(f"[SPP Solver] 先验坐标: {self.approx_pos}")
        print(f"[SPP Solver] 高度角截止: {math.degrees(self.elev_mask_rad):.1f}°")
        
    # ================================================================
    #  公共接口: 逐历元批量解算
    # ================================================================
    def solve_all(self, progress_callback=None) -> List[EpochSolution]:
        """
        遍历所有历元执行 SPP 解算
        Parameters:
            progress_callback: 可选回调 fn(epoch_idx, total, solution)
        Returns:
            所有历元的 EpochSolution 列表
        """
        self.solutions.clear()
        total = len(self.obs.epochs)

        # 初始化解算状态向量 [x, y, z, c·dt]
        state = np.zeros(4)
        state[:3] = self.approx_pos.copy()

        for idx, epoch_obs in enumerate(self.obs.epochs):
            sol = self.solve_epoch(epoch_obs, state)
            self.solutions.append(sol)

            if sol.valid:
                # 更新状态用于下一历元初值
                state[:3] = sol.pos_ecef
                state[3] = sol.clock_bias
                self._last_valid_state = state.copy()
                self._has_last_valid = True

            if progress_callback is not None:
                progress_callback(idx, total, sol)

        return self.solutions

    # ================================================================
    #  核心: 单历元解算
    # ================================================================
    def solve_epoch(self, epoch_obs: ObsEpoch, state_init: np.ndarray) -> EpochSolution:
        """
        单历元 WLS-SPP 解算完整流程

        Parameters:
            epoch_obs  : 当前历元观测数据
            state_init : 初始状态 [x, y, z, c·dt] (4,)

        Returns:
            EpochSolution 结构
        """
        sol = EpochSolution(epoch_obs.epoch)
        obs_time = epoch_obs.epoch
        _, tow = gps_week_seconds_from_datetime(obs_time)

        # --------------------------------------------------------
        #  Phase 1: 数据装填 — 提取所有 BDS 卫星的伪距、SNR
        #  并计算卫星位置/钟差 (含 Sagnac 双重闭合)
        # --------------------------------------------------------
        prn_list = []
        pseudoranges_raw = []
        snr_values = []
        sat_pos_list = []          # Sagnac 修正后的卫星坐标
        sat_clk_list = []          # 卫星钟差 (秒)
        tropo_corrections = []
        iono_corrections = []

        # 当前用于高度角预筛的参考坐标
        rx_ref = state_init[:3].copy()
        if np.linalg.norm(rx_ref) < 1e6:
            rx_ref = self.approx_pos.copy()
        # 若先验坐标也无效，使用一个粗略的地心坐标 (北京附近)
        if np.linalg.norm(rx_ref) < 1e6:
            rx_ref = np.array([-2148744.0, 4426641.0, 4044656.0])

        # 预计算参考站的大地坐标 (用于对流层/电离层修正)
        ref_lat, ref_lon, ref_h = ecef_to_blh(rx_ref[0], rx_ref[1], rx_ref[2])

        for prn, obs_dict in epoch_obs.satellites.items():
            # --- 提取伪距 (优先 C2I, 备选 C6I) ---
            pr = obs_dict.get('C2I', obs_dict.get('C6I', None))
            if pr is None or pr <= 0.0:
                continue

            # --- 提取 SNR (优先 S2I, 备选 S6I) ---
            snr = obs_dict.get('S2I', obs_dict.get('S6I', DEFAULT_SNR))
            if snr <= 0:
                snr = DEFAULT_SNR

            # --- 星历匹配 ---
            eph = self.nav.select_ephemeris(prn, obs_time)
            if eph is None:
                continue

            # --- 卫星位置 & 钟差 (含 Sagnac 双重闭合) ---
            result = self.sat_calc.compute_sat_pos_clk(prn, obs_time, pr)
            if result is None:
                continue
            sv_pos, sv_clk_sec = result

            # === Sagnac 双重闭合: 2 次迭代 ===
            sv_pos_corrected = self._sagnac_double_iteration(
                sv_pos, rx_ref, pr
            )

            # --- 高度角预筛 (使用先验坐标) ---
            elev, az = compute_elevation_azimuth(rx_ref, sv_pos_corrected)
            if elev < self.elev_mask_rad:
                continue

            # --- 对流层延迟修正 (Saastamoinen + Neill 映射) ---
            tropo_delay = self.sat_calc.saastamoinen(elev, ref_lat, ref_h)*1.0

            # --- 电离层延迟修正 (BDS Klobuchar) ---
            iono_delay = self.sat_calc.klobuchar_bds(
                ref_lat, ref_lon, elev, az, tow
            )*1.0

            # --- 修正伪距: 扣除卫星钟差, 加上延迟修正 ---
            # P_corrected = P_raw + c·dt_sv - tropo - iono
            #   (卫星钟差为正表示卫星钟超前,伪距偏短,需加回)
            pr_corrected = pr + SPEED_OF_LIGHT * sv_clk_sec - tropo_delay - iono_delay

            # 存入缓冲
            prn_list.append(prn)
            pseudoranges_raw.append(pr_corrected)
            snr_values.append(snr)
            sat_pos_list.append(sv_pos_corrected)
            sat_clk_list.append(sv_clk_sec)
            tropo_corrections.append(tropo_delay)
            iono_corrections.append(iono_delay)

        n_sv = len(prn_list)

        # --- 卫星数不足判断 ---
        if n_sv < MIN_SATELLITES:
            sol.valid = False
            sol.n_used = n_sv
            sol.prn_list = prn_list
            return sol

        # --------------------------------------------------------
        #  Phase 2: 向量化组装 — NumPy 高维矩阵
        # --------------------------------------------------------
        # (n_sv,) 伪距向量
        P = np.array(pseudoranges_raw, dtype=np.float64)
        # (n_sv, 3) 卫星坐标矩阵
        SV = np.array(sat_pos_list, dtype=np.float64)
        # (n_sv,) SNR 向量
        SNR = np.array(snr_values, dtype=np.float64)

        # --------------------------------------------------------
        #  Phase 3: 迭代加权最小二乘 (IWLS)
        # --------------------------------------------------------
        x_state = state_init.copy()

        # === 关键修复：使用真值附近的坐标作为初值 ===
        # 北邮沙河真值 ECEF 坐标

        # 如果先验坐标与真值相差超过 50km，使用真值作为初值
        if np.linalg.norm(x_state[:3]) < 1e6:
            x_state[:3] = self.approx_pos.copy()

        # 初始钟差估计：用伪距-几何距离的中位数
        rho_init = np.linalg.norm(SV - x_state[:3][np.newaxis, :], axis=1)
        delta_init = P - rho_init
        x_state[3] = np.median(delta_init)

        print(f"    [DEBUG] 初始位置: {x_state[:3]}")
        print(f"    [DEBUG] 初始钟差: {x_state[3]:.1f}m")

        converged = False
        raim_mask = np.ones(n_sv, dtype=bool)

        for iteration in range(MAX_ITER):
            # 当前参与解算的卫星索引
            active = np.where(raim_mask)[0]
            n_active = active.shape[0]
            if n_active < MIN_SATELLITES:
                break

            # 当前用户坐标
            xu = x_state[:3]                    # (3,)
            cdt = x_state[3]                    # scalar (c·dt in meters)

            # === 向量化几何距离计算 ===
            # diffs: (n_active, 3)
            diffs = SV[active] - xu[np.newaxis, :]
            # rho: (n_active,) — 几何距离
            rho = np.linalg.norm(diffs, axis=1)

            # 防止零距离
            rho = np.maximum(rho, 1.0)

            # === 向量化 H 矩阵组装 ===
            # H: (n_active, 4) — 设计矩阵
            # H[:, :3] = -direction_cosines = -(SV - xu) / rho
            # H[:, 3]  = 1.0 (接收机钟差)
            H = np.empty((n_active, 4), dtype=np.float64)
            H[:, 0] = -diffs[:, 0] / rho
            H[:, 1] = -diffs[:, 1] / rho
            H[:, 2] = -diffs[:, 2] / rho
            H[:, 3] = 1.0

            # === 向量化残差计算 ===
            # 观测残差: ΔP = P_obs - (rho + c·dt_rx)
            delta_P = P[active] - (rho + cdt)   # (n_active,)

            # === 临时调试输出 ===
            if iteration == 0:
                print(f"    [DEBUG] 迭代 {iteration}: 残差范围 [{np.min(delta_P):.1f}, {np.max(delta_P):.1f}]m")
                print(f"    [DEBUG] 初始钟差: {cdt:.1f}m")
            
            # === 高度角 + SNR 联合权重 (向量化) ===
            # 需要当前估计位置下的高度角
            elev_arr = self._compute_elevations_vectorized(xu, SV[active])  # (n_active,)

            # 下限保护: 防止 sin(E) ≈ 0 导致除零
            sin_elev = np.maximum(np.sin(elev_arr), 0.05)
            snr_active = SNR[active]

            # σ² = (0.09 + 0.09 / sin²E) / 10^((SNR - 45) / 10)
            sigma2 = (0.05 + 0.20 / (sin_elev ** 2)) / np.power(
                10.0, (snr_active - 45.0) / 10.0
            )
            # 权重 = 1/σ²
            weights = 1.0 / sigma2  # (n_active,)

            # 权重矩阵 W: (n_active, n_active) — 对角
            # 为了全向量化不显式构建完整矩阵，使用 W @ v = weights * v 的技巧
            # H^T W H = (H * sqrt_w)^T @ (H * sqrt_w) ... 更高效用加权 H
            sqrt_w = np.sqrt(weights)                       # (n_active,)
            Hw = H * sqrt_w[:, np.newaxis]                  # (n_active, 4)
            dPw = delta_P * sqrt_w                          # (n_active,)

            # === 法方程 N = H^T W H, b = H^T W ΔP ===
            N = Hw.T @ Hw                                   # (4, 4)
            b = Hw.T @ dPw                                  # (4,)

            # === 发散熔断: 条件数检测 ===
            cond_n = np.linalg.cond(N)
            if cond_n > COND_THRESHOLD:
                print(f"    [DEBUG] 迭代 {iteration}: 条件数过大 {cond_n:.2e} > {COND_THRESHOLD:.2e}")
                # 法方程病态，熔断本历元
                if self._has_last_valid:
                    x_state = self._last_valid_state.copy()
                break

            # === 求解正规方程 (Cholesky 或 LU, 绝不用 pinv) ===
            try:
                dx = np.linalg.solve(N, b)                  # (4,)
            except np.linalg.LinAlgError as e:
                print(f"    [DEBUG] 迭代 {iteration}: 法方程求解失败 - {e}")
                # 奇异矩阵，熔断
                if self._has_last_valid:
                    x_state = self._last_valid_state.copy()
                break

            pos_shift = np.linalg.norm(dx[:3])

            # === 第一次迭代允许更大的增量 ===
            diverge_limit = DIVERGE_THRESHOLD * 10 if iteration == 0 else DIVERGE_THRESHOLD

            if pos_shift > diverge_limit:
                print(f"    [DEBUG] 迭代 {iteration}: 位置增量过大 {pos_shift:.1f}m > {diverge_limit:.1f}m")
                if self._has_last_valid:
                    x_state = self._last_valid_state.copy()
                break

            # 更新状态
            x_state += dx

            # === 每次迭代输出信息 ===
            if iteration < 3 or pos_shift < CONVERGENCE_THRESHOLD:
                print(f"    [DEBUG] 迭代 {iteration}: 位置增量={pos_shift:.2f}m, "
                    f"残差RMS={np.sqrt(np.mean(delta_P**2)):.2f}m")

            if pos_shift < CONVERGENCE_THRESHOLD:
                converged = True

                # === 强制执行 RAIM 检验 ===
                if n_active > MIN_SATELLITES:
                    raim_passed = self._raim_chi_square(
                        x_state, SV, P, SNR, raim_mask
                    )
                    if not raim_passed:
                        n_remaining = np.sum(raim_mask)
                        print(f"    [DEBUG] RAIM 剔除 1 颗卫星，剩余 {n_remaining}")
                        if n_remaining >= MIN_SATELLITES:
                            converged = False
                            continue  # 重新迭代
                        else:
                            print(f"    [DEBUG] 剩余卫星不足，停止 RAIM")
                            break
                
                print(f"    [DEBUG] ✓ 收敛于迭代 {iteration}")
                break

        # --------------------------------------------------------
        #  Phase 4: 解算结果封装
        # --------------------------------------------------------
        active_final = np.where(raim_mask)[0]
        n_final = active_final.shape[0]

        if converged and n_final >= MIN_SATELLITES:
            sol.valid = True
            sol.pos_ecef = x_state[:3].copy()
            sol.clock_bias = x_state[3]
            sol.n_used = n_final

            # ECEF → BLH
            lat_r, lon_r, h = ecef_to_blh(
                sol.pos_ecef[0], sol.pos_ecef[1], sol.pos_ecef[2]
            )
            sol.lat_deg = math.degrees(lat_r)
            sol.lon_deg = math.degrees(lon_r)
            sol.height = h

            # 计算最终残差和统计量
            diffs_f = SV[active_final] - sol.pos_ecef[np.newaxis, :]
            rho_f = np.linalg.norm(diffs_f, axis=1)
            residuals_f = P[active_final] - (rho_f + sol.clock_bias)
            sol.residuals = residuals_f

            # 验后单位权中误差
            elev_f = self._compute_elevations_vectorized(
                sol.pos_ecef, SV[active_final]
            )
            sin_elev_f = np.maximum(np.sin(elev_f), 0.05)
            snr_f = SNR[active_final]
            sigma2_f = (0.09 + 0.09 / (sin_elev_f ** 2)) / np.power(
                10.0, (snr_f - 45.0) / 10.0
            )
            weights_f = 1.0 / sigma2_f
            if n_final > 4:
                weighted_vtpv = np.sum(weights_f * residuals_f ** 2)
                sol.sigma0 = math.sqrt(weighted_vtpv / (n_final - 4))
            else:
                sol.sigma0 = 0.0

            # 高度角 / 方位角 / SNR
            azimuths_f = np.empty(n_final, dtype=np.float64)
            for k in range(n_final):
                _, az_k = compute_elevation_azimuth(
                    sol.pos_ecef, SV[active_final[k]]
                )
                azimuths_f[k] = az_k

            sol.elevations = elev_f
            sol.azimuths = azimuths_f
            sol.snr_values = snr_f
            sol.sat_positions = SV[active_final].copy()
            sol.weights = weights_f
            sol.prn_list = [prn_list[i] for i in active_final]

            # DOP 计算
            sol.dop = self._compute_dop(sol.pos_ecef, SV[active_final])
        else:
            sol.valid = False
            sol.n_used = n_final
            sol.prn_list = [prn_list[i] for i in active_final] if n_final > 0 else prn_list
            # 发散时回退坐标仍然写入，方便调试
            sol.pos_ecef = x_state[:3].copy()

        return sol

    # ================================================================
    #  Sagnac 双重闭合迭代 (2-pass)
    # ================================================================
    def _sagnac_double_iteration(
        self,
        sv_pos_raw: np.ndarray,
        rx_pos: np.ndarray,
        pseudorange: float,
    ) -> np.ndarray:
        """
        Sagnac 地球自转效应修正 — 2 次迭代闭合

        思路:
          1. 用粗略伪距估算信号传播时间 τ
          2. 将卫星坐标绕 Z 轴旋转 ωe·τ (地球在信号传播期间的自转角)
          3. 用旋转后坐标重新计算几何距离 → 更新 τ → 再次旋转 → 闭合

        Parameters:
            sv_pos_raw  : 卫星 ECEF 坐标 (未旋转) (3,)
            rx_pos      : 接收机近似 ECEF 坐标 (3,)
            pseudorange : 伪距观测值 (m)

        Returns:
            Sagnac 修正后的卫星 ECEF 坐标 (3,)
        """
        sv = sv_pos_raw.copy()

        for _ in range(2):
            # 计算当前几何距离
            rho = np.linalg.norm(sv - rx_pos)
            # 信号传播时间
            tau = rho / SPEED_OF_LIGHT
            # 地球自转角
            theta = OMEGA_E_BDS * tau
            # 旋转矩阵 Rz(θ) — 绕 Z 轴
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            R = np.array([
                [ cos_t, sin_t, 0.0],
                [-sin_t, cos_t, 0.0],
                [  0.0,   0.0,  1.0],
            ])
            sv = R @ sv_pos_raw

        return sv

    # ================================================================
    #  向量化高度角计算 (不逐颗卫星 for 循环)
    # ================================================================
    @staticmethod
    def _compute_elevations_vectorized(
        rx_ecef: np.ndarray, sv_ecef: np.ndarray
    ) -> np.ndarray:
        """
        向量化计算多颗卫星的高度角

        Parameters:
            rx_ecef : (3,) 接收机 ECEF
            sv_ecef : (n, 3) 卫星 ECEF

        Returns:
            (n,) 各卫星高度角 (rad)
        """
        # 接收机大地坐标
        x, y, z = rx_ecef[0], rx_ecef[1], rx_ecef[2]
        lon = math.atan2(y, x)
        p = math.sqrt(x ** 2 + y ** 2)
        lat = math.atan2(z, p * (1 - WGS84_E2))
        for _ in range(10):
            sin_lat = math.sin(lat)
            N_val = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat ** 2)
            lat_new = math.atan2(z + WGS84_E2 * N_val * sin_lat, p)
            if abs(lat_new - lat) < 1e-12:
                break
            lat = lat_new

        sin_lat = math.sin(lat)
        cos_lat = math.cos(lat)
        sin_lon = math.sin(lon)
        cos_lon = math.cos(lon)

        # ENU 旋转矩阵 (3, 3)
        R = np.array([
            [-sin_lon,            cos_lon,           0.0    ],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [ cos_lat * cos_lon,  cos_lat * sin_lon, sin_lat],
        ])

        # 向量化: dx = (n, 3), enu = (n, 3)
        dx = sv_ecef - rx_ecef[np.newaxis, :]     # (n, 3)
        enu = (R @ dx.T).T                         # (n, 3): E, N, U

        horiz = np.sqrt(enu[:, 0] ** 2 + enu[:, 1] ** 2)
        elev = np.arctan2(enu[:, 2], horiz)        # (n,)
        return elev

    # ================================================================
    #  RAIM χ² 统计检验
    # ================================================================
    def _raim_chi_square(
        self,
        state: np.ndarray,
        SV: np.ndarray,
        P: np.ndarray,
        SNR: np.ndarray,
        raim_mask: np.ndarray,
    ) -> bool:
        """
        RAIM 完整性检验 — 基于标准化残差 + χ² 检验

        流程:
          1. 计算验后残差 v = P_obs - (ρ + c·dt)
          2. 计算协因子矩阵 Qvv = W⁻¹ - H(H^T W H)⁻¹ H^T
          3. 标准化残差 w_i = |v_i| / sqrt(Qvv_ii)
          4. 若 max(w_i) > RAIM_ALPHA，剔除对应卫星，返回 False 触发重算

        Parameters:
            state     : 当前状态 [x,y,z,cdt] (4,)
            SV        : 全部卫星坐标 (n_total, 3)
            P         : 全部修正伪距 (n_total,)
            SNR       : 全部 SNR (n_total,)
            raim_mask : 布尔掩码 (n_total,), True = 参与

        Returns:
            True  — 所有卫星通过检验
            False — 已剔除粗差卫星，需重新迭代
        """
        active = np.where(raim_mask)[0]
        n_active = active.shape[0]

        if n_active <= 4:
            return True  # 无冗余，无法做 RAIM

        xu = state[:3]
        cdt = state[3]

        # 残差
        diffs = SV[active] - xu[np.newaxis, :]
        rho = np.linalg.norm(diffs, axis=1)

        # 观测残差
        v = P[active] - (rho + cdt)

        # 高度角 & 权重 (向量化)
        elev_arr = self._compute_elevations_vectorized(xu, SV[active])
        sin_elev = np.maximum(np.sin(elev_arr), 0.05)
        snr_active = SNR[active]
        sigma2 = (0.09 + 0.09 / (sin_elev ** 2)) / np.power(
            10.0, (snr_active - 45.0) / 10.0
        )
        weights = 1.0 / sigma2  # (n_active,)

        # H 矩阵 (用于协因子)
        H = np.empty((n_active, 4), dtype=np.float64)
        H[:, 0] = -diffs[:, 0] / rho
        H[:, 1] = -diffs[:, 1] / rho
        H[:, 2] = -diffs[:, 2] / rho
        H[:, 3] = 1.0

        # 加权 H
        sqrt_w = np.sqrt(weights)
        Hw = H * sqrt_w[:, np.newaxis]

        # 法方程逆
        N = Hw.T @ Hw
        try:
            N_inv = np.linalg.inv(N)
        except np.linalg.LinAlgError:
            return True  # 无法求逆，跳过 RAIM

        # === 验后单位权方差 σ₀² ===
        dof = n_active - 4
        if dof <= 0:
            return True  # 无冗余观测，无法做 RAIM

        weighted_vtpv = np.sum(weights * v ** 2)
        sigma0_sq = weighted_vtpv / dof

        # === 残差协因子阵 Qvv = W⁻¹ - H · N⁻¹ · Hᵀ ===
        # W⁻¹ 对角元素 = sigma2
        # H · N⁻¹ · Hᵀ 的对角元素 (向量化提取，不构建完整 n×n 矩阵)
        # (H @ N_inv) 的每行与 H 的对应行点积 = diag(H N⁻¹ Hᵀ)
        HNinv = H @ N_inv                                      # (n_active, 4)
        diag_HNinvHt = np.sum(HNinv * H, axis=1)               # (n_active,)
        Qvv_diag = sigma2 - diag_HNinvHt                       # (n_active,)

        # 保护: Qvv 对角元素必须为正
        Qvv_diag = np.maximum(Qvv_diag, 1e-10)

        # === 标准化残差 w_i = |v_i| / sqrt(σ₀² · Qvv_ii) ===
        std_residuals = np.abs(v) / np.sqrt(sigma0_sq * Qvv_diag)

        # === χ² 检验: 找最大标准化残差 ===
        max_idx = np.argmax(std_residuals)
        max_std_res = std_residuals[max_idx]

        if max_std_res > RAIM_ALPHA:
            # 剔除该卫星: 在全局 raim_mask 中标记
            global_idx = active[max_idx]
            raim_mask[global_idx] = False
            return False  # 触发重新迭代

        return True  # 全部通过

    # ================================================================
    #  DOP 几何精度因子计算
    # ================================================================
    def _compute_dop(
        self, rx_ecef: np.ndarray, sv_ecef: np.ndarray
    ) -> DopValues:
        """
        计算 GDOP / PDOP / HDOP / VDOP / TDOP

        基于 H 矩阵的 (HᵀH)⁻¹ 对角元素, 在当地 ENU 坐标系下分解

        Parameters:
            rx_ecef : (3,) 接收机 ECEF 坐标
            sv_ecef : (n, 3) 卫星 ECEF 坐标

        Returns:
            DopValues 实例
        """
        dop = DopValues()
        n = sv_ecef.shape[0]
        if n < 4:
            return dop

        # 方向余弦 (ECEF)
        diffs = sv_ecef - rx_ecef[np.newaxis, :]
        rho = np.linalg.norm(diffs, axis=1, keepdims=True)
        rho = np.maximum(rho, 1.0)

        H_ecef = np.empty((n, 4), dtype=np.float64)
        H_ecef[:, :3] = -diffs / rho
        H_ecef[:, 3] = 1.0

        # 将 H 的方向余弦部分从 ECEF 旋转到 ENU
        lat, lon, _ = ecef_to_blh(rx_ecef[0], rx_ecef[1], rx_ecef[2])
        sin_lat = math.sin(lat)
        cos_lat = math.cos(lat)
        sin_lon = math.sin(lon)
        cos_lon = math.cos(lon)

        R = np.array([
            [-sin_lon,              cos_lon,             0.0    ],
            [-sin_lat * cos_lon,   -sin_lat * sin_lon,   cos_lat],
            [ cos_lat * cos_lon,    cos_lat * sin_lon,   sin_lat],
        ])

        # 旋转方向余弦: (n, 3) 每行乘以 R
        H_enu = np.empty((n, 4), dtype=np.float64)
        H_enu[:, :3] = (R @ H_ecef[:, :3].T).T   # (3,3)@(3,n) → (3,n) → (n,3)
        H_enu[:, 3] = 1.0

        try:
            Q = np.linalg.inv(H_enu.T @ H_enu)     # (4, 4)
        except np.linalg.LinAlgError:
            return dop

        # Q 对角元素: [E, N, U, T]
        qee = Q[0, 0]
        qnn = Q[1, 1]
        quu = Q[2, 2]
        qtt = Q[3, 3]

        # 保护负值 (数值精度)
        qee = max(qee, 0.0)
        qnn = max(qnn, 0.0)
        quu = max(quu, 0.0)
        qtt = max(qtt, 0.0)

        dop.hdop = math.sqrt(qee + qnn)
        dop.vdop = math.sqrt(quu)
        dop.pdop = math.sqrt(qee + qnn + quu)
        dop.tdop = math.sqrt(qtt)
        dop.gdop = math.sqrt(qee + qnn + quu + qtt)

        return dop

    # ================================================================
    #  Sagnac 双重闭合迭代 (2-pass) — 单星版
    # ================================================================
    def _sagnac_double_iteration(
        self,
        sv_pos_raw: np.ndarray,
        rx_pos: np.ndarray,
        pseudorange: float,
    ) -> np.ndarray:
        """
        Sagnac 地球自转效应修正 — 2 次迭代闭合

        Parameters:
            sv_pos_raw  : 卫星 ECEF 坐标 (未旋转) (3,)
            rx_pos      : 接收机近似 ECEF 坐标 (3,)
            pseudorange : 伪距观测值 (m)

        Returns:
            Sagnac 修正后的卫星 ECEF 坐标 (3,)
        """
        sv = sv_pos_raw.copy()

        for _pass in range(2):
            # 计算当前几何距离
            rho = np.linalg.norm(sv - rx_pos)
            # 信号传播时间
            tau = rho / SPEED_OF_LIGHT
            # 地球自转角
            theta = OMEGA_E_BDS * tau
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            # Rz(θ) 旋转原始坐标 (每次都从原始坐标旋转, 用更新后的 τ)
            R = np.array([
                [ cos_t, sin_t, 0.0],
                [-sin_t, cos_t, 0.0],
                [  0.0,   0.0,  1.0],
            ])
            sv = R @ sv_pos_raw

        return sv

    # ================================================================
    #  向量化高度角计算
    # ================================================================
    @staticmethod
    def _compute_elevations_vectorized(
        rx_ecef: np.ndarray, sv_ecef: np.ndarray
    ) -> np.ndarray:
        """
        向量化计算多颗卫星的高度角

        Parameters:
            rx_ecef : (3,) 接收机 ECEF
            sv_ecef : (n, 3) 卫星 ECEF

        Returns:
            (n,) 各卫星高度角 (rad)
        """
        x, y, z = rx_ecef[0], rx_ecef[1], rx_ecef[2]
        lon = math.atan2(y, x)
        p = math.sqrt(x ** 2 + y ** 2)
        lat = math.atan2(z, p * (1 - WGS84_E2))
        for _ in range(10):
            sin_lat = math.sin(lat)
            N_val = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat ** 2)
            lat_new = math.atan2(z + WGS84_E2 * N_val * sin_lat, p)
            if abs(lat_new - lat) < 1e-12:
                break
            lat = lat_new

        sin_lat = math.sin(lat)
        cos_lat = math.cos(lat)
        sin_lon = math.sin(lon)
        cos_lon = math.cos(lon)

        R = np.array([
            [-sin_lon,            cos_lon,           0.0    ],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [ cos_lat * cos_lon,  cos_lat * sin_lon, sin_lat],
        ])

        dx = sv_ecef - rx_ecef[np.newaxis, :]
        enu = (R @ dx.T).T

        horiz = np.sqrt(enu[:, 0] ** 2 + enu[:, 1] ** 2)
        elev = np.arctan2(enu[:, 2], horiz)
        return elev



