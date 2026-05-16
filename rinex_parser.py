"""
rinex_parser.py — RINEX 观测文件 / 导航文件 严密预处理引擎
=============================================================
核心职责:
  1. 解析 RINEX 3.x 北斗导航文件 (.nav)，动态提取电离层 α/β 参数与全部广播星历
  2. 解析 RINEX 3.x 观测文件 (.obs)，提取伪距、SNR、按历元组织数据
  3. 星历 TOE 最近邻匹配、不健康卫星剔除、先验坐标高度角预筛

设计红线:
  - Klobuchar α/β 参数 100% 从 .nav 头部正则动态提取，零硬编码
  - 星历匹配严格按 |t - toe| 最小判定
  - 高度角预筛使用 APPROX POSITION XYZ 先验坐标
"""

import re
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# ============================================================
#  常量定义
# ============================================================
SPEED_OF_LIGHT = 299792458.0          # m/s
GM_BDS = 3.986004418e14               # BDS 地心引力常数 m³/s²
OMEGA_E_BDS = 7.2921150e-5            # BDS 地球自转角速度 rad/s
PI = 3.1415926535898                   # RINEX / ICD 规范用 π
BDS_LEAP_SECONDS = 14                 # BDT 与 UTC 的跳秒差（BDT = UTC + 14 - 较 GPS 少 4s）
BDST_GPS_OFFSET = 14                  # BDT 相对 GPS 时间偏移(s)
WGS84_A = 6378137.0                   # WGS84 长半轴
WGS84_F = 1.0 / 298.257223563        # WGS84 扁率
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2  # 第一偏心率平方

# BDS 周起始参考: 2006-01-01 00:00:00 UTC
_BDS_EPOCH = datetime(2006, 1, 1, 0, 0, 0)


# ============================================================
#  工具函数
# ============================================================
def gps_week_seconds_from_datetime(dt: datetime) -> Tuple[int, float]:
    """将 datetime 转为 BDS 周 + 周内秒 (BDT 时间系统)"""
    delta = dt - _BDS_EPOCH
    total_seconds = delta.total_seconds()
    week = int(total_seconds // 604800)
    sow = total_seconds - week * 604800
    return week, sow


def ecef_to_blh(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """ECEF → WGS84 大地坐标 (lat_rad, lon_rad, h_m)，迭代法"""
    lon = math.atan2(y, x)
    p = math.sqrt(x ** 2 + y ** 2)
    lat = math.atan2(z, p * (1 - WGS84_E2))
    for _ in range(10):
        sin_lat = math.sin(lat)
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat ** 2)
        lat_new = math.atan2(z + WGS84_E2 * N * sin_lat, p)
        if abs(lat_new - lat) < 1e-12:
            break
        lat = lat_new
    h = p / math.cos(lat) - N
    return lat, lon, h


def compute_elevation_azimuth(
    rx_ecef: np.ndarray, sv_ecef: np.ndarray
) -> Tuple[float, float]:
    """
    计算从接收机到卫星的高度角(rad)和方位角(rad)
    rx_ecef: (3,)  接收机 ECEF 坐标
    sv_ecef: (3,)  卫星 ECEF 坐标
    """
    lat, lon, _ = ecef_to_blh(rx_ecef[0], rx_ecef[1], rx_ecef[2])
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    # ENU 旋转矩阵
    R = np.array([
        [-sin_lon,              cos_lon,             0       ],
        [-sin_lat * cos_lon,   -sin_lat * sin_lon,   cos_lat],
        [ cos_lat * cos_lon,    cos_lat * sin_lon,   sin_lat],
    ])

    dx = sv_ecef - rx_ecef
    enu = R @ dx
    e_val, n_val, u_val = enu[0], enu[1], enu[2]
    horiz = math.sqrt(e_val ** 2 + n_val ** 2)
    elev = math.atan2(u_val, horiz)
    az = math.atan2(e_val, n_val)
    return elev, az


# ============================================================
#  星历数据结构
# ============================================================
class BdsEphemeris:
    """单颗 BDS 卫星单历元广播星历参数集"""
    __slots__ = [
        'prn', 'toc', 'toe', 'week',
        'af0', 'af1', 'af2',
        'iode', 'crs', 'delta_n', 'm0',
        'cuc', 'ecc', 'cus', 'sqrt_a',
        'toe_sow', 'cic', 'omega0', 'cis',
        'i0', 'crc', 'omega', 'omega_dot',
        'idot', 'spare1', 'bds_week', 'spare2',
        'sv_accuracy', 'sv_health', 'tgd1', 'tgd2',
        'ttom', 'aodc', 'iodc',
    ]

    def __init__(self):
        for attr in self.__slots__:
            setattr(self, attr, 0.0)
        self.prn = ''
        self.toc = None  # datetime


# ============================================================
#  观测数据结构
# ============================================================
class ObsEpoch:
    """单历元观测数据"""
    __slots__ = ['epoch', 'satellites']

    def __init__(self, epoch: datetime):
        self.epoch = epoch
        self.satellites: Dict[str, dict] = {}
        # satellites[prn] = {'C2I': float, 'C6I': float, 'S2I': float, ...}


# ============================================================
#  RINEX Nav Parser
# ============================================================
class RinexNavParser:
    """
    RINEX 3.x 北斗导航文件解析器
    --------------------------------
    - 从头部正则提取 BDSA / BDSB 电离层参数 (多组取首次出现)
    - 逐块解析卫星星历，存入 ephemeris_pool: Dict[prn, List[BdsEphemeris]]
    """

    def __init__(self):
        self.iono_alpha: np.ndarray = np.zeros(4)
        self.iono_beta: np.ndarray = np.zeros(4)
        self.iono_params_found = False
        self.ephemeris_pool: Dict[str, List[BdsEphemeris]] = {}
        self.leap_seconds: int = BDS_LEAP_SECONDS

    # --------------------------------------------------------
    #  公共接口
    # --------------------------------------------------------
    def parse(self, nav_path: str) -> None:
        """解析整个 .nav 文件"""
        with open(nav_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        header_end = 0
        for idx, line in enumerate(lines):
            if 'END OF HEADER' in line:
                header_end = idx
                break

        self._parse_header(lines[:header_end + 1])
        self._parse_body(lines[header_end + 1:])

        n_sv = len(self.ephemeris_pool)
        n_eph = sum(len(v) for v in self.ephemeris_pool.values())
        print(f"[NavParser] 解析完成: {n_sv} 颗卫星, {n_eph} 条星历记录")
        print(f"[NavParser] 电离层 α = {self.iono_alpha}")
        print(f"[NavParser] 电离层 β = {self.iono_beta}")

    def select_ephemeris(self, prn: str, obs_time: datetime) -> Optional[BdsEphemeris]:
        """
        TOE 最近邻匹配 — 选取与观测时间 |t - toe| 最小的星历
        同时剔除不健康卫星 (sv_health != 0)
        """
        if prn not in self.ephemeris_pool:
            return None

        _, obs_sow = gps_week_seconds_from_datetime(obs_time)
        best_eph = None
        best_dt = float('inf')

        for eph in self.ephemeris_pool[prn]:
            # 健康检查
            if eph.sv_health != 0:
                continue
            dt = abs(obs_sow - eph.toe_sow)
            # 处理周跨越
            if dt > 302400:
                dt = 604800 - dt
            if dt < best_dt:
                best_dt = dt
                best_eph = eph

        # 超过 2 小时 (7200s) 的星历视为过期
        if best_eph is not None and best_dt > 7200:
            return None

        return best_eph

    # --------------------------------------------------------
    #  头部解析 — 动态正则提取 BDSA / BDSB
    # --------------------------------------------------------
    def _parse_header(self, header_lines: List[str]) -> None:
        """从 NAV 头部正则匹配 BDSA / BDSB 电离层修正参数"""
        # 正则: 匹配 BDSA/BDSB 后跟 4 个科学计数法数值
        # 示例: BDSA   3.3528E-08  7.4506E-09 -7.7486E-07  1.3709E-06 a C02 IONOSPHERIC CORR
        sci_num = r'([+-]?\d+\.\d+[EeDd][+-]?\d+)'
        pat_alpha = re.compile(
            r'BDSA\s+' + r'\s+'.join([sci_num] * 4), re.IGNORECASE
        )
        pat_beta = re.compile(
            r'BDSB\s+' + r'\s+'.join([sci_num] * 4), re.IGNORECASE
        )

        alpha_found = False
        beta_found = False

        for line in header_lines:
            if not alpha_found:
                m = pat_alpha.search(line)
                if m:
                    self.iono_alpha = np.array(
                        [float(m.group(i).replace('D', 'E').replace('d', 'e'))
                         for i in range(1, 5)]
                    )
                    alpha_found = True

            if not beta_found:
                m = pat_beta.search(line)
                if m:
                    self.iono_beta = np.array(
                        [float(m.group(i).replace('D', 'E').replace('d', 'e'))
                         for i in range(1, 5)]
                    )
                    beta_found = True

            if alpha_found and beta_found:
                break

        self.iono_params_found = alpha_found and beta_found
        if not self.iono_params_found:
            print("[NavParser] ⚠ 警告: 未在 NAV 头部找到完整的 BDSA/BDSB 电离层参数!")

    # --------------------------------------------------------
    #  数据体解析 — 逐块提取星历
    # --------------------------------------------------------
    def _parse_body(self, body_lines: List[str]) -> None:
        """
        解析 NAV 数据体, 每条星历占 8 行 (BDS 标准):
          Line 0: PRN  年 月 日 时 分 秒  af0  af1  af2
          Line 1: IODE  Crs  Δn  M0
          Line 2: Cuc  e  Cus  √a
          Line 3: toe  Cic  Ω0  Cis
          Line 4: i0  Crc  ω  Ω̇
          Line 5: IDOT  spare  BDS_week  spare
          Line 6: SV_accuracy  SV_health  TGD1  TGD2
          Line 7: t_tm  AODC (IODC)  ...
        """
        # 预处理: 将 'D'/'d' 指数替换为 'E' 以便 float() 解析
        cleaned = []
        for line in body_lines:
            cleaned.append(line.replace('D', 'E').replace('d', 'e').rstrip('\n'))

        idx = 0
        total = len(cleaned)
        while idx < total:
            line0 = cleaned[idx]
            # 跳过空行
            if len(line0.strip()) == 0:
                idx += 1
                continue

            # 检测星历首行: 以 C/c 开头 + 数字 (PRN)
            m0 = re.match(r'^(C\d{2})\s+(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+'
                          r'(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})', line0, re.IGNORECASE)
            if not m0:
                idx += 1
                continue

            # 确保后续 7 行存在
            if idx + 7 >= total:
                break

            eph = BdsEphemeris()
            eph.prn = m0.group(1).upper()

            # 解析时钟参考时间 toc
            yr = int(m0.group(2))
            mo = int(m0.group(3))
            dy = int(m0.group(4))
            hr = int(m0.group(5))
            mi = int(m0.group(6))
            sc = int(m0.group(7))
            eph.toc = datetime(yr, mo, dy, hr, mi, sc)

            # 提取 line0 末尾 3 个浮点数: af0, af1, af2
            vals0 = self._extract_floats(line0, start_col=23, count=3)
            eph.af0, eph.af1, eph.af2 = vals0

            # Line 1: IODE, Crs, Δn, M0
            vals1 = self._extract_floats(cleaned[idx + 1], start_col=4, count=4)
            eph.iode, eph.crs, eph.delta_n, eph.m0 = vals1

            # Line 2: Cuc, e, Cus, √a
            vals2 = self._extract_floats(cleaned[idx + 2], start_col=4, count=4)
            eph.cuc, eph.ecc, eph.cus, eph.sqrt_a = vals2

            # Line 3: toe, Cic, Ω0, Cis
            vals3 = self._extract_floats(cleaned[idx + 3], start_col=4, count=4)
            eph.toe_sow, eph.cic, eph.omega0, eph.cis = vals3

            # Line 4: i0, Crc, ω, Ω̇
            vals4 = self._extract_floats(cleaned[idx + 4], start_col=4, count=4)
            eph.i0, eph.crc, eph.omega, eph.omega_dot = vals4

            # Line 5: IDOT, spare1, BDS_week, spare2
            vals5 = self._extract_floats(cleaned[idx + 5], start_col=4, count=4)
            eph.idot = vals5[0]
            eph.spare1 = vals5[1]
            eph.bds_week = vals5[2]
            eph.spare2 = vals5[3]

            # Line 6: SV_accuracy, SV_health, TGD1, TGD2
            vals6 = self._extract_floats(cleaned[idx + 6], start_col=4, count=4)
            eph.sv_accuracy = vals6[0]
            eph.sv_health = vals6[1]
            eph.tgd1 = vals6[2]
            eph.tgd2 = vals6[3]

            # Line 7: t_tm, AODC/IODC (可能不足 4 个值)
            vals7 = self._extract_floats(cleaned[idx + 7], start_col=4, count=2)
            eph.ttom = vals7[0] if len(vals7) > 0 else 0.0
            eph.aodc = vals7[1] if len(vals7) > 1 else 0.0

            # 推算 toe 的 BDS 周
            eph.week = int(eph.bds_week)
            eph.toe = eph.toe_sow  # 保留周内秒

            # 存入池
            if eph.prn not in self.ephemeris_pool:
                self.ephemeris_pool[eph.prn] = []
            self.ephemeris_pool[eph.prn].append(eph)

            idx += 8

    @staticmethod
    def _extract_floats(line: str, start_col: int = 4, count: int = 4) -> List[float]:
        """
        从固定列宽行中提取浮点数
        RINEX 3.x 导航数据每字段 19 字符宽
        """
        values = []
        for i in range(count):
            col_start = start_col + i * 19
            col_end = col_start + 19
            if col_start >= len(line):
                break
            token = line[col_start:col_end].strip()
            if not token:
                break
            try:
                val = float(token)
                if math.isnan(val) or math.isinf(val):
                    values.append(0.0)
                else:
                    values.append(val)
            except (ValueError, OverflowError):
                # 尝试更宽容的正则提取
                m = re.search(r'[+-]?\d+\.?\d*[Ee][+-]?\d+', token)
                if m:
                    values.append(float(m.group()))
                else:
                    values.append(0.0)
        return values


# ============================================================
#  RINEX Obs Parser
# ============================================================
class RinexObsParser:
    """
    RINEX 3.x 北斗观测文件解析器
    --------------------------------
    - 解析头部: APPROX POSITION XYZ, 观测类型
    - 逐历元提取 BDS 卫星伪距 (C2I/C6I) 及信噪比 (S2I/S6I)
    """

    def __init__(self):
        self.approx_pos: np.ndarray = np.zeros(3)  # ECEF 先验坐标
        self.obs_types_bds: List[str] = []           # BDS 观测类型列表
        self.epochs: List[ObsEpoch] = []
        self.header_lines: List[str] = []
        self.antenna_delta: np.ndarray = np.zeros(3)

    def parse(self, obs_path: str) -> None:
        """解析整个 .obs 文件"""
        with open(obs_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        header_end = 0
        for idx, line in enumerate(lines):
            if 'END OF HEADER' in line:
                header_end = idx
                break

        self._parse_header(lines[:header_end + 1])
        self._parse_data(lines[header_end + 1:])

        n_epochs = len(self.epochs)
        print(f"[ObsParser] 解析完成: {n_epochs} 个历元")
        print(f"[ObsParser] 先验坐标 ECEF = {self.approx_pos}")
        print(f"[ObsParser] BDS 观测类型 = {self.obs_types_bds}")

    # --------------------------------------------------------
    #  头部解析
    # --------------------------------------------------------
    def _parse_header(self, header_lines: List[str]) -> None:
        """解析 OBS 文件头部"""
        self.header_lines = header_lines
        idx = 0
        while idx < len(header_lines):
            line = header_lines[idx]

            # 先验坐标
            if 'APPROX POSITION XYZ' in line:
                parts = line[:60].split()
                if len(parts) >= 3:
                    self.approx_pos = np.array([float(parts[0]),
                                                float(parts[1]),
                                                float(parts[2])])

            # 天线偏移
            if 'ANTENNA: DELTA H/E/N' in line:
                parts = line[:60].split()
                if len(parts) >= 3:
                    self.antenna_delta = np.array([float(parts[0]),
                                                   float(parts[1]),
                                                   float(parts[2])])

            # 观测类型 (SYS / # / OBS TYPES)
            if 'SYS / # / OBS TYPES' in line:
                sys_char = line[0].strip()
                if sys_char == 'C':  # BDS
                    n_types = int(line[3:6].strip())
                    type_str = line[7:60]
                    # 可能需要续行
                    types = type_str.split()
                    while len(types) < n_types:
                        idx += 1
                        if idx >= len(header_lines):
                            break
                        cont_line = header_lines[idx]
                        types.extend(cont_line[7:60].split())
                    self.obs_types_bds = types[:n_types]

            idx += 1

    # --------------------------------------------------------
    #  数据体解析
    # --------------------------------------------------------
    def _parse_data(self, data_lines: List[str]) -> None:
        """逐历元解析观测数据"""
        idx = 0
        total = len(data_lines)

        while idx < total:
            line = data_lines[idx].rstrip('\n')

            # 历元头标识: > YYYY MM DD HH MM SS.SSSSSSS  0  n_sv
            if not line.startswith('>'):
                idx += 1
                continue

            try:
                yr = int(line[2:6])
                mo = int(line[7:9])
                dy = int(line[10:12])
                hr = int(line[13:15])
                mi = int(line[16:18])
                sec_str = line[19:29].strip()
                sec = float(sec_str)
                sec_int = int(sec)
                microsec = int(round((sec - sec_int) * 1e6))
                epoch_dt = datetime(yr, mo, dy, hr, mi, sec_int, microsec)

                # 历元标志与卫星数
                epoch_flag = int(line[29:32].strip()) if line[29:32].strip() else 0
                n_sv = int(line[32:35].strip()) if line[32:35].strip() else 0
            except (ValueError, IndexError):
                idx += 1
                continue

            epoch = ObsEpoch(epoch_dt)

            # 读取 n_sv 行卫星数据
            for _ in range(n_sv):
                idx += 1
                if idx >= total:
                    break
                sv_line = data_lines[idx].rstrip('\n')
                if len(sv_line) < 3:
                    continue

                prn = sv_line[0:3].strip()
                # 仅保留 BDS 卫星
                if not prn.startswith('C'):
                    continue

                # 解析各观测值 (每个观测值占 16 字符，从第 3 列开始)
                obs_dict = {}
                for k, obs_type in enumerate(self.obs_types_bds):
                    col_start = 3 + k * 16
                    col_end = col_start + 14
                    if col_start >= len(sv_line):
                        break
                    val_str = sv_line[col_start:col_end].strip()
                    if val_str:
                        try:
                            obs_dict[obs_type] = float(val_str)
                        except ValueError:
                            pass

                # 至少要有一个伪距观测值
                if any(key.startswith('C') for key in obs_dict):
                    epoch.satellites[prn] = obs_dict

            if epoch.satellites:
                self.epochs.append(epoch)

            idx += 1

        # 按时间排序
        self.epochs.sort(key=lambda ep: ep.epoch)

    # --------------------------------------------------------
    #  高度角预筛 — 在进入 LS 前剔除低仰角卫星
    # --------------------------------------------------------
    def filter_by_elevation(
        self,
        epoch: ObsEpoch,
        sat_positions: Dict[str, np.ndarray],
        rx_approx: np.ndarray,
        mask_deg: float = 10.0,
    ) -> Dict[str, dict]:
        """
        基于先验坐标进行高度角预筛
        Parameters:
            epoch       : 当前历元观测数据
            sat_positions: {prn: np.array([x,y,z])} 卫星 ECEF 坐标
            rx_approx   : 接收机先验 ECEF 坐标 (3,)
            mask_deg    : 高度角截止角(度)
        Returns:
            过滤后的 {prn: obs_dict}
        """
        mask_rad = math.radians(mask_deg)
        filtered = {}
        for prn, obs in epoch.satellites.items():
            if prn not in sat_positions:
                continue
            elev, _ = compute_elevation_azimuth(rx_approx, sat_positions[prn])
            if elev >= mask_rad:
                filtered[prn] = obs
        return filtered