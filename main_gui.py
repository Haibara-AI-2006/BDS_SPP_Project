"""
main_gui.py — PyQt5 可视化主调度界面
======================================
已上传至github,若有一致请查看实验报告是否提到链接
核心职责:
  1. RINEX 数据导入
  2. 解算参数配置
  3. SppWorker(QThread) 后台解算
  4. pyqtSignal(dict) 高频回传解算状态
  5. 实时定位轨迹显示 & 天空图
  6. AI 补偿 & 精度对比
  7. OOM 防御: deque 缓存 + set_data 原地更新
"""

import os
import sys
import math
import numpy as np
from collections import deque
from datetime import datetime
from typing import Optional, List

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QLabel, QLineEdit, QFileDialog,
    QProgressBar, QTextEdit, QSplitter, QTabWidget, QSpinBox,
    QDoubleSpinBox, QFormLayout, QMessageBox, QStatusBar,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QColor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle

# ============================================================
#  导入项目模块 (必须先导入 rinex_parser 以获取 WGS84 常量)
# ============================================================
from rinex_parser import (
    RinexNavParser, RinexObsParser,
    WGS84_A, WGS84_E2,
)
from sat_pos_calculator import SatPosCalculator
from spp_solver import SppSolver, EpochSolution
from trajectory_analyzer import TrajectoryAnalyzer
from ai_compensator import AiCompensator

# ============================================================
#  中文字体
# ============================================================
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
#  真值常量定义 (北邮沙河校区)
# ============================================================
GT_LAT_DEG = 40.1575
GT_LON_DEG = 116.2885
GT_HEIGHT = 35.0
GT_LAT_RAD = math.radians(GT_LAT_DEG)
GT_LON_RAD = math.radians(GT_LON_DEG)

# 计算 GT_ECEF
_sin_lat = math.sin(GT_LAT_RAD)
_cos_lat = math.cos(GT_LAT_RAD)
_sin_lon = math.sin(GT_LON_RAD)
_cos_lon = math.cos(GT_LON_RAD)
_N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * _sin_lat ** 2)

GT_ECEF = np.array([
    (_N + GT_HEIGHT) * _cos_lat * _cos_lon,
    (_N + GT_HEIGHT) * _cos_lat * _sin_lon,
    (_N * (1.0 - WGS84_E2) + GT_HEIGHT) * _sin_lat,
], dtype=np.float64)

# 计算 R_ENU
R_ENU = np.array([
    [-_sin_lon,              _cos_lon,             0.0     ],
    [-_sin_lat * _cos_lon,  -_sin_lat * _sin_lon,  _cos_lat],
    [ _cos_lat * _cos_lon,   _cos_lat * _sin_lon,  _sin_lat],
], dtype=np.float64)

print(f"[main_gui] GT_ECEF = {GT_ECEF}")

# ============================================================
#  缓冲区大小
# ============================================================
MAX_BUFFER = 2880  # 24h × 30s = 2880 个历元


# ============================================================
#  SppWorker — QThread 后台解算工作线程
# ============================================================
class SppWorker(QThread):
    """后台 SPP 解算工作线程"""

    progress_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        nav_path: str,
        obs_path: str,
        elev_mask: float = 5.0,
        max_iter: int = 20,
        parent=None,
    ):
        super().__init__(parent)
        self.nav_path = nav_path
        self.obs_path = obs_path
        self.elev_mask = elev_mask
        self.max_iter = max_iter
        self._is_cancelled = False

    def cancel(self):
        """请求取消解算"""
        self._is_cancelled = True

    def run(self):
        """工作线程主函数"""
        solutions = []  # 在最开始定义
        total = 0       # 在最开始定义

        try:
            # --- 解析 NAV ---
            nav_parser = RinexNavParser()
            nav_parser.parse(self.nav_path)

            # --- 解析 OBS ---
            obs_parser = RinexObsParser()
            obs_parser.parse(self.obs_path)

            # --- 构建计算器 ---
            sat_calc = SatPosCalculator(nav_parser)

            # --- 构建解算器 ---
            solver = SppSolver(
                nav_parser, obs_parser, sat_calc,
                elevation_mask=self.elev_mask,
            )

            # --- 真值 ECEF (使用全局常量) ---
            gt_ecef = GT_ECEF.copy()
            r_enu = R_ENU.copy()

            total = len(obs_parser.epochs)
            state = np.zeros(4)
            state[:3] = obs_parser.approx_pos.copy()

            # 逐历元解算
            for idx, epoch_obs in enumerate(obs_parser.epochs):
                if self._is_cancelled:
                    break

                sol = solver.solve_epoch(epoch_obs, state)
                solutions.append(sol)

                if sol.valid:
                    state[:3] = sol.pos_ecef
                    state[3] = sol.clock_bias
                    solver._last_valid_state = state.copy()
                    solver._has_last_valid = True

                # 构建进度信号字典
                sig = {
                    'epoch_idx': idx,
                    'total': total,
                    'valid': sol.valid,
                    'n_used': sol.n_used,
                    'n_excluded': 0,
                    'pdop': sol.dop.pdop if sol.valid else 99.9,
                    'enu': (0.0, 0.0, 0.0),
                    'skyplot': [],
                    'lat_deg': sol.lat_deg,
                    'lon_deg': sol.lon_deg,
                    'height': sol.height,
                    'epoch_str': epoch_obs.epoch.strftime('%H:%M:%S'),
                }

                if sol.valid:
                    dx = sol.pos_ecef - gt_ecef
                    enu = r_enu @ dx
                    sig['enu'] = (float(enu[0]), float(enu[1]), float(enu[2]))

                    sky_data = []
                    for k, prn in enumerate(sol.prn_list):
                        if k < len(sol.elevations) and k < len(sol.azimuths):
                            az_d = math.degrees(sol.azimuths[k])
                            el_d = math.degrees(sol.elevations[k])
                            sky_data.append((prn, az_d, el_d))
                    sig['skyplot'] = sky_data
                    sig['n_excluded'] = max(0, len(epoch_obs.satellites) - sol.n_used)

                self.progress_signal.emit(sig)

            self.finished_signal.emit(solutions)

        except Exception as exc:
            import traceback
            error_msg = traceback.format_exc()
            self.error_signal.emit(error_msg)


# ============================================================
#  实时绘图画布
# ============================================================
class RealtimeCanvas(FigureCanvas):
    """实时绘图画布 — set_data 原地更新, 防止 OOM"""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(12, 7), dpi=100, constrained_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)

        # 创建子图布局
        gs = self.fig.add_gridspec(3, 2, width_ratios=[2.5, 1])
        self.ax_e = self.fig.add_subplot(gs[0, 0])
        self.ax_n = self.fig.add_subplot(gs[1, 0])
        self.ax_u = self.fig.add_subplot(gs[2, 0])
        self.ax_sky = self.fig.add_subplot(gs[:, 1], projection='polar')

        # ENU 线对象
        self.line_e, = self.ax_e.plot([], [], color='#d62728', linewidth=0.8)
        self.line_n, = self.ax_n.plot([], [], color='#2ca02c', linewidth=0.8)
        self.line_u, = self.ax_u.plot([], [], color='#1f77b4', linewidth=0.8)

        for ax, label in [(self.ax_e, 'E(m)'), (self.ax_n, 'N(m)'), (self.ax_u, 'U(m)')]:
            ax.set_ylabel(label, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

        self.ax_u.set_xlabel('历元', fontsize=9)

        # 天空图
        self.ax_sky.set_theta_zero_location('N')
        self.ax_sky.set_theta_direction(-1)
        self.ax_sky.set_ylim(0, 90)
        self.ax_sky.set_yticks([0, 15, 30, 45, 60, 75, 90])
        self.ax_sky.set_yticklabels(['90°', '75°', '60°', '45°', '30°', '15°', '0°'], fontsize=7)
        self.ax_sky.set_title('卫星天空图', fontsize=10, pad=12)

        # 数据缓冲
        self.buf_idx = deque(maxlen=MAX_BUFFER)
        self.buf_e = deque(maxlen=MAX_BUFFER)
        self.buf_n = deque(maxlen=MAX_BUFFER)
        self.buf_u = deque(maxlen=MAX_BUFFER)

        self._sky_counter = 0

    def update_enu(self, epoch_idx: int, e: float, n: float, u: float):
        """追加 ENU 数据并原地更新"""
        self.buf_idx.append(epoch_idx)
        self.buf_e.append(e)
        self.buf_n.append(n)
        self.buf_u.append(u)

        idx_arr = list(self.buf_idx)
        self.line_e.set_data(idx_arr, list(self.buf_e))
        self.line_n.set_data(idx_arr, list(self.buf_n))
        self.line_u.set_data(idx_arr, list(self.buf_u))

        for ax, buf in [(self.ax_e, self.buf_e),
                         (self.ax_n, self.buf_n),
                         (self.ax_u, self.buf_u)]:
            if len(buf) > 0:
                ymin = min(buf) - 1
                ymax = max(buf) + 1
                ax.set_xlim(max(0, idx_arr[0]), idx_arr[-1] + 1)
                ax.set_ylim(ymin, ymax)

        self.draw_idle()

    def update_skyplot(self, skydata: list):
        """更新天空图 (每 5 个历元重绘一次)"""
        self._sky_counter += 1
        if self._sky_counter % 5 != 0:
            return

        self.ax_sky.clear()
        self.ax_sky.set_theta_zero_location('N')
        self.ax_sky.set_theta_direction(-1)
        self.ax_sky.set_ylim(0, 90)
        self.ax_sky.set_yticks([0, 15, 30, 45, 60, 75, 90])
        self.ax_sky.set_yticklabels(['90°', '75°', '60°', '45°', '30°', '15°', '0°'], fontsize=7)
        self.ax_sky.set_title('卫星天空图', fontsize=10, pad=12)

        if not skydata:
            self.draw_idle()
            return

        for prn, az_deg, elev_deg in skydata:
            az_rad = math.radians(az_deg)
            r = 90.0 - elev_deg

            prn_num = int(prn[1:])
            is_geo = (1 <= prn_num <= 5) or (59 <= prn_num <= 63)
            color = '#d62728' if is_geo else '#1f77b4'
            marker = 's' if is_geo else 'o'

            self.ax_sky.scatter(az_rad, r, c=color, s=50, marker=marker,
                                edgecolors='black', linewidth=0.5, zorder=5)
            self.ax_sky.annotate(prn, (az_rad, r), fontsize=7,
                                 ha='center', va='bottom',
                                 textcoords='offset points', xytext=(0, 5))

        self.draw_idle()

    def reset(self):
        """清空缓冲区"""
        self.buf_idx.clear()
        self.buf_e.clear()
        self.buf_n.clear()
        self.buf_u.clear()
        self.line_e.set_data([], [])
        self.line_n.set_data([], [])
        self.line_u.set_data([], [])
        self._sky_counter = 0
        self.ax_sky.clear()
        self.draw_idle()


# ============================================================
#  主窗口
# ============================================================
class MainWindow(QMainWindow):
    """北斗单点定位解算软件 — PyQt5 主界面"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("北斗 SPP 单点定位解算软件 v2.0 — 北京邮电大学")
        self.setMinimumSize(1400, 900)

        # 状态变量
        self.nav_path = ''
        self.obs_path = ''
        self.worker: Optional[SppWorker] = None
        self.solutions: List[EpochSolution] = []
        self.solutions_compensated: List[EpochSolution] = []

        self._build_ui()
        self._connect_signals()

        self.statusBar().showMessage("就绪 — 请加载 RINEX 数据文件")

    def _build_ui(self):
        """构建界面布局"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # 左侧控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(380)

        # 文件选择组
        file_group = QGroupBox("数据文件")
        file_layout = QFormLayout()

        self.nav_edit = QLineEdit()
        self.nav_edit.setReadOnly(True)
        self.nav_edit.setPlaceholderText("选择 .nav 文件...")
        self.btn_nav = QPushButton("浏览")
        nav_row = QHBoxLayout()
        nav_row.addWidget(self.nav_edit)
        nav_row.addWidget(self.btn_nav)
        file_layout.addRow("导航文件:", nav_row)

        self.obs_edit = QLineEdit()
        self.obs_edit.setReadOnly(True)
        self.obs_edit.setPlaceholderText("选择 .obs 文件...")
        self.btn_obs = QPushButton("浏览")
        obs_row = QHBoxLayout()
        obs_row.addWidget(self.obs_edit)
        obs_row.addWidget(self.btn_obs)
        file_layout.addRow("观测文件:", obs_row)

        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # 参数配置组
        param_group = QGroupBox("解算参数")
        param_layout = QFormLayout()

        self.spin_elev = QDoubleSpinBox()
        self.spin_elev.setRange(0, 30)
        self.spin_elev.setValue(5.0)
        self.spin_elev.setSuffix(" °")
        param_layout.addRow("高度角截止:", self.spin_elev)

        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(5, 30)
        self.spin_iter.setValue(20)
        param_layout.addRow("最大迭代数:", self.spin_iter)

        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)

        # 操作按钮组
        btn_group = QGroupBox("操作")
        btn_layout = QVBoxLayout()

        self.btn_start = QPushButton("▶  开始解算")
        self.btn_start.setStyleSheet(
            "QPushButton{background-color:#2ca02c;color:white;font-size:14px;"
            "padding:8px;border-radius:4px;}"
            "QPushButton:hover{background-color:#27ae60;}"
        )
        btn_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("⏹  停止解算")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(
            "QPushButton{background-color:#d62728;color:white;font-size:13px;"
            "padding:6px;border-radius:4px;}"
        )
        btn_layout.addWidget(self.btn_stop)

        self.btn_ai = QPushButton("🤖  AI 误差补偿")
        self.btn_ai.setEnabled(False)
        self.btn_ai.setStyleSheet(
            "QPushButton{background-color:#1f77b4;color:white;font-size:13px;"
            "padding:6px;border-radius:4px;}"
        )
        btn_layout.addWidget(self.btn_ai)

        self.btn_export = QPushButton("📊  导出报告与图表")
        self.btn_export.setEnabled(False)
        btn_layout.addWidget(self.btn_export)

        btn_group.setLayout(btn_layout)
        left_layout.addWidget(btn_group)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        left_layout.addWidget(self.progress_bar)

        # 实时状态标签
        status_group = QGroupBox("实时状态")
        status_layout = QFormLayout()

        self.lbl_epoch = QLabel("—")
        self.lbl_sats = QLabel("—")
        self.lbl_pdop = QLabel("—")
        self.lbl_pos = QLabel("—")
        self.lbl_enu = QLabel("—")

        status_layout.addRow("历元:", self.lbl_epoch)
        status_layout.addRow("卫星数:", self.lbl_sats)
        status_layout.addRow("PDOP:", self.lbl_pdop)
        status_layout.addRow("位置:", self.lbl_pos)
        status_layout.addRow("ENU误差:", self.lbl_enu)

        status_group.setLayout(status_layout)
        left_layout.addWidget(status_group)

        # 日志区
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setFont(QFont("Consolas", 9))
        left_layout.addWidget(QLabel("运行日志:"))
        left_layout.addWidget(self.log_text)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # 右侧绘图区 (Tab)
        self.tab_widget = QTabWidget()

        self.realtime_canvas = RealtimeCanvas()
        self.tab_widget.addTab(self.realtime_canvas, "实时监控")

        self.result_canvas = FigureCanvas(Figure(figsize=(12, 8)))
        self.tab_widget.addTab(self.result_canvas, "分析结果")

        self.ai_canvas = FigureCanvas(Figure(figsize=(12, 8)))
        self.tab_widget.addTab(self.ai_canvas, "AI 补偿")

        main_layout.addWidget(self.tab_widget, stretch=1)

    def _connect_signals(self):
        """连接信号槽"""
        self.btn_nav.clicked.connect(self._select_nav)
        self.btn_obs.clicked.connect(self._select_obs)
        self.btn_start.clicked.connect(self._start_solve)
        self.btn_stop.clicked.connect(self._stop_solve)
        self.btn_ai.clicked.connect(self._run_ai_compensation)
        self.btn_export.clicked.connect(self._export_results)

    def _select_nav(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择导航文件", "data", "NAV Files (*.nav);;All (*)")
        if path:
            self.nav_path = path
            self.nav_edit.setText(os.path.basename(path))
            self._log(f"已选择导航文件: {path}")

    def _select_obs(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择观测文件", "data", "OBS Files (*.obs);;All (*)")
        if path:
            self.obs_path = path
            self.obs_edit.setText(os.path.basename(path))
            self._log(f"已选择观测文件: {path}")

    def _start_solve(self):
        """启动后台解算"""
        if not self.nav_path or not self.obs_path:
            QMessageBox.warning(self, "提示", "请先选择导航文件和观测文件！")
            return

        self.realtime_canvas.reset()
        self.solutions.clear()
        self.solutions_compensated.clear()
        self.progress_bar.setValue(0)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_ai.setEnabled(False)
        self.btn_export.setEnabled(False)

        self._log("=" * 50)
        self._log("开始解算...")

        self.worker = SppWorker(
            nav_path=self.nav_path,
            obs_path=self.obs_path,
            elev_mask=self.spin_elev.value(),
            max_iter=self.spin_iter.value(),
        )

        self.worker.progress_signal.connect(self._on_progress)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.error_signal.connect(self._on_error)
        self.worker.start()

    def _stop_solve(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self._log("正在停止解算...")

    @pyqtSlot(dict)
    def _on_progress(self, sig: dict):
        """接收解算进度"""
        idx = sig['epoch_idx']
        total = sig['total']

        if total > 0:
            pct = int((idx + 1) / total * 100)
            self.progress_bar.setValue(pct)

        self.lbl_epoch.setText(f"{sig['epoch_str']}  ({idx + 1}/{total})")
        self.lbl_sats.setText(f"{sig['n_used']}  (剔除 {sig['n_excluded']})")
        self.lbl_pdop.setText(f"{sig['pdop']:.2f}")

        if sig['valid']:
            e, n, u = sig['enu']
            self.lbl_pos.setText(
                f"{sig['lat_deg']:.6f}°N, {sig['lon_deg']:.6f}°E, "
                f"H={sig['height']:.1f}m"
            )
            self.lbl_enu.setText(f"E={e:.2f}  N={n:.2f}  U={u:.2f}m")

            self.realtime_canvas.update_enu(idx, e, n, u)
            self.realtime_canvas.update_skyplot(sig['skyplot'])
        else:
            self.lbl_pos.setText("无效解")
            self.lbl_enu.setText("—")

    @pyqtSlot(list)
    def _on_finished(self, solutions: list):
        """解算完成"""
        self.solutions = solutions
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_ai.setEnabled(True)
        self.btn_export.setEnabled(True)

        n_valid = sum(1 for s in solutions if s.valid)
        self._log(f"解算完成! 总历元: {len(solutions)}, 有效: {n_valid}")
        self.statusBar().showMessage(f"解算完成 — {n_valid}/{len(solutions)} 有效历元")

        if n_valid > 0:
            self._render_result_tab()
        else:
            QMessageBox.warning(self, "警告", "解算完成但无有效历元！")

    @pyqtSlot(str)
    def _on_error(self, traceback_str: str):
        """解算异常"""
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._log(f"解算异常:\n{traceback_str}")
        QMessageBox.critical(self, "解算异常", traceback_str[:500])

    def _render_result_tab(self):
        """渲染分析结果 Tab"""
        if not self.solutions:
            self._log("警告: 无解算结果")
            return

        analyzer = TrajectoryAnalyzer(self.solutions, output_dir='data',
                                       scene_label='SPP解算')
        analyzer.compute_errors()

        if analyzer.df is None or len(analyzer.df) == 0:
            self._log("警告: 无有效历元数据")
            return

        report = analyzer.generate_report()

        self.result_canvas.figure.clear()
        gs = self.result_canvas.figure.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        ax1 = self.result_canvas.figure.add_subplot(gs[0, :])
        ax2 = self.result_canvas.figure.add_subplot(gs[1, 0])
        ax3 = self.result_canvas.figure.add_subplot(gs[1, 1])

        df = analyzer.df

        # ENU 时序
        ax1.plot(df['epoch'], df['err_e'], label='E', color='#d62728', linewidth=0.6, alpha=0.7)
        ax1.plot(df['epoch'], df['err_n'], label='N', color='#2ca02c', linewidth=0.6, alpha=0.7)
        ax1.plot(df['epoch'], df['err_u'], label='U', color='#1f77b4', linewidth=0.6, alpha=0.7)
        ax1.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.4)
        ax1.set_ylabel('误差 (m)', fontsize=9)
        ax1.set_title('ENU 误差时序', fontsize=10, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=8)

        # 水平散点
        e = df['err_e'].values
        n = df['err_n'].values
        r = np.sqrt(e ** 2 + n ** 2)
        cep95 = np.percentile(r, 95)

        ax2.scatter(e, n, c='#1f77b4', s=5, alpha=0.5)
        circle = Circle((0, 0), cep95, fill=False, color='red',
                       linewidth=2, linestyle='--',
                       label=f'95% CEP={cep95:.2f}m')
        ax2.add_patch(circle)
        ax2.axhline(0, color='gray', linewidth=0.5)
        ax2.axvline(0, color='gray', linewidth=0.5)
        ax2.set_xlabel('East (m)', fontsize=9)
        ax2.set_ylabel('North (m)', fontsize=9)
        ax2.set_title('水平误差散点', fontsize=10, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        ax2.tick_params(labelsize=8)

        # PDOP 时序
        ax3.plot(df['epoch'], df['pdop'], color='#ff7f0e', linewidth=0.8)
        ax3.fill_between(df['epoch'], df['pdop'], alpha=0.2, color='#ff7f0e')
        ax3.set_ylabel('PDOP', fontsize=9)
        ax3.set_title('PDOP 时序', fontsize=10, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=8)
        ax3.set_ylim(bottom=0)

        try:
            self.result_canvas.figure.tight_layout()
        except Exception:
            pass
        self.result_canvas.draw()

        self._log(f"分析完成: RMS_3D={report.get('rms_3d', 0):.3f}m, "
                  f"95%CEP={report.get('cep_95', 0):.3f}m")

    def _run_ai_compensation(self):
        """执行 AI 误差补偿"""
        if not self.solutions:
            QMessageBox.warning(self, "提示", "请先完成 SPP 解算！")
            return

        self._log("=" * 50)
        self._log("开始 AI 误差补偿训练...")

        try:
            compensator = AiCompensator()

            X = compensator.extract_features(self.solutions)
            Y = compensator.extract_labels(self.solutions)

            if len(X) < 10:
                QMessageBox.warning(self, "提示", "有效历元数不足，无法训练模型！")
                return

            metrics = compensator.train(
                X, Y,
                test_size=0.3,
                n_estimators=100,
                max_depth=20,
                verbose=False,
            )

            self._log(f"训练完成: RMSE_3D={metrics.get('rmse_3d', 0):.4f}m")

            self.solutions_compensated = compensator.compensate_solutions(self.solutions)

            model_path = os.path.join('data', 'bds_rf_model.pkl')
            compensator.save_model(model_path)
            self._log(f"模型已保存: {model_path}")

            self._render_ai_tab(compensator)

            self._log("AI 补偿完成!")
            self.statusBar().showMessage("AI 补偿完成")

        except Exception as exc:
            import traceback
            err_msg = traceback.format_exc()
            self._log(f"AI 补偿异常:\n{err_msg}")
            QMessageBox.critical(self, "AI 补偿异常", err_msg[:500])

    def _render_ai_tab(self, compensator: AiCompensator):
        """渲染 AI 补偿 Tab"""
        if not self.solutions_compensated:
            return

        gt_ecef = GT_ECEF.copy()
        r_enu = R_ENU.copy()

        enu_raw, enu_comp, epochs = [], [], []
        for sr, sc in zip(self.solutions, self.solutions_compensated):
            if not sr.valid:
                continue
            enu_raw.append(r_enu @ (sr.pos_ecef - gt_ecef))
            enu_comp.append(r_enu @ (sc.pos_ecef - gt_ecef))
            epochs.append(sr.epoch)

        if not enu_raw:
            return

        enu_raw = np.array(enu_raw)
        enu_comp = np.array(enu_comp)

        self.ai_canvas.figure.clear()
        gs = self.ai_canvas.figure.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        ax1 = self.ai_canvas.figure.add_subplot(gs[0, :])
        ax2 = self.ai_canvas.figure.add_subplot(gs[1, 0])
        ax3 = self.ai_canvas.figure.add_subplot(gs[1, 1])

        # ENU 对比 (E 分量)
        rms_e_raw = np.sqrt(np.mean(enu_raw[:, 0] ** 2))
        rms_e_comp = np.sqrt(np.mean(enu_comp[:, 0] ** 2))

        ax1.plot(epochs, enu_raw[:, 0], label=f'补偿前 E (RMS={rms_e_raw:.3f}m)',
                 color='#d62728', linewidth=0.6, alpha=0.6)
        ax1.plot(epochs, enu_comp[:, 0], label=f'补偿后 E (RMS={rms_e_comp:.3f}m)',
                 color='#1f77b4', linewidth=0.8, alpha=0.8)
        ax1.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.4)
        ax1.set_ylabel('East 误差 (m)', fontsize=9)
        ax1.set_title('AI 补偿前后 East 误差对比', fontsize=10, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=8)

        # 补偿前散点
        e_raw, n_raw = enu_raw[:, 0], enu_raw[:, 1]
        r_raw = np.sqrt(e_raw ** 2 + n_raw ** 2)
        cep95_raw = np.percentile(r_raw, 95)

        ax2.scatter(e_raw, n_raw, c='#d62728', s=5, alpha=0.5)
        circle_raw = Circle((0, 0), cep95_raw, fill=False, color='red',
                           linewidth=2, linestyle='--',
                           label=f'95% CEP={cep95_raw:.2f}m')
        ax2.add_patch(circle_raw)
        ax2.axhline(0, color='gray', linewidth=0.5)
        ax2.axvline(0, color='gray', linewidth=0.5)
        ax2.set_xlabel('East (m)', fontsize=9)
        ax2.set_ylabel('North (m)', fontsize=9)
        ax2.set_title('补偿前', fontsize=10, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        ax2.tick_params(labelsize=8)

        # 补偿后散点
        e_comp, n_comp = enu_comp[:, 0], enu_comp[:, 1]
        r_comp = np.sqrt(e_comp ** 2 + n_comp ** 2)
        cep95_comp = np.percentile(r_comp, 95)

        ax3.scatter(e_comp, n_comp, c='#1f77b4', s=5, alpha=0.5)
        circle_comp = Circle((0, 0), cep95_comp, fill=False, color='blue',
                            linewidth=2, linestyle='--',
                            label=f'95% CEP={cep95_comp:.2f}m')
        ax3.add_patch(circle_comp)
        ax3.axhline(0, color='gray', linewidth=0.5)
        ax3.axvline(0, color='gray', linewidth=0.5)
        ax3.set_xlabel('East (m)', fontsize=9)
        ax3.set_ylabel('North (m)', fontsize=9)
        ax3.set_title('补偿后', fontsize=10, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        ax3.tick_params(labelsize=8)

        try:
            self.ai_canvas.figure.tight_layout()
        except Exception:
            pass
        self.ai_canvas.draw()

        rms3d_raw = np.sqrt(np.mean(np.sum(enu_raw ** 2, axis=1)))
        rms3d_comp = np.sqrt(np.mean(np.sum(enu_comp ** 2, axis=1)))
        improve = (1 - rms3d_comp / max(rms3d_raw, 1e-9)) * 100

        self._log(f"补偿前 3D RMS: {rms3d_raw:.3f}m")
        self._log(f"补偿后 3D RMS: {rms3d_comp:.3f}m")
        self._log(f"精度提升: {improve:.1f}%")

    def _export_results(self):
        """导出报告与图表"""
        if not self.solutions:
            QMessageBox.warning(self, "提示", "请先完成解算！")
            return

        self._log("=" * 50)
        self._log("正在导出报告与图表...")

        try:
            analyzer = TrajectoryAnalyzer(self.solutions, output_dir='data',
                                           scene_label='SPP解算')
            analyzer.compute_errors()
            report = analyzer.generate_report()

            paths = analyzer.plot_all(prefix='export')
            csv_path = analyzer.export_csv('spp_results.csv')

            if self.solutions_compensated:
                comp_paths = analyzer.plot_compensation_comparison(
                    self.solutions, self.solutions_compensated,
                    tag='AI补偿对比'
                )
                paths.extend(comp_paths)

            self._log(f"导出完成! 共生成 {len(paths)} 张图表")
            for p in paths:
                self._log(f"  - {p}")

            QMessageBox.information(self, "导出成功",
                                    f"报告与图表已保存至 data/ 目录\n共 {len(paths)} 张图表")

        except Exception as exc:
            import traceback
            err_msg = traceback.format_exc()
            self._log(f"导出异常:\n{err_msg}")
            QMessageBox.critical(self, "导出异常", err_msg[:500])

    def _log(self, msg: str):
        """追加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {msg}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )


# ============================================================
#  主程序入口
# ============================================================
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()