"""
test_trajectory_features.py — 轨迹功能扩展测试套件
====================================================
测试范围:
  1. TrajectoryGenerator 椭圆参数化
  2. 数据模拟器轨迹OBS+真值CSV输出
  3. TrajectoryAnalyzer 双模式 (static/trajectory)
  4. 真值CSV读写循环
  5. GUI TrajectoryCanvas 数据缓冲

运行:
    python -m unittest test_trajectory_features.py -v
"""

import os
import sys
import unittest
import math
import tempfile
import shutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
#  测试 1: TrajectoryGenerator 椭圆参数化
# ============================================================
class TestTrajectoryGenerator(unittest.TestCase):
    """椭圆轨迹生成器 — 验证参数方程正确性"""

    def setUp(self):
        from data_simulator import TrajectoryGenerator
        self.gen = TrajectoryGenerator(a=500.0, b=300.0,
                                        period_sec=3600.0, height=35.0)

    def test_enu_at_t0(self):
        """t=0 应位于椭圆长轴正端: ENU=(a, 0, 0)"""
        enu = self.gen.get_enu_at(0.0)
        np.testing.assert_array_almost_equal(enu, [500.0, 0.0, 0.0], decimal=4)

    def test_enu_quarter_period(self):
        """t=T/4 应位于短轴正端: ENU=(0, b, 0)"""
        enu = self.gen.get_enu_at(900.0)
        np.testing.assert_array_almost_equal(enu, [0.0, 300.0, 0.0], decimal=4)

    def test_enu_half_period(self):
        """t=T/2 应位于长轴负端: ENU=(-a, 0, 0)"""
        enu = self.gen.get_enu_at(1800.0)
        np.testing.assert_array_almost_equal(enu, [-500.0, 0.0, 0.0], decimal=4)

    def test_enu_full_period_closed(self):
        """t=T 应回到起点 (椭圆闭合性)"""
        enu0 = self.gen.get_enu_at(0.0)
        enu_T = self.gen.get_enu_at(3600.0)
        np.testing.assert_array_almost_equal(enu0, enu_T, decimal=4)

    def test_ecef_distance_from_center(self):
        """ECEF 到中心距离应在 [b, a] 范围内"""
        from data_simulator import GT_ECEF
        for t in np.linspace(0, 3600, 200):
            ecef = self.gen.get_ecef_at(t)
            dist = np.linalg.norm(ecef - GT_ECEF)
            self.assertGreaterEqual(dist, 300.0 - 0.1, f"t={t}: dist={dist}")
            self.assertLessEqual(dist, 500.0 + 0.1, f"t={t}: dist={dist}")

    def test_ecef_enu_roundtrip(self):
        """ECEF↔ENU 转换闭合性 (R 正交)"""
        from data_simulator import GT_ECEF, GT_R_ENU
        for t in [0.0, 500.0, 1500.0, 2700.0]:
            ecef = self.gen.get_ecef_at(t)
            enu_back = GT_R_ENU @ (ecef - GT_ECEF)
            enu_direct = self.gen.get_enu_at(t)
            np.testing.assert_array_almost_equal(enu_back, enu_direct, decimal=4)


# ============================================================
#  测试 2: 数据模拟器轨迹OBS + 真值CSV
# ============================================================
class TestDataSimulatorTrajectory(unittest.TestCase):
    """数据模拟器轨迹模式集成测试"""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix='bds_traj_test_')
        cls.nav_path = os.path.join("data", "BUPT_20260510.nav")
        if not os.path.exists(cls.nav_path):
            raise unittest.SkipTest(f"NAV 文件缺失: {cls.nav_path}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_trajectory_obs_and_csv_generation(self):
        """生成短时段轨迹OBS+真值CSV, 验证文件存在和列结构"""
        from data_simulator import (
            DataSimulator, SCENE_OPEN_SKY, TrajectoryGenerator, GT_ECEF,
        )

        sim = DataSimulator(self.nav_path, output_dir=self.tmpdir)
        gen = TrajectoryGenerator(a=500.0, b=300.0, period_sec=3600.0)
        start_time = datetime(2026, 5, 10, 0, 0, 0)

        obs_path, csv_path, truth_path = sim.generate_trajectory_obs_file(
            start_time=start_time,
            duration_hours=0.1,                # 6 分钟 = 12 个 30s 历元
            interval_sec=30.0,
            scene=SCENE_OPEN_SKY,
            trajectory_gen=gen,
            obs_filename='test_traj.obs',
            csv_filename='test_traj.csv',
            truth_filename='test_traj_truth.csv',
        )

        # 文件存在
        self.assertTrue(os.path.exists(obs_path))
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(truth_path))

        # 真值 CSV 列结构
        truth_df = pd.read_csv(truth_path)
        for col in ['epoch', 'epoch_idx', 'gt_x', 'gt_y', 'gt_z',
                    'gt_lat_deg', 'gt_lon_deg', 'gt_height',
                    'gt_e', 'gt_n', 'gt_u']:
            self.assertIn(col, truth_df.columns, f"缺失列: {col}")

        # 真值 ECEF 应符合椭圆
        for _, row in truth_df.iterrows():
            ecef = np.array([row['gt_x'], row['gt_y'], row['gt_z']])
            dist = np.linalg.norm(ecef - GT_ECEF)
            self.assertGreaterEqual(dist, 290.0)
            self.assertLessEqual(dist, 510.0)


# ============================================================
#  测试 3: TrajectoryAnalyzer 双模式
# ============================================================
class TestTrajectoryAnalyzerDualMode(unittest.TestCase):
    """轨迹分析器静态/轨迹模式测试"""

    def _make_fake_solutions(self, n=20, ecef_offset=(0.0, 0.0, 0.0)):
        """构造带 ECEF 偏移的假解算结果"""
        from spp_solver import EpochSolution, DopValues
        from trajectory_analyzer import GT_ECEF
        sols = []
        t0 = datetime(2026, 5, 10, 0, 0, 0)
        for i in range(n):
            sol = EpochSolution(t0 + timedelta(seconds=i * 30))
            sol.valid = True
            sol.pos_ecef = GT_ECEF.copy() + np.array(ecef_offset)
            sol.lat_deg = 40.1575
            sol.lon_deg = 116.2885
            sol.height = 35.0
            sol.clock_bias = 60.0
            sol.n_used = 8
            sol.dop = DopValues()
            sol.dop.gdop = 2.0
            sol.dop.pdop = 1.5
            sol.dop.hdop = 1.0
            sol.dop.vdop = 1.2
            sol.dop.tdop = 1.0
            sol.sigma0 = 0.5
            sol.residuals = np.zeros(8)
            sol.prn_list = ['C01'] * 8
            sol.elevations = np.full(8, math.radians(45))
            sol.azimuths = np.zeros(8)
            sol.snr_values = np.full(8, 40.0)
            sol.sat_positions = np.zeros((8, 3))
            sol.weights = np.ones(8)
            sols.append(sol)
        return sols

    def test_static_mode_default(self):
        """static 模式: 默认行为, 与原代码一致"""
        from trajectory_analyzer import TrajectoryAnalyzer
        sols = self._make_fake_solutions(n=10, ecef_offset=(5.0, 0.0, 0.0))
        analyzer = TrajectoryAnalyzer(sols, output_dir=tempfile.gettempdir())
        df = analyzer.compute_errors()
        self.assertEqual(len(df), 10)
        np.testing.assert_array_almost_equal(df['err_3d'].values,
                                             [5.0] * 10, decimal=3)

    def test_trajectory_mode_zero_error(self):
        """trajectory 模式: 解算位置=真值 → 误差应为 0"""
        from trajectory_analyzer import TrajectoryAnalyzer
        from data_simulator import TrajectoryGenerator

        sols = self._make_fake_solutions(n=10)
        gen = TrajectoryGenerator(a=500, b=300, period_sec=3600)

        truth_records = []
        for i, sol in enumerate(sols):
            tau = i * 30.0
            ecef_truth = gen.get_ecef_at(tau)
            sol.pos_ecef = ecef_truth.copy()
            truth_records.append({
                'epoch': sol.epoch,
                'gt_x': ecef_truth[0],
                'gt_y': ecef_truth[1],
                'gt_z': ecef_truth[2],
            })
        truth_df = pd.DataFrame(truth_records)

        analyzer = TrajectoryAnalyzer(
            sols, output_dir=tempfile.gettempdir(),
            truth_mode='trajectory', truth_trajectory=truth_df,
        )
        df = analyzer.compute_errors()
        np.testing.assert_array_almost_equal(df['err_3d'].values,
                                             [0.0] * 10, decimal=4)

    def test_trajectory_mode_with_offset(self):
        """trajectory 模式: 解算=真值+(3,4,0) → err_3d=5 (ENU旋转保模长)"""
        from trajectory_analyzer import TrajectoryAnalyzer
        from data_simulator import TrajectoryGenerator

        sols = self._make_fake_solutions(n=10)
        gen = TrajectoryGenerator(a=500, b=300, period_sec=3600)

        offset = np.array([3.0, 4.0, 0.0])
        truth_records = []
        for i, sol in enumerate(sols):
            tau = i * 30.0
            ecef_truth = gen.get_ecef_at(tau)
            sol.pos_ecef = ecef_truth + offset
            truth_records.append({
                'epoch': sol.epoch,
                'gt_x': ecef_truth[0],
                'gt_y': ecef_truth[1],
                'gt_z': ecef_truth[2],
            })
        truth_df = pd.DataFrame(truth_records)

        analyzer = TrajectoryAnalyzer(
            sols, output_dir=tempfile.gettempdir(),
            truth_mode='trajectory', truth_trajectory=truth_df,
        )
        df = analyzer.compute_errors()
        np.testing.assert_array_almost_equal(df['err_3d'].values,
                                             [5.0] * 10, decimal=3)


# ============================================================
#  测试 4: 真值CSV读写循环
# ============================================================
class TestTruthCsvIO(unittest.TestCase):
    def test_save_load_roundtrip(self):
        from data_simulator import TrajectoryGenerator
        from trajectory_analyzer import load_truth_trajectory_csv

        gen = TrajectoryGenerator(a=500, b=300, period_sec=3600)
        t0 = datetime(2026, 5, 10, 0, 0, 0)
        records = []
        for i in range(20):
            tau = i * 30.0
            ecef = gen.get_ecef_at(tau)
            enu = gen.get_enu_at(tau)
            records.append({
                'epoch': (t0 + timedelta(seconds=tau)).strftime('%Y-%m-%d %H:%M:%S'),
                'epoch_idx': i,
                'gt_x': ecef[0], 'gt_y': ecef[1], 'gt_z': ecef[2],
                'gt_lat_deg': 40.1575, 'gt_lon_deg': 116.2885,
                'gt_height': 35.0,
                'gt_e': enu[0], 'gt_n': enu[1], 'gt_u': enu[2],
            })
        df = pd.DataFrame(records)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                         delete=False) as f:
            tmp_path = f.name
        df.to_csv(tmp_path, index=False)

        try:
            loaded = load_truth_trajectory_csv(tmp_path)
            self.assertEqual(len(loaded), 20)
            self.assertIn('gt_x', loaded.columns)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded['epoch']))
        finally:
            os.remove(tmp_path)


# ============================================================
#  测试 5: GUI TrajectoryCanvas (轻量, 不启动事件循环)
# ============================================================
class TestTrajectoryCanvas(unittest.TestCase):
    """仅测试数据缓冲与状态变化"""

    @classmethod
    def setUpClass(cls):
        try:
            from PyQt5.QtWidgets import QApplication
            cls.app = QApplication.instance()
            if cls.app is None:
                cls.app = QApplication([])
        except ImportError:
            raise unittest.SkipTest("PyQt5 不可用")

    def test_init_empty_buffers(self):
        from main_gui import TrajectoryCanvas
        canvas = TrajectoryCanvas()
        self.assertEqual(len(canvas.buf_e_solve), 0)
        self.assertEqual(len(canvas.buf_n_solve), 0)

    def test_update_appends_to_buffers(self):
        from main_gui import TrajectoryCanvas
        canvas = TrajectoryCanvas()
        canvas.update_point(epoch_idx=0, sol_e=11.0, sol_n=22.0,
                            gt_e=10.0, gt_n=20.0)
        self.assertEqual(len(canvas.buf_e_solve), 1)
        self.assertAlmostEqual(canvas.buf_e_solve[0], 11.0)
        self.assertAlmostEqual(canvas.buf_n_solve[0], 22.0)

    def test_set_truth_curve(self):
        from main_gui import TrajectoryCanvas
        canvas = TrajectoryCanvas()
        truth_e = np.linspace(-500, 500, 100)
        truth_n = np.linspace(-300, 300, 100)
        canvas.set_truth_curve(truth_e, truth_n)
        self.assertTrue(True)  # 不抛异常即通过


if __name__ == '__main__':
    unittest.main(verbosity=2)