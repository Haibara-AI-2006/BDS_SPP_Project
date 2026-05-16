# test_spp_basic.py
"""
基础 SPP 解算测试脚本 - 诊断问题
"""

import os
import numpy as np
from datetime import datetime

from rinex_parser import RinexNavParser, RinexObsParser
from sat_pos_calculator import SatPosCalculator
from spp_solver import SppSolver

def test_basic_spp():
    """测试基础 SPP 解算流程"""
    
    print("=" * 60)
    print("北斗 SPP 解算诊断测试")
    print("=" * 60)
    
    # 文件路径
    nav_path = os.path.join("data", "BUPT_20260510.nav")
    obs_path = os.path.join("data", "BUPT_Shahe_20260510.obs")
    
    if not os.path.exists(nav_path):
        print(f"错误: 导航文件不存在 {nav_path}")
        return
    
    if not os.path.exists(obs_path):
        print(f"错误: 观测文件不存在 {obs_path}")
        return
    
    # 1. 解析导航文件
    print("\n[1] 解析导航文件...")
    nav_parser = RinexNavParser()
    nav_parser.parse(nav_path)
    print(f"    ✓ 解析成功: {len(nav_parser.ephemeris_pool)} 颗卫星")
    
    # 2. 解析观测文件
    print("\n[2] 解析观测文件...")
    obs_parser = RinexObsParser()
    obs_parser.parse(obs_path)
    print(f"    ✓ 解析成功: {len(obs_parser.epochs)} 个历元")
    print(f"    先验坐标: {obs_parser.approx_pos}")
    
    # 3. 检查第一个历元的观测数据
    print("\n[3] 检查第一个历元...")
    if len(obs_parser.epochs) > 0:
        first_epoch = obs_parser.epochs[0]
        print(f"    时间: {first_epoch.epoch}")
        print(f"    观测卫星数: {len(first_epoch.satellites)}")
        
        # 显示前 5 颗卫星的观测值
        for i, (prn, obs) in enumerate(list(first_epoch.satellites.items())[:5]):
            c2i = obs.get('C2I', 0)
            s2i = obs.get('S2I', 0)
            print(f"    {prn}: C2I={c2i:.3f}m, S2I={s2i:.1f}")
    
    # 4. 创建卫星位置计算器
    print("\n[4] 创建卫星位置计算器...")
    sat_calc = SatPosCalculator(nav_parser)
    print(f"    ✓ 电离层参数已加载")
    
    # 5. 测试单颗卫星位置计算
    print("\n[5] 测试卫星位置计算...")
    if len(obs_parser.epochs) > 0:
        test_epoch = obs_parser.epochs[0]
        test_prn = list(test_epoch.satellites.keys())[0]
        test_obs = test_epoch.satellites[test_prn]
        
        result = sat_calc.compute_sat_pos_clk(
            test_prn, 
            test_epoch.epoch, 
            test_obs.get('C2I', 0)
        )
        
        if result:
            sv_pos, sv_clk = result
            print(f"    ✓ {test_prn} 位置: {sv_pos}")
            print(f"    ✓ {test_prn} 钟差: {sv_clk:.9f}s")
        else:
            print(f"    ✗ {test_prn} 位置计算失败")
    
    # 6. 创建 SPP 解算器
    print("\n[6] 创建 SPP 解算器...")
    solver = SppSolver(
        nav_parser, 
        obs_parser, 
        sat_calc,
        elevation_mask=5.0
    )
    print(f"    ✓ 解算器已创建")
    print(f"    先验坐标: {solver.approx_pos}")
    
    # 7. 测试单历元解算（详细调试）
    print("\n[7] 测试单历元解算（详细模式）...")
    state = np.zeros(4)
    state[:3] = obs_parser.approx_pos.copy()
    
    # 只测试第一个历元，详细输出
    if len(obs_parser.epochs) > 0:
        epoch_obs = obs_parser.epochs[0]
        print(f"\n    历元时间: {epoch_obs.epoch}")
        print(f"    原始观测卫星数: {len(epoch_obs.satellites)}")
        
        # 手动检查每颗卫星
        valid_sats = []
        for prn, obs_dict in epoch_obs.satellites.items():
            pr = obs_dict.get('C2I', 0)
            if pr <= 0:
                print(f"    ✗ {prn}: 伪距无效 ({pr})")
                continue
            
            # 检查星历
            eph = nav_parser.select_ephemeris(prn, epoch_obs.epoch)
            if eph is None:
                print(f"    ✗ {prn}: 无星历")
                continue
            
            # 检查卫星位置
            result = sat_calc.compute_sat_pos_clk(prn, epoch_obs.epoch, pr)
            if result is None:
                print(f"    ✗ {prn}: 位置计算失败")
                continue
            
            sv_pos, sv_clk = result
            if np.linalg.norm(sv_pos) < 1e6:
                print(f"    ✗ {prn}: 位置异常 ({np.linalg.norm(sv_pos):.0f}m)")
                continue
            
            valid_sats.append(prn)
            print(f"    ✓ {prn}: 伪距={pr:.1f}m, 位置范数={np.linalg.norm(sv_pos):.0f}m")
        
        print(f"\n    通过预检的卫星数: {len(valid_sats)}")
        
        # 执行解算
        sol = solver.solve_epoch(epoch_obs, state)
        
        if sol.valid:
            print(f"\n    ✓✓✓ 解算成功！")
            print(f"    使用卫星数: {sol.n_used}")
            print(f"    PDOP: {sol.dop.pdop:.2f}")
            print(f"    位置: ({sol.lat_deg:.6f}°N, {sol.lon_deg:.6f}°E, {sol.height:.1f}m)")
        else:
            print(f"\n    ✗✗✗ 解算失败")
            print(f"    卫星数: {sol.n_used}")
            print(f"    PRN列表: {sol.prn_list}")
    
    print("\n" + "=" * 60)
    print("诊断测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_basic_spp()