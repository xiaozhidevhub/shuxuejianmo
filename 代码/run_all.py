# -*- coding: utf-8 -*-
"""
run_all.py —— 运行所有问题并生成完整结果
============================================
按顺序执行：
  1. relay_site.py - 预处理中继候选点
  2. q1.py - 问题一：单点往返与组批
  3. q2.py - 问题二：多点多架次调度
  4. q3.py - 问题三：通信约束下联合调度
  5. q4.py - 问题四：任务分区与资源配置
"""
import os
import sys
import time
import subprocess

def run_script(script_name, description):
    """运行Python脚本并显示进度"""
    print("\n" + "=" * 70)
    print(f"正在运行：{description}")
    print(f"脚本：{script_name}")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )
        
        elapsed = time.time() - start_time
        
        # 打印输出
        if result.stdout:
            print(result.stdout)
        
        if result.returncode == 0:
            print(f"\n✓ {description} 完成！用时 {elapsed:.1f} 秒")
            return True
        else:
            print(f"\n✗ {description} 失败！")
            if result.stderr:
                print("错误信息：")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n✗ {description} 超时（>10分钟）")
        return False
    except Exception as e:
        print(f"\n✗ {description} 出现异常：{e}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("山区洪涝灾害下无人机运输与通信协同优化")
    print("2026年中国研究生数学建模竞赛 D题")
    print("=" * 70)
    print(f"\n开始时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    overall_start = time.time()
    
    tasks = [
        ("relay_site.py", "预处理：中继候选点分析"),
        ("q1.py", "问题一：单点往返运输能力与货箱组批"),
        ("q2.py", "问题二：异构无人机多点多架次运输调度"),
        ("q3.py", "问题三：通信约束下的运输与中继联合调度"),
        ("q4.py", "问题四：救援任务分区与资源配置优化")
    ]
    
    results = {}
    
    for script, desc in tasks:
        success = run_script(script, desc)
        results[desc] = "成功" if success else "失败"
        
        if not success and script == "relay_site.py":
            print("\n警告：预处理失败可能影响问题三，将继续执行...")
        elif not success:
            print(f"\n警告：{desc} 失败，将继续执行后续问题...")
    
    # 总结
    overall_time = time.time() - overall_start
    
    print("\n" + "=" * 70)
    print("所有问题执行完毕！")
    print("=" * 70)
    print(f"\n总用时：{overall_time / 60:.1f} 分钟")
    print("\n执行结果汇总：")
    for desc, status in results.items():
        symbol = "✓" if status == "成功" else "✗"
        print(f"  {symbol} {desc}: {status}")
    
    print("\n结果文件位置：")
    print("  - 结果/Q1_结果.xlsx")
    print("  - 结果/Q2_结果.xlsx")
    print("  - 结果/Q3_结果.xlsx")
    print("  - 结果/Q4_结果.xlsx")
    print("  - 结果/Q4_分析报告.txt")
    print("  - 结果/图/*.png")
    
    print(f"\n完成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
