#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
重新生成演示数据

这个脚本用于重新生成SunflowNexus系统的演示数据
"""

import os
import sys

# 添加项目根目录到系统路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from app.utils.demo_data_generator import DemoDataGenerator

if __name__ == "__main__":
    print("开始生成新的演示数据...")
    
    # 删除旧的演示数据文件（如果存在）
    demo_data_path = os.path.join(script_dir, "data", "demo_data.json")
    if os.path.exists(demo_data_path):
        os.remove(demo_data_path)
        print(f"已删除旧的演示数据: {demo_data_path}")

    # 创建数据目录（如果不存在）
    os.makedirs(os.path.join(script_dir, "data"), exist_ok=True)
    
    # 创建演示数据生成器
    generator = DemoDataGenerator()
    
    # 生成并保存演示数据
    generator.save_demo_data()
    
    print("演示数据重新生成完成！")