#!/usr/bin/env python3
"""
区块链分布式光伏发电点对点交易系统 - 启动脚本
"""

import os
import sys

# 将app目录添加到路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, 'app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# 导入主程序
from app.main import create_gradio_app

if __name__ == "__main__":
    # 创建并启动应用
    app = create_gradio_app()
    app.launch(server_name="0.0.0.0", share=True)
    print("区块链分布式光伏发电点对点交易系统已启动！")