import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import time
import os
import hashlib
import json
import sys
from typing import Dict, List, Any, Optional, Tuple

# 添加当前目录到路径，保证可以导入本地模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入自定义模块
from blockchain.core import Blockchain
from blockchain.smart_contract import EnergyTradingContract, User
from blockchain.energy_coin import EnergyCoin
from models.prediction_model import SolarPowerForecaster, LoadForecaster
from utils.encryption import DataEncryptor
from utils.data_collector import EnergyDataGenerator, EnergyDataCollector
from utils.demo_data_generator import DemoDataGenerator

# 创建应用实例
blockchain = Blockchain(difficulty=4)
smart_contract = EnergyTradingContract()
data_encryptor = DataEncryptor()
data_collector = EnergyDataCollector()

# 创建数据目录
os.makedirs("data", exist_ok=True)

# 存储当前登录用户
current_user = {"address": None, "name": None}

# 生成随机地址
def generate_address() -> str:
    """生成随机用户地址"""
    random_bytes = os.urandom(20)
    return hashlib.sha256(random_bytes).hexdigest()[:40]


# 用户注册函数
def register_user(name: str, is_producer: bool, is_consumer: bool, panel_capacity: float = 5.0, 
                user_type: str = "住宅", base_load: float = 2.0) -> Dict[str, Any]:
    """
    注册新用户
    
    Args:
        name: 用户名
        is_producer: 是否为发电方
        is_consumer: 是否为用电方
        panel_capacity: 光伏板容量(kW)
        user_type: 用户类型 ("住宅", "商业", "工业")
        base_load: 基础负荷(kW)
        
    Returns:
        注册结果信息
    """
    # 生成用户地址
    address = generate_address()
    
    # 在智能合约中注册用户
    try:
        user = smart_contract.register_user(address, name, is_producer, is_consumer)
        
        # 生成用户密钥对
        public_key = data_encryptor.generate_user_keypair(address)
        
        # 创建用户配置
        user_config = {
            "panel_capacity": panel_capacity,
            "user_type": user_type,
            "base_load": base_load
        }
        
        # 生成样本历史数据
        data_collector.generate_sample_data(address, user_config, days_back=30)
        
        return {
            "success": True,
            "message": f"用户 {name} 注册成功",
            "user": {
                "address": address,
                "name": name,
                "is_producer": is_producer,
                "is_consumer": is_consumer,
                "reputation": user.reputation,
                "is_green_certified": user.is_green_certified,
                "energy_coin_address": user.energy_coin_address,
                "energy_coin_balance": smart_contract.get_energy_coin_balance(address)
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"注册失败: {str(e)}"
        }


# 用户登录函数
def login_user(address: str) -> Dict[str, Any]:
    """
    用户登录
    
    Args:
        address: 用户地址
        
    Returns:
        登录结果信息
    """
    # 检查用户是否存在
    if address in smart_contract.users:
        user = smart_contract.users[address]
        current_user["address"] = address
        current_user["name"] = user.name
        
        # 获取EnergyCoin余额
        energy_coin_balance = smart_contract.get_energy_coin_balance(address)
        
        return {
            "success": True,
            "message": f"欢迎回来, {user.name}!",
            "user": {
                "address": address,
                "name": user.name,
                "is_producer": user.is_producer,
                "is_consumer": user.is_consumer,
                "reputation": user.reputation,
                "is_green_certified": user.is_green_certified,
                "green_energy_subsidy": user.green_energy_subsidy,
                "energy_coin_address": user.energy_coin_address,
                "energy_coin_balance": energy_coin_balance,
                "is_mining": user.is_mining
            }
        }
    else:
        return {
            "success": False,
            "message": "用户不存在"
        }


# 获取用户信息
def get_user_info(address: str) -> Dict[str, Any]:
    """
    获取用户详细信息
    
    Args:
        address: 用户地址
        
    Returns:
        用户信息
    """
    if address in smart_contract.users:
        user = smart_contract.users[address]
        
        # 计算综合信誉值
        reputation = smart_contract.calculate_reputation(address)
        
        # 获取用户数据
        user_data = data_collector.get_user_data(address)
        
        # 获取EnergyCoin余额
        energy_coin_balance = smart_contract.get_energy_coin_balance(address)
        
        # 获取EnergyCoin质押信息
        staking_info = smart_contract.energy_coin.get_staking_info(user.energy_coin_address)
        
        # 如果有数据，计算汇总信息
        if not user_data.empty:
            total_generation = user_data["solar_power"].sum()
            total_consumption = user_data["load"].sum()
            net_energy = total_generation - total_consumption
        else:
            total_generation = 0
            total_consumption = 0
            net_energy = 0
        
        return {
            "address": address,
            "name": user.name,
            "is_producer": user.is_producer,
            "is_consumer": user.is_consumer,
            "reputation": reputation,
            "is_green_certified": user.is_green_certified,
            "green_energy_subsidy": user.green_energy_subsidy,
            "active_score": user.active_score,
            "total_generation": total_generation,
            "total_consumption": total_consumption,
            "net_energy": net_energy,
            "energy_coin_address": user.energy_coin_address,
            "energy_coin_balance": energy_coin_balance,
            "is_mining": user.is_mining,
            "staking_info": staking_info
        }
    else:
        return {
            "success": False,
            "message": "用户不存在"
        }


# 获取用户能源数据
def get_user_energy_data(address: str) -> pd.DataFrame:
    """
    获取用户能源数据
    
    Args:
        address: 用户地址
        
    Returns:
        能源数据DataFrame
    """
    return data_collector.get_user_data(address)


# 创建挂单
def create_energy_listing(producer_address: str, amount: float, price_per_kwh: float, 
                        valid_hours: int = 24) -> Dict[str, Any]:
    """
    创建能源挂单
    
    Args:
        producer_address: 发电方地址
        amount: 电量(kWh)
        price_per_kwh: 每kWh价格
        valid_hours: 有效时长(小时)
        
    Returns:
        挂单结果
    """
    try:
        valid_until = time.time() + valid_hours * 3600
        listing = smart_contract.list_energy_for_sale(
            producer_address, amount, price_per_kwh, valid_until
        )
        
        # 将交易记录到区块链
        transaction_data = {
            "type": "listing",
            "producer": producer_address,
            "amount": amount,
            "price_per_kwh": price_per_kwh,
            "listing_id": listing["id"],
            "timestamp": time.time()
        }
        blockchain.add_transaction(transaction_data)
        
        return {
            "success": True,
            "message": f"能源挂单创建成功，ID: {listing['id']}",
            "listing": listing
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"创建挂单失败: {str(e)}"
        }


# 投标
def place_energy_bid(consumer_address: str, listing_id: int, amount: float) -> Dict[str, Any]:
    """
    对能源挂单进行投标
    
    Args:
        consumer_address: 用电方地址
        listing_id: 挂单ID
        amount: 电量(kWh)
        
    Returns:
        投标结果
    """
    try:
        bid = smart_contract.place_bid(consumer_address, listing_id, amount)
        
        # 将交易记录到区块链
        transaction_data = {
            "type": "bid",
            "consumer": consumer_address,
            "listing_id": listing_id,
            "amount": amount,
            "bid_id": bid["id"],
            "timestamp": time.time()
        }
        blockchain.add_transaction(transaction_data)
        
        return {
            "success": True,
            "message": f"投标成功，ID: {bid['id']}",
            "bid": bid
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"投标失败: {str(e)}"
        }


# 接受投标
def accept_energy_bid(producer_address: str, bid_id: int) -> Dict[str, Any]:
    """
    接受电力投标
    
    Args:
        producer_address: 发电方地址
        bid_id: 投标ID
        
    Returns:
        接受投标结果
    """
    try:
        transaction = smart_contract.accept_bid(producer_address, bid_id)
        
        # 将交易记录到区块链
        transaction_data = {
            "type": "accepted_bid",
            "producer": producer_address,
            "consumer": transaction["consumer_address"],
            "amount": transaction["amount"],
            "price_per_kwh": transaction["price_per_kwh"],
            "total_price": transaction["total_price"],
            "transaction_id": transaction["id"],
            "timestamp": time.time()
        }
        
        # 记录此交易到区块链
        block_index = blockchain.add_transaction(transaction_data)
        
        return {
            "success": True,
            "message": f"成功接受投标，交易ID: {transaction['id']}",
            "transaction": transaction
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"接受投标失败: {str(e)}"
        }


# 确认电力交付
def confirm_energy_delivery(transaction_id: str) -> Dict[str, Any]:
    """
    确认能源已交付
    
    Args:
        transaction_id: 交易ID
        
    Returns:
        确认结果
    """
    try:
        transaction = smart_contract.confirm_delivery(transaction_id)
        
        # 将确认记录到区块链
        confirmation_data = {
            "type": "delivery_confirmed",
            "transaction_id": transaction_id,
            "producer": transaction["producer_address"],
            "consumer": transaction["consumer_address"],
            "amount": transaction["amount"],
            "price_per_kwh": transaction["price_per_kwh"],
            "total_price": transaction["total_price"],
            "timestamp": time.time()
        }
        
        # 记录此确认到区块链
        block_index = blockchain.add_transaction(confirmation_data)
        
        return {
            "success": True,
            "message": f"成功确认能源交付，交易ID: {transaction_id}",
            "transaction": transaction
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"确认交付失败: {str(e)}"
        }


# 申请绿能认证
def apply_for_green_certification(producer_address: str) -> Dict[str, Any]:
    """
    申请绿能认证
    
    Args:
        producer_address: 发电方地址
        
    Returns:
        申请结果
    """
    try:
        # 获取用户数据
        user_data = data_collector.get_user_data(producer_address)
        
        if user_data.empty:
            return {
                "success": False,
                "message": "没有足够的发电数据，无法申请绿能认证"
            }
        
        # 计算总发电量
        total_production = user_data["solar_power"].sum()
        
        # 申请认证
        is_certified = smart_contract.apply_for_green_certification(
            producer_address, total_production
        )
        
        if is_certified:
            return {
                "success": True,
                "message": "恭喜！您已获得绿能认证",
                "certification_status": True
            }
        else:
            return {
                "success": False,
                "message": "未满足绿能认证条件",
                "certification_status": False
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"申请绿能认证失败: {str(e)}"
        }


# 预测发电量
def predict_solar_generation(address: str, days: int = 1) -> Dict[str, Any]:
    """
    预测未来光伏发电量
    
    Args:
        address: 用户地址
        days: 预测天数
        
    Returns:
        预测结果
    """
    try:
        print(f"开始为用户 {address} 预测未来 {days} 天的发电量...")
        
        # 获取历史数据
        user_data = data_collector.get_user_data(address)
        
        if user_data.empty:
            print("预测失败: 找不到用户历史数据")
            return {
                "success": False,
                "message": "没有找到用户历史数据"
            }
        
        # 准备输入数据
        power_data = user_data[["timestamp", "solar_power"]].copy()
        power_data["timestamp"] = pd.to_datetime(power_data["timestamp"])
        power_data.set_index("timestamp", inplace=True)
        
        weather_data = user_data[["timestamp", "temperature", "irradiance"]].copy()
        weather_data["timestamp"] = pd.to_datetime(weather_data["timestamp"])
        weather_data.set_index("timestamp", inplace=True)
        
        print(f"准备训练数据完成, 发电数据点: {len(power_data)}, 气象数据点: {len(weather_data)}")
        
        # 检查数据连续性
        if len(power_data) < 48:  # 至少需要48小时的数据
            print("预测失败: 历史数据不足48小时")
            return {
                "success": False,
                "message": "历史数据不足，至少需要48小时的数据进行预测"
            }
        
        # 创建预测模型
        print("初始化SolarPowerForecaster模型...")
        forecaster = SolarPowerForecaster(sequence_length=24, forecast_horizon=24 * days)
        
        # 训练模型 (在实际应用中，应该预先训练好模型并保存)
        print("开始训练模型...")
        # 增加训练轮数，从30轮增加到100轮
        history = forecaster.train_with_weather(power_data, weather_data, "solar_power", epochs=5, batch_size=16)
        print(f"模型训练完成，最终损失: {history['loss'][-1]:.6f}")
        
        # 预测
        print("开始预测...")
        try:
            prediction = forecaster.predict(power_data, ["temperature", "irradiance"], "solar_power")
            print(f"预测完成，预测数据点数量: {len(prediction)}")
        except Exception as e:
            print(f"预测过程出错: {str(e)}")
            return {
                "success": False,
                "message": f"预测失败: {str(e)}"
            }
        
        # 创建预测时间索引
        last_time = power_data.index[-1]
        prediction_index = pd.date_range(
            start=last_time + pd.Timedelta(hours=1),
            periods=len(prediction),
            freq='H'
        )
        
        # 创建预测结果DataFrame
        prediction_df = pd.DataFrame({
            "timestamp": prediction_index,
            "solar_power_prediction": prediction
        })
        
        # 可视化
        print("创建可视化图表...")
        fig = go.Figure()
        
        # 添加历史数据
        fig.add_trace(go.Scatter(
            x=power_data.index[-48:],  # 只显示最近48小时的历史数据
            y=power_data["solar_power"][-48:],
            mode='lines',
            name='历史发电量'
        ))
        
        # 添加预测数据
        fig.add_trace(go.Scatter(
            x=prediction_df["timestamp"],
            y=prediction_df["solar_power_prediction"],
            mode='lines',
            name='预测发电量',
            line=dict(dash='dash')
        ))
        
        # 更新布局
        fig.update_layout(
            title="光伏发电预测",
            xaxis_title="时间",
            yaxis_title="发电量 (kW)",
            legend=dict(x=0, y=1, traceorder="normal"),
            hovermode="x"
        )
        
        print("预测过程成功完成")
        return {
            "success": True,
            "message": "预测成功",
            "prediction": prediction_df.to_dict(orient="records"),
            "plot": fig
        }
    except Exception as e:
        import traceback
        print(f"预测过程发生错误: {str(e)}")
        traceback.print_exc()
        return {
            "success": False,
            "message": f"预测失败: {str(e)}"
        }


# 预测用电量
def predict_load(address: str, days: int = 1) -> Dict[str, Any]:
    """
    预测未来用电量
    
    Args:
        address: 用户地址
        days: 预测天数
        
    Returns:
        预测结果
    """
    try:
        print(f"开始为用户 {address} 预测未来 {days} 天的用电量...")
        
        # 获取历史数据
        user_data = data_collector.get_user_data(address)
        
        if user_data.empty:
            print("预测失败: 找不到用户历史数据")
            return {
                "success": False,
                "message": "没有找到用户历史数据"
            }
        
        # 获取用户画像
        user_profile = data_collector.get_user_profile(address)
        
        if not user_profile:
            print("预测失败: 找不到用户画像数据")
            return {
                "success": False,
                "message": "没有找到用户画像数据"
            }
            
        # 准备输入数据
        load_data = user_data[["timestamp", "load"]].copy()
        load_data["timestamp"] = pd.to_datetime(load_data["timestamp"])
        load_data.set_index("timestamp", inplace=True)
        
        print(f"准备训练数据完成, 用电数据点: {len(load_data)}")
        
        # 检查数据连续性
        if len(load_data) < 48:  # 至少需要48小时的数据
            print("预测失败: 历史数据不足48小时")
            return {
                "success": False,
                "message": "历史数据不足，至少需要48小时的数据进行预测"
            }
        
        # 创建预测模型
        print("初始化LoadForecaster模型...")
        forecaster = LoadForecaster(sequence_length=24, forecast_horizon=24 * days)
        
        # 训练模型 (在实际应用中，应该预先训练好模型并保存)
        print("开始训练模型...")
        # 增加训练轮数
        history = forecaster.train_with_user_profile(load_data, user_profile, "load", epochs=5, batch_size=16)
        print(f"模型训练完成，最终损失: {history['loss'][-1]:.6f}")
        
        # 预测
        print("开始预测...")
        try:
            prediction = forecaster.predict(load_data, [], "load")
            print(f"预测完成，预测数据点数量: {len(prediction)}")
        except Exception as e:
            print(f"预测过程出错: {str(e)}")
            return {
                "success": False,
                "message": f"预测失败: {str(e)}"
            }
        
        # 创建预测时间索引
        last_time = load_data.index[-1]
        prediction_index = pd.date_range(
            start=last_time + pd.Timedelta(hours=1),
            periods=len(prediction),
            freq='H'
        )
        
        # 创建预测结果DataFrame
        prediction_df = pd.DataFrame({
            "timestamp": prediction_index,
            "load_prediction": prediction
        })
        
        # 可视化
        print("创建可视化图表...")
        fig = go.Figure()
        
        # 添加历史数据
        fig.add_trace(go.Scatter(
            x=load_data.index[-48:],  # 只显示最近48小时的历史数据
            y=load_data["load"][-48:],
            mode='lines',
            name='历史用电量'
        ))
        
        # 添加预测数据
        fig.add_trace(go.Scatter(
            x=prediction_df["timestamp"],
            y=prediction_df["load_prediction"],
            mode='lines',
            name='预测用电量',
            line=dict(dash='dash')
        ))
        
        # 更新布局
        fig.update_layout(
            title="用电负荷预测",
            xaxis_title="时间",
            yaxis_title="用电量 (kW)",
            legend=dict(x=0, y=1, traceorder="normal"),
            hovermode="x"
        )
        
        print("预测过程成功完成")
        return {
            "success": True,
            "message": "预测成功",
            "prediction": prediction_df.to_dict(orient="records"),
            "plot": fig
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"预测失败: {str(e)}"
        }


# 获取活跃挂单列表
def get_active_listings() -> List[Dict[str, Any]]:
    """
    获取所有活跃的能源挂单
    
    Returns:
        活跃挂单列表
    """
    active_listings = []
    
    for listing in smart_contract.energy_listings:
        if listing["status"] == "active" and listing["valid_until"] > time.time():
            # 获取发布者信息
            producer = smart_contract.users[listing["producer_address"]]
            
            # 添加挂单信息
            active_listings.append({
                "id": listing["id"],
                "producer_address": listing["producer_address"],
                "producer_name": producer.name,
                "amount": listing["amount"],
                "price_per_kwh": listing["price_per_kwh"],
                "total_price": listing["amount"] * listing["price_per_kwh"],
                "created_at": datetime.datetime.fromtimestamp(listing["created_at"]).strftime('%Y-%m-%d %H:%M:%S'),
                "valid_until": datetime.datetime.fromtimestamp(listing["valid_until"]).strftime('%Y-%m-%d %H:%M:%S'),
                "green_certified": listing["green_certified"],
                "producer_reputation": smart_contract.calculate_reputation(listing["producer_address"])
            })
    
    # 按创建时间排序
    active_listings.sort(key=lambda x: x["id"])
    
    return active_listings


# 获取用户交易历史
def get_user_transactions(address: str) -> List[Dict[str, Any]]:
    """
    获取用户的交易历史
    
    Args:
        address: 用户地址
        
    Returns:
        交易历史列表
    """
    user_transactions = []
    
    # 查找用户相关的所有交易
    for tx in smart_contract.transactions:
        if tx["producer_address"] == address or tx["consumer_address"] == address:
            producer = smart_contract.users[tx["producer_address"]]
            consumer = smart_contract.users[tx["consumer_address"]]
            
            user_transactions.append({
                "id": tx["id"],
                "producer_address": tx["producer_address"],
                "producer_name": producer.name,
                "consumer_address": tx["consumer_address"],
                "consumer_name": consumer.name,
                "amount": tx["amount"],
                "price_per_kwh": tx["price_per_kwh"],
                "total_price": tx["total_price"],
                "status": tx["status"],
                "created_at": datetime.datetime.fromtimestamp(tx["created_at"]).strftime('%Y-%m-%d %H:%M:%S'),
                "actual_amount": tx.get("actual_amount"),
                "delivery_ratio": tx.get("delivery_ratio")
            })
    
    # 按创建时间排序
    user_transactions.sort(key=lambda x: x["id"], reverse=True)
    
    return user_transactions


# 创建样例用户
def create_sample_users():
    """创建系统的样例用户，包括发电方、用电方和双向用户"""
    # 样例发电方
    register_user("太阳能发电公司", True, False, panel_capacity=50.0, user_type="工业", base_load=10.0)
    
    # 样例用电方
    register_user("社区居民", False, True, panel_capacity=0.0, user_type="住宅", base_load=3.0)
    
    # 样例双向用户（既发电又用电）
    register_user("绿色农场", True, True, panel_capacity=10.0, user_type="商业", base_load=5.0)


# 加载演示数据
def load_demo_data():
    """加载或创建演示数据"""
    print("开始加载演示数据...")
    generator = DemoDataGenerator()
    
    # 检查data/demo_data.json是否存在
    if os.path.exists("data/demo_data.json"):
        # 加载现有数据
        demo_data = generator.load_demo_data_to_system(blockchain, smart_contract, data_collector)
        print(f"已从文件加载演示数据: {len(smart_contract.users)}个用户, {len(blockchain.chain)}个区块")
    else:
        # 生成并保存新数据
        print("未发现现有数据，正在生成新的演示数据...")
        generator.save_demo_data()
        demo_data = generator.load_demo_data_to_system(blockchain, smart_contract, data_collector)
        print(f"已生成新的演示数据: {len(smart_contract.users)}个用户, {len(blockchain.chain)}个区块")
    
    return demo_data


# 创建Gradio应用界面
def create_gradio_app():
    # 不再调用create_sample_users，改为加载演示数据
    demo_data = load_demo_data()
    
    # 创建Gradio界面
    with gr.Blocks(title="Sunflow Nexus分布式光伏发电点对点交易系统") as app:
        gr.Markdown("# Sunflow Nexus分布式光伏发电点对点交易系统")
        
        with gr.Tab("用户注册与登录"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 新用户注册")
                    name_input = gr.Textbox(label="用户名")
                    with gr.Row():
                        is_producer = gr.Checkbox(label="是发电方")
                        is_consumer = gr.Checkbox(label="是用电方")
                    
                    with gr.Accordion("高级配置", open=False):
                        panel_capacity = gr.Slider(minimum=0, maximum=100, value=5, label="光伏板容量(kW)")
                        user_type = gr.Dropdown(["住宅", "商业", "工业"], label="用户类型", value="住宅")
                        base_load = gr.Slider(minimum=0, maximum=50, value=2, label="基础负荷(kW)")
                    
                    register_button = gr.Button("注册")
                    register_output = gr.Textbox(label="注册结果")
                    user_address_output = gr.Textbox(label="用户地址 (保存此地址用于登录)")
                    
                with gr.Column():
                    gr.Markdown("## 用户登录")
                    address_input = gr.Textbox(label="用户地址")
                    login_button = gr.Button("登录")
                    login_output = gr.Textbox(label="登录结果")
        
        with gr.Tab("用户信息"):
            gr.Markdown("## 个人信息")
            refresh_user_info_button = gr.Button("刷新个人信息")
            
            with gr.Row():
                with gr.Column():
                    user_name = gr.Textbox(label="用户名")
                    user_address = gr.Textbox(label="用户地址")
                    user_type_info = gr.Textbox(label="用户类型")
                    is_green_certified = gr.Checkbox(label="绿能认证", interactive=False)
                
                with gr.Column():
                    reputation = gr.Number(label="信誉值")
                    active_score = gr.Number(label="活跃度")
                    green_energy_subsidy = gr.Number(label="绿能补贴余额")
                    energy_coin_balance = gr.Number(label="EnergyCoin余额")
            
            with gr.Row():
                with gr.Column():
                    total_generation = gr.Number(label="总发电量(kWh)")
                    total_consumption = gr.Number(label="总用电量(kWh)")
                    net_energy = gr.Number(label="净电力(kWh)")
                
                with gr.Column():
                    energy_coin_address = gr.Textbox(label="EnergyCoin钱包地址")
                    is_mining = gr.Checkbox(label="是否参与挖矿", interactive=False)
                    staked_amount = gr.Number(label="质押金额", value=0)
                    stake_status = gr.Textbox(label="质押状态", value="未质押")
            
            with gr.Row():
                apply_certification_button = gr.Button("申请绿能认证")
                certification_result = gr.Textbox(label="认证结果")
        
        with gr.Tab("我的能量数据看板"):
            gr.Markdown("## 我的能量数据看板")
            
            with gr.Row():
                show_data_button = gr.Button("显示历史数据")
                data_days = gr.Slider(minimum=1, maximum=30, value=7, step=1, label="显示天数")
            
            with gr.Row():
                energy_plot = gr.Plot(label="能量数据图表")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 发电预测")
                    predict_generation_button = gr.Button("预测发电量")
                    generation_days = gr.Slider(minimum=1, maximum=7, value=1, step=1, label="预测天数")
                    generation_plot = gr.Plot(label="发电预测")
                
                with gr.Column():
                    gr.Markdown("## 用电预测")
                    predict_load_button = gr.Button("预测用电量")
                    load_days = gr.Slider(minimum=1, maximum=7, value=1, step=1, label="预测天数")
                    load_plot = gr.Plot(label="用电预测")
        
        with gr.Tab("交易平台"):
            gr.Markdown("## 能源交易市场")
            
            with gr.Row():
                refresh_market_button = gr.Button("刷新市场")
            
            with gr.Row():
                market_listings = gr.DataFrame(
                    headers=["ID", "发电方", "电量(kWh)", "单价", "总价", "创建时间", "有效期至", "绿能认证", "信誉值"],
                    label="可用能源列表"
                )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 发布能源")
                    listing_amount = gr.Number(label="电量(kWh)")
                    listing_price = gr.Number(label="单价")
                    listing_valid_hours = gr.Slider(minimum=1, maximum=72, value=24, step=1, label="有效期(小时)")
                    create_listing_button = gr.Button("发布")
                    listing_result = gr.Textbox(label="发布结果")
                
                with gr.Column():
                    gr.Markdown("### 购买能源")
                    listing_id_input = gr.Number(label="挂单ID")
                    bid_amount = gr.Number(label="电量(kWh)")
                    place_bid_button = gr.Button("购买")
                    bid_result = gr.Textbox(label="购买结果")
            
            gr.Markdown("## 我的交易")
            refresh_transactions_button = gr.Button("刷新我的交易")
            
            with gr.Row():
                my_transactions = gr.DataFrame(
                    headers=["ID", "发电方", "用电方", "合约电量", "单价", "总价", "状态", "创建时间"],
                    label="我的交易列表"
                )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 接受购买请求")
                    accept_bid_id = gr.Number(label="投标ID")
                    accept_bid_button = gr.Button("接受")
                    accept_result = gr.Textbox(label="接受结果")
                
                with gr.Column():
                    gr.Markdown("### 确认电力交付")
                    transaction_id_input = gr.Number(label="交易ID")
                    actual_amount_input = gr.Number(label="实际交付电量(kWh)")
                    confirm_delivery_button = gr.Button("确认交付")
                    confirm_result = gr.Textbox(label="确认结果")
        
        with gr.Tab("智能合约"):
            with gr.Tabs():
                with gr.TabItem("合约参数设置"):
                    gr.Markdown("## 智能合约参数设置")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 绿能认证要求")
                            min_reputation = gr.Slider(minimum=50, maximum=100, value=85, step=1, label="最低信誉值")
                            min_active_score = gr.Slider(minimum=10, maximum=100, value=70, step=1, label="最低活跃度")
                            min_energy_production = gr.Slider(minimum=10, maximum=500, value=100, step=10, label="最低发电量(kWh)")
                        
                        with gr.Column():
                            gr.Markdown("### 交易监管参数")
                            delivery_threshold = gr.Slider(minimum=0.5, maximum=1.0, value=0.9, step=0.05, label="交付率阈值")
                            penalty_factor = gr.Slider(minimum=1, maximum=20, value=10, step=1, label="惩罚系数")
                    
                    update_contract_button = gr.Button("更新合约参数")
                    update_contract_result = gr.Textbox(label="更新结果")
                
                with gr.TabItem("自动交易策略"):
                    gr.Markdown("## 自动交易策略管理")
                    
                    with gr.Row():
                        refresh_strategy_button = gr.Button("刷新策略列表")
                    
                    with gr.Row():
                        strategies_table = gr.DataFrame(
                            headers=["索引", "类型", "描述", "能源量(kWh)", "价格范围", "有效期(小时)", "自动接受", "状态", "创建时间"],
                            label="我的交易策略"
                        )
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 创建新策略")
                            
                            strategy_type = gr.Radio(["买入策略", "卖出策略"], label="策略类型", value="买入策略")
                            strategy_description = gr.Textbox(label="策略描述", placeholder="例如：每日自动购电策略")
                            strategy_energy_amount = gr.Number(label="能源量(kWh)", value=10.0)
                            
                            with gr.Row():
                                min_price = gr.Number(label="最低价格", value=0.5)
                                max_price = gr.Number(label="最高价格", value=2.0)
                            
                            strategy_valid_hours = gr.Slider(minimum=1, maximum=72, value=24, step=1, label="有效期(小时)")
                            auto_accept = gr.Checkbox(label="自动接受报价/投标", value=True)
                            
                            create_strategy_button = gr.Button("创建策略")
                            create_strategy_result = gr.Textbox(label="创建结果")
                        
                        with gr.Column():
                            gr.Markdown("### 添加交易条件")
                            
                            strategy_index = gr.Number(label="策略索引", value=0, precision=0)
                            condition_type = gr.Dropdown(
                                ["价格条件", "时间条件", "能源量条件", "信誉度条件", "绿能认证条件", "避开高峰条件"],
                                label="条件类型",
                                value="价格条件"
                            )
                            
                            # 针对不同类型的条件，显示不同的参数设置
                            with gr.Group(visible=True) as price_condition:
                                condition_min_price = gr.Number(label="最低价格", value=0.5)
                                condition_max_price = gr.Number(label="最高价格", value=2.0)
                            
                            with gr.Group(visible=False) as time_condition:
                                condition_valid_hours = gr.Number(label="最短有效期(小时)", value=24)
                            
                            with gr.Group(visible=False) as amount_condition:
                                condition_min_amount = gr.Number(label="最小能源量(kWh)", value=1.0)
                                condition_max_amount = gr.Number(label="最大能源量(kWh)", value=50.0)
                            
                            with gr.Group(visible=False) as reputation_condition:
                                condition_min_reputation = gr.Slider(minimum=0, maximum=100, value=70, step=1, label="最低信誉值")
                            
                            with gr.Group(visible=False) as green_condition:
                                require_green = gr.Checkbox(label="要求绿能认证", value=True)
                            
                            with gr.Group(visible=False) as peak_condition:
                                peak_hours = gr.CheckboxGroup(
                                    [str(i) for i in range(24)],
                                    label="高峰时段(小时)",
                                    value=["8", "9", "10", "11", "18", "19", "20", "21"]
                                )
                            
                            add_condition_button = gr.Button("添加条件")
                            add_condition_result = gr.Textbox(label="添加结果")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 策略管理")
                            manage_strategy_index = gr.Number(label="策略索引", value=0, precision=0)
                            with gr.Row():
                                activate_strategy_button = gr.Button("激活策略")
                                deactivate_strategy_button = gr.Button("停用策略")
                            strategy_management_result = gr.Textbox(label="管理结果")
                        
                        with gr.Column():
                            gr.Markdown("### 执行自动交易")
                            execute_trading_button = gr.Button("立即执行自动交易")
                            execute_trading_result = gr.Textbox(label="执行结果")
                
                with gr.TabItem("自动交易日志"):
                    gr.Markdown("## 自动交易日志")
                    
                    with gr.Row():
                        refresh_logs_button = gr.Button("刷新日志")
                        logs_limit = gr.Number(label="显示数量", value=20, precision=0, minimum=1, maximum=100)
                    
                    with gr.Row():
                        logs_table = gr.DataFrame(
                            headers=["时间", "类型", "用户", "金额", "策略类型", "操作", "结果"],
                            label="交易日志"
                        )
        
        with gr.Tab("区块链浏览器"):
            gr.Markdown("## 区块链浏览器")
            refresh_blockchain_button = gr.Button("刷新区块链")
            
            with gr.Row():
                blockchain_info = gr.DataFrame(
                    headers=["索引", "时间戳", "前一个哈希", "数据", "哈希值"],
                    label="区块链"
                )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 区块链状态")
                    chain_length = gr.Number(label="区块链长度")
                    pending_tx_count = gr.Number(label="待处理交易数")
                    is_valid = gr.Checkbox(label="区块链有效", interactive=False)
        
        with gr.Tab("EnergyCoin"):
            gr.Markdown("## EnergyCoin加密货币管理")
            
            with gr.Row():
                refresh_coin_button = gr.Button("刷新EnergyCoin信息")
            
            with gr.Tabs():
                with gr.TabItem("挖矿与质押"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 挖矿管理")
                            with gr.Row():
                                start_mining_button = gr.Button("开始挖矿")
                                stop_mining_button = gr.Button("停止挖矿")
                            mining_status = gr.Textbox(label="挖矿状态")
                            
                            process_block_button = gr.Button("处理区块(挖矿/验证)")
                            block_result = gr.Textbox(label="区块处理结果")
                        
                        with gr.Column():
                            gr.Markdown("### 质押管理")
                            stake_amount = gr.Number(label="质押金额", value=10)
                            stake_duration = gr.Slider(minimum=7, maximum=365, value=30, step=1, label="质押期限(天)")
                            create_stake_button = gr.Button("创建质押")
                            stake_result = gr.Textbox(label="质押结果")
                            
                            release_stake_button = gr.Button("释放质押")
                            release_result = gr.Textbox(label="释放结果")
                
                with gr.TabItem("交易记录"):
                    gr.Markdown("### EnergyCoin交易记录")
                    refresh_transactions_coin_button = gr.Button("刷新交易记录")
                    
                    coin_transactions = gr.DataFrame(
                        headers=["发送方", "接收方", "金额", "类型", "时间", "状态"],
                        label="EnergyCoin交易记录"
                    )
                
                with gr.TabItem("转账与支付"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 转账EnergyCoin")
                            receiver_address = gr.Textbox(label="接收方地址")
                            transfer_amount = gr.Number(label="转账金额", value=1.0)
                            transfer_button = gr.Button("转账")
                            transfer_result = gr.Textbox(label="转账结果")
                        
                        with gr.Column():
                            gr.Markdown("### 支付电费")
                            bill_amount = gr.Number(label="电费金额(EnergyCoin)", value=1.0)
                            pay_bill_button = gr.Button("支付电费")
                            bill_result = gr.Textbox(label="支付结果")
                
                with gr.TabItem("统计信息"):
                    gr.Markdown("### EnergyCoin统计信息")
                    refresh_stats_button = gr.Button("刷新统计信息")
                    
                    with gr.Row():
                        with gr.Column():
                            total_supply = gr.Number(label="总供应量")
                            total_staked = gr.Number(label="总质押量")
                            staking_ratio = gr.Number(label="质押比例(%)")
                        
                        with gr.Column():
                            active_miners = gr.Number(label="活跃矿工数")
                            block_count = gr.Number(label="区块数量")
                            current_difficulty = gr.Number(label="当前难度")
                            
                    with gr.Row():
                        with gr.Column():
                            mining_reward = gr.Number(label="挖矿奖励")
                            staking_reward = gr.Number(label="质押奖励")
                        
                        with gr.Column():
                            coin_stats_plot = gr.Plot(label="EnergyCoin统计图表")
        
        # 注册回调函数
        
        # 用户注册与登录回调
        def register_callback(name, is_producer, is_consumer, panel_capacity, user_type, base_load):
            result = register_user(name, is_producer, is_consumer, panel_capacity, user_type, base_load)
            if result["success"]:
                return result["message"], result["user"]["address"]
            else:
                return result["message"], ""
        
        register_button.click(
            register_callback, 
            inputs=[name_input, is_producer, is_consumer, panel_capacity, user_type, base_load], 
            outputs=[register_output, user_address_output]
        )
        
        def login_callback(address):
            result = login_user(address)
            return result["message"]
        
        login_button.click(login_callback, inputs=[address_input], outputs=[login_output])
        
        # 用户信息回调
        def refresh_user_info():
            if current_user["address"]:
                info = get_user_info(current_user["address"])
                user_type_text = ""
                if info["is_producer"] and info["is_consumer"]:
                    user_type_text = "双向用户 (发电方+用电方)"
                elif info["is_producer"]:
                    user_type_text = "发电方"
                elif info["is_consumer"]:
                    user_type_text = "用电方"
                
                # 质押状态
                staking_status = "未质押"
                staked_amount_value = 0
                if info["staking_info"]:
                    stake_info = info["staking_info"]
                    staked_amount_value = stake_info["amount"]
                    if stake_info["days_remaining"] > 0:
                        staking_status = f"已质押: 剩余{stake_info['days_remaining']:.1f}天"
                    else:
                        staking_status = "可释放质押"
                    
                return (
                    info["name"],
                    info["address"],
                    user_type_text,
                    info["is_green_certified"],
                    info["reputation"],
                    info["active_score"],
                    info["green_energy_subsidy"],
                    info["energy_coin_balance"],
                    info["total_generation"],
                    info["total_consumption"],
                    info["net_energy"],
                    info["energy_coin_address"],
                    info["is_mining"],
                    staked_amount_value,
                    staking_status
                )
            else:
                return "", "", "", False, 0, 0, 0, 0, 0, 0, 0, "", False, 0, "未质押"
        
        refresh_user_info_button.click(
            refresh_user_info,
            inputs=[],
            outputs=[user_name, user_address, user_type_info, is_green_certified, 
                    reputation, active_score, green_energy_subsidy, energy_coin_balance,
                    total_generation, total_consumption, net_energy, 
                    energy_coin_address, is_mining, staked_amount, stake_status]
        )
        
        def apply_certification_callback():
            if current_user["address"]:
                result = apply_for_green_certification(current_user["address"])
                return result["message"]
            else:
                return "请先登录"
        
        apply_certification_button.click(apply_certification_callback, inputs=[], outputs=[certification_result])
        
        # 数据可视化回调
        def show_data_callback(days):
            if not current_user["address"]:
                return None
                
            user_data = data_collector.get_user_data(current_user["address"])
            if user_data.empty:
                return None
                
            # 转换时间戳
            user_data["datetime"] = pd.to_datetime(user_data["timestamp"])
            
            # 获取最近几天的数据
            recent_data = user_data.sort_values("datetime").tail(days * 24 * 4) # 假设每15分钟一个数据点
            
            # 创建可视化图表
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 添加发电量
            fig.add_trace(
                go.Scatter(
                    x=recent_data["datetime"],
                    y=recent_data["solar_power"],
                    name="发电量",
                    line=dict(color='orange')
                ),
                secondary_y=False
            )
            
            # 添加用电量
            fig.add_trace(
                go.Scatter(
                    x=recent_data["datetime"],
                    y=recent_data["load"],
                    name="用电量",
                    line=dict(color='blue')
                ),
                secondary_y=False
            )
            
            # 添加温度数据
            fig.add_trace(
                go.Scatter(
                    x=recent_data["datetime"],
                    y=recent_data["temperature"],
                    name="温度",
                    line=dict(color='red')
                ),
                secondary_y=True
            )
            
            # 更新布局
            fig.update_layout(
                title="能源数据可视化",
                xaxis_title="时间",
                legend=dict(x=0, y=1, traceorder="normal"),
                hovermode="x"
            )
            
            fig.update_yaxes(title_text="功率 (kW)", secondary_y=False)
            fig.update_yaxes(title_text="温度 (°C)", secondary_y=True)
            
            return fig
        
        show_data_button.click(show_data_callback, inputs=[data_days], outputs=[energy_plot])
        
        def predict_generation_callback(days):
            if not current_user["address"]:
                return None
                
            result = predict_solar_generation(current_user["address"], days)
            if result["success"]:
                return result["plot"]
            else:
                return None
        
        predict_generation_button.click(
            predict_generation_callback,
            inputs=[generation_days],
            outputs=[generation_plot]
        )
        
        def predict_load_callback(days):
            if not current_user["address"]:
                return None
                
            result = predict_load(current_user["address"], days)
            if result["success"]:
                return result["plot"]
            else:
                return None
        
        predict_load_button.click(
            predict_load_callback,
            inputs=[load_days],
            outputs=[load_plot]
        )
        
        # 交易平台回调
        def refresh_market_callback():
            listings = get_active_listings()
            market_data = []
            
            for listing in listings:
                market_data.append([
                    listing["id"],
                    listing["producer_name"],
                    listing["amount"],
                    listing["price_per_kwh"],
                    listing["total_price"],
                    listing["created_at"],
                    listing["valid_until"],
                    "是" if listing["green_certified"] else "否",
                    round(listing["producer_reputation"], 2)
                ])
                
            return market_data
        
        refresh_market_button.click(refresh_market_callback, inputs=[], outputs=[market_listings])
        
        def create_listing_callback(amount, price, valid_hours):
            if not current_user["address"]:
                return "请先登录"
                
            user = smart_contract.users[current_user["address"]]
            if not user.is_producer:
                return "只有发电方可以发布能源挂单"
                
            result = create_energy_listing(current_user["address"], amount, price, valid_hours)
            return result["message"]
        
        create_listing_button.click(
            create_listing_callback,
            inputs=[listing_amount, listing_price, listing_valid_hours],
            outputs=[listing_result]
        )
        
        def place_bid_callback(listing_id, amount):
            if not current_user["address"]:
                return "请先登录"
                
            user = smart_contract.users[current_user["address"]]
            if not user.is_consumer:
                return "只有用电方可以购买能源"
                
            result = place_energy_bid(current_user["address"], int(listing_id), amount)
            return result["message"]
        
        place_bid_button.click(
            place_bid_callback,
            inputs=[listing_id_input, bid_amount],
            outputs=[bid_result]
        )
        
        def refresh_transactions_callback():
            if not current_user["address"]:
                return []
                
            transactions = get_user_transactions(current_user["address"])
            tx_data = []
            
            for tx in transactions:
                tx_data.append([
                    tx["id"],
                    tx["producer_name"],
                    tx["consumer_name"],
                    tx["amount"],
                    tx["price_per_kwh"],
                    tx["total_price"],
                    tx["status"],
                    tx["created_at"]
                ])
                
            return tx_data
        
        refresh_transactions_button.click(
            refresh_transactions_callback,
            inputs=[],
            outputs=[my_transactions]
        )
        
        def accept_bid_callback(bid_id):
            if not current_user["address"]:
                return "请先登录"
                
            user = smart_contract.users[current_user["address"]]
            if not user.is_producer:
                return "只有发电方可以接受投标"
                
            result = accept_energy_bid(current_user["address"], int(bid_id))
            return result["message"]
        
        accept_bid_button.click(
            accept_bid_callback,
            inputs=[accept_bid_id],
            outputs=[accept_result]
        )
        
        def confirm_delivery_callback(transaction_id, actual_amount):
            if not current_user["address"]:
                return "请先登录"
                
            result = confirm_energy_delivery(int(transaction_id), actual_amount)
            return result["message"]
        
        confirm_delivery_button.click(
            confirm_delivery_callback,
            inputs=[transaction_id_input, actual_amount_input],
            outputs=[confirm_result]
        )
        
        # 智能合约回调
        def update_contract_parameters(min_rep, min_active, min_energy, threshold, penalty):
            # 更新绿能认证要求
            smart_contract.green_certification_requirements = {
                "min_reputation": min_rep,
                "min_active_score": min_active,
                "min_energy_production": min_energy
            }
            
            # 由于惩罚逻辑硬编码在 confirm_energy_delivery 方法中，
            # 这里只是提供一个演示，实际上不会修改该逻辑
            
            return "智能合约参数已更新"
        
        update_contract_button.click(
            update_contract_parameters,
            inputs=[min_reputation, min_active_score, min_energy_production, delivery_threshold, penalty_factor],
            outputs=[update_contract_result]
        )
        
        # 自动交易策略回调
        def refresh_strategies():
            if not current_user["address"]:
                return []
                
            strategies = get_user_strategies(current_user["address"])
            strategy_data = []
            
            for strategy in strategies:
                strategy_data.append([
                    strategy["index"],
                    strategy["type"],
                    strategy["description"],
                    strategy["energy_amount"] or "--",
                    strategy["price_range"],
                    strategy["valid_hours"],
                    strategy["auto_accept"],
                    strategy["status"],
                    strategy["created_at"]
                ])
                
            return strategy_data
        
        refresh_strategy_button.click(
            refresh_strategies,
            inputs=[],
            outputs=[strategies_table]
        )
        
        def create_strategy(type_str, description, energy, min_p, max_p, valid_hours, auto_accept):
            if not current_user["address"]:
                return "请先登录"
                
            is_buy = type_str == "买入策略"
            
            result = create_auto_trading_strategy(
                current_user["address"],
                is_buy,
                description,
                energy,
                max_p if is_buy else None,
                None if is_buy else min_p,
                valid_hours,
                auto_accept
            )
            
            return result["message"]
        
        create_strategy_button.click(
            create_strategy,
            inputs=[strategy_type, strategy_description, strategy_energy_amount, min_price, max_price, strategy_valid_hours, auto_accept],
            outputs=[create_strategy_result]
        )
        
        # 条件类型变更时切换对应的参数界面
        def switch_condition_params(condition_type_str):
            is_price = condition_type_str == "价格条件"
            is_time = condition_type_str == "时间条件"
            is_amount = condition_type_str == "能源量条件"
            is_reputation = condition_type_str == "信誉度条件"
            is_green = condition_type_str == "绿能认证条件"
            is_peak = condition_type_str == "避开高峰条件"
            
            return {
                price_condition: is_price,
                time_condition: is_time,
                amount_condition: is_amount,
                reputation_condition: is_reputation,
                green_condition: is_green,
                peak_condition: is_peak
            }
        
        condition_type.change(
            switch_condition_params,
            inputs=[condition_type],
            outputs=[price_condition, time_condition, amount_condition, reputation_condition, green_condition, peak_condition]
        )
        
        # 添加条件回调
        def add_condition_callback(strategy_idx, condition_type_str, min_price, max_price, valid_hours, min_amount, max_amount, min_reputation, require_green, peak_hours_strs):
            if not current_user["address"]:
                return "请先登录"
            
            # 根据条件类型准备参数
            condition_type_map = {
                "价格条件": "price",
                "时间条件": "time",
                "能源量条件": "amount",
                "信誉度条件": "reputation",
                "绿能认证条件": "green",
                "避开高峰条件": "peak"
            }
            
            condition_type_code = condition_type_map.get(condition_type_str)
            if not condition_type_code:
                return f"不支持的条件类型: {condition_type_str}"
                
            # 准备条件参数
            condition_params = {}
            if condition_type_code == "price":
                condition_params = {"min_price": min_price, "max_price": max_price}
            elif condition_type_code == "time":
                condition_params = {"valid_hours": valid_hours}
            elif condition_type_code == "amount":
                condition_params = {"min_amount": min_amount, "max_amount": max_amount}
            elif condition_type_code == "reputation":
                condition_params = {"min_reputation": min_reputation}
            elif condition_type_code == "green":
                condition_params = {"require_certified": require_green}
            elif condition_type_code == "peak":
                # 转换为小时数字列表
                peak_hours_list = [int(h) for h in peak_hours_strs]
                condition_params = {"peak_hours": peak_hours_list}
            
            # 添加条件
            result = add_strategy_condition(
                current_user["address"],
                int(strategy_idx),
                condition_type_code,
                condition_params
            )
            
            return result["message"]
        
        add_condition_button.click(
            add_condition_callback,
            inputs=[
                strategy_index, condition_type,
                condition_min_price, condition_max_price,
                condition_valid_hours,
                condition_min_amount, condition_max_amount,
                condition_min_reputation,
                require_green,
                peak_hours
            ],
            outputs=[add_condition_result]
        )
        
        # 策略管理回调
        def activate_strategy_callback(strategy_idx):
            if not current_user["address"]:
                return "请先登录"
                
            result = activate_trading_strategy(
                current_user["address"],
                int(strategy_idx),
                True
            )
            
            return result["message"]
        
        activate_strategy_button.click(
            activate_strategy_callback,
            inputs=[manage_strategy_index],
            outputs=[strategy_management_result]
        )
        
        def deactivate_strategy_callback(strategy_idx):
            if not current_user["address"]:
                return "请先登录"
                
            result = activate_trading_strategy(
                current_user["address"],
                int(strategy_idx),
                False
            )
            
            return result["message"]
        
        deactivate_strategy_button.click(
            deactivate_strategy_callback,
            inputs=[manage_strategy_index],
            outputs=[strategy_management_result]
        )
        
        # 执行自动交易回调
        def execute_auto_trading_callback():
            if not current_user["address"]:
                return "请先登录"
                
            result = execute_auto_trading()
            return result["message"]
        
        execute_trading_button.click(
            execute_auto_trading_callback,
            inputs=[],
            outputs=[execute_trading_result]
        )
        
        # 交易日志回调
        def refresh_trading_logs(limit):
            if not current_user["address"]:
                return []
                
            logs = get_auto_trading_logs(current_user["address"], int(limit))
            logs_data = []
            
            for log in logs:
                log_row = [
                    log.get("timestamp", ""),
                    log.get("type", ""),
                    log.get("user", log.get("producer", log.get("consumer", ""))),
                    log.get("amount", ""),
                    log.get("strategy_type", ""),
                    log.get("description", ""),
                    log.get("error", "成功")
                ]
                logs_data.append(log_row)
                
            return logs_data
        
        refresh_logs_button.click(
            refresh_trading_logs,
            inputs=[logs_limit],
            outputs=[logs_table]
        )
        
        # 区块链浏览器回调
        def refresh_blockchain_callback():
            blockchain_data = []
            
            for block in blockchain.chain:
                block_dict = block.to_dict()
                blockchain_data.append([
                    block_dict["index"],
                    datetime.datetime.fromtimestamp(block_dict["timestamp"]).strftime('%Y-%m-%d %H:%M:%S'),
                    block_dict["previous_hash"],
                    str(block_dict["transactions"])[:50] + "..." if len(str(block_dict["transactions"])) > 50 else str(block_dict["transactions"]),
                    block_dict["hash"][:15] + "..."
                ])
                
            # 区块链状态
            chain_len = len(blockchain.chain)
            pending_count = len(blockchain.pending_transactions)
            is_chain_valid = blockchain.is_chain_valid()
            
            return blockchain_data, chain_len, pending_count, is_chain_valid
        
        refresh_blockchain_button.click(
            refresh_blockchain_callback,
            inputs=[],
            outputs=[blockchain_info, chain_length, pending_tx_count, is_valid]
        )
    
    return app


# 主入口点
if __name__ == "__main__":
    print("正在启动区块链分布式光伏发电点对点交易系统...")
    app = create_gradio_app()
    app.launch(share=True)
    print("系统已成功启动！")


# 创建自动交易策略
def create_auto_trading_strategy(
    user_address: str, is_buy_strategy: bool, description: str, 
    energy_amount: float = None, max_price_per_kwh: float = None, 
    min_price_per_kwh: float = None, valid_hours: int = 24, 
    auto_accept_offer: bool = False
) -> Dict[str, Any]:
    """
    创建自动交易策略
    
    Args:
        user_address: 用户地址
        is_buy_strategy: 是否为买入策略
        description: 策略描述
        energy_amount: 目标交易能源数量
        max_price_per_kwh: 最高接受价格
        min_price_per_kwh: 最低接受价格
        valid_hours: 挂单有效期(小时)
        auto_accept_offer: 是否自动接受报价
        
    Returns:
        创建结果
    """
    try:
        strategy = smart_contract.create_trading_strategy(
            user_address, is_buy_strategy, description,
            energy_amount, max_price_per_kwh, min_price_per_kwh,
            valid_hours, auto_accept_offer
        )
        
        return {
            "success": True,
            "message": f"{'买入' if is_buy_strategy else '卖出'}策略创建成功",
            "strategy": strategy
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"创建自动交易策略失败: {str(e)}"
        }


# 添加交易条件
def add_strategy_condition(user_address: str, strategy_index: int, condition_type: str, 
                         condition_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    添加交易条件到策略
    
    Args:
        user_address: 用户地址
        strategy_index: 策略索引
        condition_type: 条件类型
        condition_params: 条件参数
        
    Returns:
        添加结果
    """
    try:
        if user_address not in smart_contract.users:
            return {
                "success": False,
                "message": "用户不存在"
            }
            
        user = smart_contract.users[user_address]
        if strategy_index >= len(user.trading_strategies):
            return {
                "success": False,
                "message": "策略不存在"
            }
            
        strategy = user.trading_strategies[strategy_index]
        
        # 根据条件类型创建对应的条件对象
        from blockchain.smart_contract import (
            TradingConditionType, PriceCondition, TimeCondition, 
            EnergyAmountCondition, ReputationCondition,
            GreenCertifiedCondition, PeakAvoidanceCondition
        )
        
        condition = None
        if condition_type == "price":
            condition = PriceCondition(
                condition_type=TradingConditionType.PRICE_THRESHOLD,
                description="",
                max_price_per_kwh=condition_params.get("max_price", 10.0),
                min_price_per_kwh=condition_params.get("min_price", 0.0)
            )
        elif condition_type == "time":
            condition = TimeCondition(
                condition_type=TradingConditionType.TIME_CONSTRAINT,
                description="",
                valid_hours=condition_params.get("valid_hours", 24)
            )
        elif condition_type == "amount":
            condition = EnergyAmountCondition(
                condition_type=TradingConditionType.ENERGY_AMOUNT,
                description="",
                min_amount=condition_params.get("min_amount", 0.0),
                max_amount=condition_params.get("max_amount", 100.0)
            )
        elif condition_type == "reputation":
            condition = ReputationCondition(
                condition_type=TradingConditionType.REPUTATION_THRESHOLD,
                description="",
                min_reputation=condition_params.get("min_reputation", 70.0)
            )
        elif condition_type == "green":
            condition = GreenCertifiedCondition(
                condition_type=TradingConditionType.GREEN_CERTIFIED,
                description="",
                require_certified=condition_params.get("require_certified", True)
            )
        elif condition_type == "peak":
            condition = PeakAvoidanceCondition(
                condition_type=TradingConditionType.PEAK_AVOIDANCE,
                description="",
                peak_hours=condition_params.get("peak_hours", [8, 9, 10, 11, 18, 19, 20, 21])
            )
        else:
            return {
                "success": False,
                "message": f"不支持的条件类型: {condition_type}"
            }
        
        updated_strategy = smart_contract.add_strategy_condition(user_address, strategy, condition)
        
        return {
            "success": True,
            "message": f"条件添加成功: {condition.description}",
            "strategy": updated_strategy
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"添加条件失败: {str(e)}"
        }


# 激活或停用策略
def activate_trading_strategy(user_address: str, strategy_index: int, activate: bool) -> Dict[str, Any]:
    """
    激活或停用交易策略
    
    Args:
        user_address: 用户地址
        strategy_index: 策略索引
        activate: 是否激活
        
    Returns:
        操作结果
    """
    try:
        if user_address not in smart_contract.users:
            return {
                "success": False,
                "message": "用户不存在"
            }
            
        user = smart_contract.users[user_address]
        if strategy_index >= len(user.trading_strategies):
            return {
                "success": False,
                "message": "策略不存在"
            }
            
        strategy = user.trading_strategies[strategy_index]
        updated_strategy = smart_contract.activate_strategy(user_address, strategy, activate)
        
        return {
            "success": True,
            "message": f"策略已{'激活' if activate else '停用'}",
            "strategy": updated_strategy
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"操作失败: {str(e)}"
        }


# 获取用户的所有交易策略
def get_user_strategies(user_address: str) -> List[Dict[str, Any]]:
    """
    获取用户的所有交易策略
    
    Args:
        user_address: 用户地址
        
    Returns:
        策略列表
    """
    if user_address not in smart_contract.users:
        return []
        
    user = smart_contract.users[user_address]
    strategies = []
    
    for idx, strategy in enumerate(user.trading_strategies):
        strategy_data = {
            "index": idx,
            "type": "买入" if strategy.is_buy_strategy else "卖出",
            "description": strategy.description,
            "energy_amount": strategy.energy_amount,
            "price_range": f"{strategy.min_price_per_kwh or 0.0} - {strategy.max_price_per_kwh or '不限'}",
            "valid_hours": strategy.valid_hours,
            "auto_accept": "是" if strategy.auto_accept_offer else "否",
            "status": "激活" if strategy.is_active else "停用",
            "conditions": [condition.description for condition in strategy.conditions],
            "created_at": datetime.datetime.fromtimestamp(strategy.created_at).strftime('%Y-%m-%d %H:%M:%S')
        }
        strategies.append(strategy_data)
    
    return strategies


# 执行自动交易
def execute_auto_trading() -> Dict[str, Any]:
    """
    执行自动交易
    
    Returns:
        执行结果
    """
    try:
        executed_transactions = smart_contract.execute_auto_trading()
        
        return {
            "success": True,
            "message": f"自动执行了 {len(executed_transactions)} 笔交易",
            "transactions": executed_transactions
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"自动交易执行失败: {str(e)}"
        }


# 获取自动交易日志
def get_auto_trading_logs(user_address: str = None, limit: int = 20) -> List[Dict[str, Any]]:
    """
    获取自动交易日志
    
    Args:
        user_address: 用户地址
        limit: 日志数量限制
        
    Returns:
        日志列表
    """
    logs = smart_contract.get_auto_trading_logs(user_address, limit)
    
    # 格式化日志
    formatted_logs = []
    for log in logs:
        formatted_log = {
            "type": log["type"],
            "timestamp": datetime.datetime.fromtimestamp(log["timestamp"]).strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        # 添加用户相关信息
        if "user_address" in log:
            user = smart_contract.users.get(log["user_address"])
            formatted_log["user"] = user.name if user else log["user_address"]
            
        if "producer_address" in log:
            producer = smart_contract.users.get(log["producer_address"])
            formatted_log["producer"] = producer.name if producer else log["producer_address"]
            
        if "consumer_address" in log:
            consumer = smart_contract.users.get(log["consumer_address"])
            formatted_log["consumer"] = consumer.name if consumer else log["consumer_address"]
        
        # 添加交易相关信息
        if "strategy_type" in log:
            formatted_log["strategy_type"] = "买入" if log["strategy_type"] == "buy" else "卖出"
            
        if "description" in log:
            formatted_log["description"] = log["description"]
            
        if "amount" in log:
            formatted_log["amount"] = f"{log['amount']} kWh"
            
        if "price_per_kwh" in log:
            formatted_log["price"] = f"{log['price_per_kwh']} 元/kWh"
            
        if "error_message" in log:
            formatted_log["error"] = log["error_message"]
        
        formatted_logs.append(formatted_log)
