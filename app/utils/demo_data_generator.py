import os
import sys
import time
import datetime
import random
import pandas as pd
import numpy as np
import hashlib
import json
from typing import Dict, List, Any, Tuple

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入项目模块
from blockchain.core import Blockchain, Block
from blockchain.smart_contract import EnergyTradingContract, User
from blockchain.energy_coin import EnergyCoin
from utils.data_collector import EnergyDataGenerator, EnergyDataCollector
from utils.encryption import DataEncryptor


class DemoDataGenerator:
    """演示数据生成器，用于创建完整的系统演示数据"""
    
    def __init__(self, seed: int = 42):
        """
        初始化演示数据生成器
        
        Args:
            seed: 随机种子，保证生成数据的一致性
        """
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 创建系统组件
        self.blockchain = Blockchain(difficulty=2)  # 降低难度以加快挖矿速度
        self.smart_contract = EnergyTradingContract()
        self.data_collector = EnergyDataCollector(EnergyDataGenerator(seed=seed))
        self.data_encryptor = DataEncryptor()
        
        # 用户信息
        self.users = []
        
        # 创建数据目录
        os.makedirs("data", exist_ok=True)
    
    def generate_address(self) -> str:
        """生成随机用户地址"""
        random_bytes = os.urandom(20)
        return hashlib.sha256(random_bytes).hexdigest()[:40]
    
    def create_demo_users(self) -> List[Dict[str, Any]]:
        """
        创建演示用户
        
        Returns:
            用户信息列表
        """
        # 用户配置
        user_configs = [
            # 发电方
            {
                "name": "太阳能发电公司",
                "is_producer": True,
                "is_consumer": False,
                "panel_capacity": 50.0,
                "user_type": "工业",
                "base_load": 10.0,
                "location": "北京",
                "is_miner": True,  # 参与EnergyCoin挖矿
                "staking_amount": 200.0,  # EnergyCoin质押金额
                "staking_days": 30  # 质押天数
            },
            {
                "name": "绿色能源集团",
                "is_producer": True,
                "is_consumer": False,
                "panel_capacity": 80.0,
                "user_type": "工业",
                "base_load": 15.0,
                "location": "上海",
                "is_miner": True,
                "staking_amount": 300.0,
                "staking_days": 60
            },
            # 用电方
            {
                "name": "星城小区",
                "is_producer": False,
                "is_consumer": True,
                "panel_capacity": 0.0,
                "user_type": "住宅",
                "base_load": 20.0,
                "location": "北京",
                "is_miner": False
            },
            {
                "name": "城市商场",
                "is_producer": False,
                "is_consumer": True,
                "panel_capacity": 0.0,
                "user_type": "商业",
                "base_load": 30.0,
                "location": "广州",
                "is_miner": True,
                "staking_amount": 100.0,
                "staking_days": 14
            },
            # 双向用户（既发电又用电）
            {
                "name": "绿色农场",
                "is_producer": True,
                "is_consumer": True,
                "panel_capacity": 15.0,
                "user_type": "商业",
                "base_load": 8.0,
                "location": "成都",
                "is_miner": True,
                "staking_amount": 50.0,
                "staking_days": 7
            },
            {
                "name": "创新科技园",
                "is_producer": True,
                "is_consumer": True,
                "panel_capacity": 25.0,
                "user_type": "工业",
                "base_load": 18.0,
                "location": "上海",
                "is_miner": False
            }
        ]
        
        # 创建用户
        for config in user_configs:
            address = self.generate_address()
            
            # 在智能合约中注册用户
            user = self.smart_contract.register_user(
                address, 
                config["name"], 
                config["is_producer"], 
                config["is_consumer"]
            )
            
            # 生成用户密钥对
            public_key = self.data_encryptor.generate_user_keypair(address)
            
            # 提取用户配置
            user_config = {
                "panel_capacity": config["panel_capacity"],
                "user_type": config["user_type"],
                "base_load": config["base_load"],
                "location": config.get("location", "北京")
            }
            
            # 生成样本历史数据
            self.data_collector.generate_sample_data(address, user_config, days_back=60)
            
            # 修改用户信誉和活跃度
            user.active_score = random.uniform(50, 100)
            
            if config["is_producer"]:
                # 给一些发电用户添加绿能认证
                if random.random() > 0.5:
                    user.is_green_certified = True
                    user.green_energy_subsidy = random.uniform(50, 200)
            
            # EnergyCoin相关配置
            if config.get("is_miner", False):
                user.is_mining = True
                self.smart_contract.energy_coin.add_miner(user.energy_coin_address)
                
                # 为用户增加一些EnergyCoin初始余额
                init_balance = random.uniform(20, 100)
                self.smart_contract.energy_coin.add_transaction(
                    from_address="system",
                    to_address=user.energy_coin_address,
                    amount=init_balance,
                    tx_type="initial_grant"
                )
            
            # 如果配置了质押，创建质押
            if config.get("staking_amount", 0) > 0 and config.get("staking_days", 0) > 0:
                stake_amount = config["staking_amount"]
                stake_days = config["staking_days"]
                
                try:
                    # 确保用户有足够余额质押
                    current_balance = self.smart_contract.energy_coin.get_balance(user.energy_coin_address)
                    if current_balance < stake_amount:
                        # 追加发放代币 - 确保发放足够数量
                        self.smart_contract.energy_coin.add_transaction(
                            from_address="system",
                            to_address=user.energy_coin_address,
                            amount=stake_amount - current_balance + 50,  # 多加50个作为缓冲
                            tx_type="additional_grant"
                        )
                        # 处理交易确保余额更新
                        self.smart_contract.energy_coin.process_block(user.energy_coin_address)
                    
                    # 创建质押
                    self.smart_contract.create_stake(address, stake_amount, stake_days)
                except Exception as e:
                    print(f"为用户 {config['name']} 创建质押时出错: {str(e)}")
            
            # 保存用户信息
            user_info = {
                "address": address,
                "name": config["name"],
                "is_producer": config["is_producer"],
                "is_consumer": config["is_consumer"],
                "panel_capacity": config["panel_capacity"],
                "user_type": config["user_type"],
                "base_load": config["base_load"],
                "location": config.get("location", "北京"),
                "active_score": user.active_score,
                "reputation": user.reputation,
                "is_green_certified": user.is_green_certified,
                "green_energy_subsidy": user.green_energy_subsidy,
                "energy_coin_address": user.energy_coin_address,
                "energy_coin_balance": self.smart_contract.energy_coin.get_balance(user.energy_coin_address),
                "is_mining": user.is_mining,
                "staking_info": self.smart_contract.energy_coin.get_staking_info(user.energy_coin_address)
            }
            
            self.users.append(user_info)
        
        return self.users
    
    def create_energy_listings(self) -> List[Dict[str, Any]]:
        """
        创建能源挂单
        
        Returns:
            挂单信息列表
        """
        listings = []
        
        # 只有发电方可以创建挂单
        producer_users = [u for u in self.users if u["is_producer"]]
        
        for producer in producer_users:
            # 创建1-3个挂单
            for _ in range(random.randint(1, 3)):
                amount = random.uniform(5, 20)
                price_per_kwh = random.uniform(0.5, 2.0)
                hours = random.randint(12, 72)
                
                valid_until = time.time() + hours * 3600
                
                # 创建挂单
                listing = self.smart_contract.list_energy_for_sale(
                    producer["address"],
                    amount,
                    price_per_kwh,
                    valid_until
                )
                
                # 记录交易到区块链
                transaction_data = {
                    "type": "listing",
                    "producer": producer["address"],
                    "amount": amount,
                    "price_per_kwh": price_per_kwh,
                    "listing_id": listing["id"],
                    "timestamp": time.time()
                }
                self.blockchain.add_transaction(transaction_data)
                
                listings.append(listing)
        
        return listings
    
    def create_energy_bids_and_transactions(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        创建能源投标和交易
        
        Returns:
            投标信息列表, 交易信息列表
        """
        bids = []
        transactions = []
        
        # 只有用电方可以投标
        consumer_users = [u for u in self.users if u["is_consumer"]]
        
        # 遍历活跃挂单
        for listing in self.smart_contract.energy_listings:
            # 跳过非活跃挂单
            if listing["status"] != "active" or listing["valid_until"] < time.time():
                continue
                
            # 随机选择是否有人投标
            if random.random() < 0.7:  # 70%概率有投标
                # 随机选择一个消费者
                consumer = random.choice(consumer_users)
                
                # 投标的电量（不超过挂单数量）
                amount = min(listing["amount"], random.uniform(1, listing["amount"]))
                
                # 创建投标
                try:
                    bid = self.smart_contract.place_bid(
                        consumer["address"],
                        listing["id"],
                        amount
                    )
                    
                    # 记录交易到区块链
                    transaction_data = {
                        "type": "bid",
                        "consumer": consumer["address"],
                        "listing_id": listing["id"],
                        "amount": amount,
                        "bid_id": bid["id"],
                        "timestamp": time.time()
                    }
                    self.blockchain.add_transaction(transaction_data)
                    
                    bids.append(bid)
                    
                    # 随机决定是否接受投标
                    if random.random() < 0.8:  # 80%概率接受投标
                        try:
                            # 获取发电方地址
                            producer_address = listing["producer_address"]
                            
                            # 接受投标
                            transaction = self.smart_contract.accept_bid(
                                producer_address,
                                bid["id"]
                            )
                            
                            # 记录交易到区块链
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
                            self.blockchain.add_transaction(transaction_data)
                            
                            # 挖矿生成新区块
                            self.blockchain.mine_pending_transactions(producer_address)
                            
                            # 随机决定是否确认交付
                            if random.random() < 0.9:  # 90%概率确认交付
                                # 随机生成实际交付电量（正常情况下接近合约电量）
                                if random.random() < 0.8:  # 80%概率完美交付
                                    actual_amount = transaction["amount"]
                                else:  # 20%概率部分交付
                                    actual_amount = transaction["amount"] * random.uniform(0.7, 0.95)
                                
                                # 确认交付
                                self.smart_contract.confirm_energy_delivery(
                                    transaction["id"],
                                    actual_amount
                                )
                                
                                # 记录确认到区块链
                                confirmation_data = {
                                    "type": "energy_delivery",
                                    "transaction_id": transaction["id"],
                                    "contracted_amount": transaction["amount"],
                                    "actual_amount": actual_amount,
                                    "delivery_ratio": actual_amount / transaction["amount"],
                                    "timestamp": time.time()
                                }
                                self.blockchain.add_transaction(confirmation_data)
                                self.blockchain.mine_pending_transactions(producer_address)
                            
                            transactions.append(transaction)
                        except Exception as e:
                            print(f"接受投标时出错: {e}")
                except Exception as e:
                    print(f"投标时出错: {e}")
        
        return bids, transactions
    
    def mine_blockchain(self) -> None:
        """挖掘区块链中的所有待处理交易"""
        # 随机选择一个用户作为矿工
        miner_address = random.choice(self.users)["address"]
        
        # 如果有待处理的交易，进行挖矿
        if len(self.blockchain.pending_transactions) > 0:
            self.blockchain.mine_pending_transactions(miner_address)
    
    def apply_green_certifications(self) -> None:
        """为符合条件的发电方申请绿能认证"""
        for user_info in self.users:
            if user_info["is_producer"]:
                address = user_info["address"]
                
                # 获取用户数据
                user_data = self.data_collector.get_user_data(address)
                
                if not user_data.empty:
                    # 计算总发电量
                    total_production = user_data["solar_power"].sum()
                    
                    # 降低获得认证的门槛以便演示
                    self.smart_contract.green_certification_requirements = {
                        "min_reputation": 60.0,
                        "min_active_score": 40.0,
                        "min_energy_production": 50.0
                    }
                    
                    try:
                        # 申请认证
                        is_certified = self.smart_contract.apply_for_green_certification(
                            address, total_production
                        )
                        
                        # 更新用户信息
                        user_info["is_green_certified"] = is_certified
                        if is_certified:
                            user_info["green_energy_subsidy"] = self.smart_contract.users[address].green_energy_subsidy
                    except Exception as e:
                        print(f"申请绿能认证时出错: {e}")
    
    def generate_energy_coin_transactions(self) -> List[Dict[str, Any]]:
        """
        生成EnergyCoin交易数据
        
        Returns:
            交易记录列表
        """
        transactions = []
        
        # 用户之间的转账交易
        for _ in range(random.randint(5, 15)):
            # 随机选择两个不同的用户
            sender_idx = random.randint(0, len(self.users) - 1)
            receiver_idx = random.randint(0, len(self.users) - 1)
            while receiver_idx == sender_idx:
                receiver_idx = random.randint(0, len(self.users) - 1)
            
            sender = self.users[sender_idx]
            receiver = self.users[receiver_idx]
            
            # 获取发送方余额
            sender_balance = self.smart_contract.get_energy_coin_balance(sender["address"])
            
            if sender_balance > 1:  # 确保有足够的余额
                # 随机转账金额
                amount = random.uniform(0.1, sender_balance / 2)
                
                try:
                    # 创建转账交易
                    tx = self.smart_contract.transfer_energy_coin(
                        sender["address"],
                        receiver["address"],
                        amount
                    )
                    
                    transactions.append(tx)
                except Exception as e:
                    print(f"创建EnergyCoin转账时出错: {str(e)}")
        
        # 电费支付交易
        for _ in range(random.randint(3, 8)):
            # 随机选择一个用户
            user_idx = random.randint(0, len(self.users) - 1)
            user = self.users[user_idx]
            
            # 获取用户余额
            user_balance = self.smart_contract.get_energy_coin_balance(user["address"])
            
            if user_balance > 1:  # 确保有足够的余额
                # 随机支付金额
                amount = random.uniform(0.1, user_balance / 3)
                
                try:
                    # 创建支付电费交易
                    tx = self.smart_contract.pay_energy_bill(
                        user["address"],
                        amount
                    )
                    
                    transactions.append(tx)
                except Exception as e:
                    print(f"创建EnergyCoin电费支付时出错: {str(e)}")
        
        # 处理EnergyCoin交易
        self.process_energy_coin_blocks()
        
        return transactions
    
    def process_energy_coin_blocks(self) -> List[Dict[str, Any]]:
        """
        处理EnergyCoin区块
        
        Returns:
            处理的区块信息列表
        """
        blocks = []
        
        # 查找正在挖矿的用户
        mining_users = [u for u in self.users if u.get("is_mining", False)]
        
        if not mining_users:
            return blocks
        
        # 确保有一些交易等待处理
        if len(self.smart_contract.energy_coin.pending_transactions) == 0:
            # 创建一些系统交易
            for _ in range(3):
                random_user = random.choice(self.users)
                self.smart_contract.energy_coin.add_transaction(
                    "system",
                    random_user["energy_coin_address"],
                    random.uniform(1, 5),
                    "system_reward"
                )
        
        # 处理多个区块
        for _ in range(random.randint(2, 5)):
            try:
                # 随机选择一个矿工
                miner = random.choice(mining_users)
                
                # 处理区块
                block = self.smart_contract.process_block(miner["address"])
                blocks.append(block)
            except Exception as e:
                print(f"处理EnergyCoin区块时出错: {str(e)}")
        
        return blocks
    
    def generate_complete_demo_data(self) -> Dict[str, Any]:
        """
        生成完整的演示数据
        
        Returns:
            包含所有演示数据的字典
        """
        # 1. 创建用户
        print("创建演示用户...")
        users = self.create_demo_users()
        
        # 2. 应用绿能认证
        print("申请绿能认证...")
        self.apply_green_certifications()
        
        # 3. 创建能源挂单
        print("创建能源挂单...")
        listings = self.create_energy_listings()
        
        # 4. 创建能源投标和交易
        print("创建能源投标和交易...")
        bids, transactions = self.create_energy_bids_and_transactions()
        
        # 5. 生成EnergyCoin交易
        print("生成EnergyCoin交易...")
        energy_coin_transactions = self.generate_energy_coin_transactions()
        
        # 6. 确保所有交易被挖矿确认
        print("完成区块链挖矿...")
        self.mine_blockchain()
        
        # 7. 再处理一些EnergyCoin区块
        print("处理EnergyCoin区块...")
        energy_coin_blocks = self.process_energy_coin_blocks()
        
        # 更新用户EnergyCoin余额信息
        for user_info in self.users:
            user_address = user_info["address"]
            user = self.smart_contract.users[user_address]
            user_info["energy_coin_balance"] = self.smart_contract.get_energy_coin_balance(user_address)
            user_info["staking_info"] = self.smart_contract.energy_coin.get_staking_info(user.energy_coin_address)
        
        # 8. 获取EnergyCoin统计信息
        energy_coin_stats = self.smart_contract.get_energy_coin_stats()
        
        # 整合所有演示数据
        demo_data = {
            "users": users,
            "listings": listings,
            "bids": bids,
            "transactions": transactions,
            "blockchain": {
                "chain_length": len(self.blockchain.chain),
                "blocks": [block.to_dict() for block in self.blockchain.chain]
            },
            "smart_contract": {
                "users_count": len(self.smart_contract.users),
                "completed_trades": len(self.smart_contract.completed_trades)
            },
            "energy_coin": {
                "transactions": energy_coin_transactions,
                "blocks": energy_coin_blocks,
                "stats": energy_coin_stats
            }
        }
        
        print("演示数据生成完成！")
        return demo_data
    
    def save_demo_data(self, file_path: str = "data/demo_data.json") -> None:
        """
        保存演示数据到JSON文件
        
        Args:
            file_path: 保存路径
        """
        demo_data = self.generate_complete_demo_data()
        
        # 转换数据为可序列化的格式
        serializable_data = json.dumps(demo_data, default=lambda obj: str(obj) if isinstance(obj, datetime.datetime) else obj.__dict__ if hasattr(obj, '__dict__') else str(obj))
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(serializable_data)
        
        print(f"演示数据已保存到 {file_path}")
    
    def load_demo_data_to_system(self, blockchain=None, smart_contract=None, data_collector=None) -> Dict[str, Any]:
        """
        将演示数据加载到系统组件中
        
        Args:
            blockchain: 区块链实例，如果为None则使用内部实例
            smart_contract: 智能合约实例，如果为None则使用内部实例
            data_collector: 数据采集器实例，如果为None则使用内部实例
            
        Returns:
            加载的演示数据
        """
        # 使用传入的实例或内部实例
        blockchain = blockchain or self.blockchain
        smart_contract = smart_contract or self.smart_contract
        data_collector = data_collector or self.data_collector
        
        # 创建数据目录
        os.makedirs("data", exist_ok=True)
        
        try:
            # 尝试读取保存的演示数据
            file_path = "data/demo_data.json"
            if not os.path.exists(file_path):
                print(f"演示数据文件 {file_path} 不存在，将生成新数据")
                return self.generate_complete_demo_data()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                serialized_data = f.read()
                demo_data = json.loads(serialized_data)
            
            print("正在加载演示数据...")
            
            # 1. 加载用户数据
            loaded_users = []
            for user_info in demo_data["users"]:
                address = user_info["address"]
                name = user_info["name"]
                is_producer = user_info["is_producer"]
                is_consumer = user_info["is_consumer"]
                
                # 注册用户
                if address not in smart_contract.users:
                    user = smart_contract.register_user(address, name, is_producer, is_consumer)
                    # 设置用户属性
                    user.reputation = user_info.get("reputation", 100.0)
                    user.active_score = user_info.get("active_score", 0.0)
                    user.is_green_certified = user_info.get("is_green_certified", False)
                    user.green_energy_subsidy = user_info.get("green_energy_subsidy", 0.0)
                    user.is_mining = user_info.get("is_mining", False)
                    
                    # 生成用户密钥对
                    self.data_encryptor.generate_user_keypair(address)
                else:
                    user = smart_contract.users[address]
                    
                # EnergyCoin相关配置
                if user.is_mining:
                    smart_contract.energy_coin.add_miner(user.energy_coin_address)
                
                # 如果有质押信息，创建质押
                staking_info = user_info.get("staking_info")
                if staking_info and staking_info.get("amount", 0) > 0:
                    try:
                        # 确保用户有足够余额质押
                        stake_amount = staking_info["amount"]
                        stake_days = staking_info.get("duration", 30)
                        
                        # 追加发放代币
                        smart_contract.energy_coin.add_transaction(
                            from_address="system",
                            to_address=user.energy_coin_address,
                            amount=stake_amount + 10,  # 多加10个作为缓冲
                            tx_type="initial_grant"
                        )
                        
                        # 创建质押
                        smart_contract.create_stake(address, stake_amount, stake_days)
                    except Exception as e:
                        print(f"为用户 {name} 创建质押时出错: {str(e)}")
                
                # 生成用户的能源数据
                user_config = {
                    "panel_capacity": user_info.get("panel_capacity", 5.0),
                    "user_type": user_info.get("user_type", "住宅"),
                    "base_load": user_info.get("base_load", 2.0),
                    "location": user_info.get("location", "北京")
                }
                
                # 如果没有缓存的用户能源数据，则生成样本数据
                if address not in data_collector.cached_data:
                    data_collector.generate_sample_data(address, user_config, days_back=60)
                
                loaded_users.append({
                    "address": address,
                    "name": name,
                    "is_producer": is_producer,
                    "is_consumer": is_consumer,
                    "energy_coin_address": user.energy_coin_address,
                    "is_mining": user.is_mining
                })
            
            print(f"已加载 {len(loaded_users)} 个用户")
            
            # 2. 加载能源挂单
            for listing in demo_data["listings"]:
                # 检查此挂单是否已经存在
                existing = False
                for l in smart_contract.energy_listings:
                    if l.get("id") == listing["id"]:
                        existing = True
                        break
                
                if not existing:
                    smart_contract.energy_listings.append(listing)
            
            print(f"已加载 {len(demo_data['listings'])} 个能源挂单")
            
            # 3. 加载投标
            for bid in demo_data["bids"]:
                # 检查此投标是否已经存在
                existing = False
                for b in smart_contract.energy_bids:
                    if b.get("id") == bid["id"]:
                        existing = True
                        break
                
                if not existing:
                    smart_contract.energy_bids.append(bid)
            
            print(f"已加载 {len(demo_data['bids'])} 个投标")
            
            # 4. 加载交易记录
            for tx in demo_data["transactions"]:
                # 检查此交易是否已经存在
                existing = False
                for t in smart_contract.transactions:
                    if t.get("id") == tx["id"]:
                        existing = True
                        break
                
                if not existing:
                    smart_contract.transactions.append(tx)
                    
                    # 对于已完成的交易，添加到已完成列表
                    if tx.get("status") == "completed":
                        smart_contract.completed_trades.append(tx)
            
            print(f"已加载 {len(demo_data['transactions'])} 个交易记录")
            
            # 5. 加载区块链数据
            if blockchain.chain and len(blockchain.chain) == 1:  # 只有创世区块
                # 清除现有链并重建
                blockchain.chain = []
                
                # 从保存的数据重建区块链
                for block_data in demo_data["blockchain"]["blocks"]:
                    block = Block(
                        index=block_data["index"],
                        transactions=block_data["transactions"],
                        timestamp=block_data["timestamp"],
                        previous_hash=block_data["previous_hash"],
                        nonce=block_data["nonce"]
                    )
                    block.hash = block_data["hash"]
                    blockchain.chain.append(block)
                
                print(f"已加载区块链，共 {len(blockchain.chain)} 个区块")
            else:
                print("区块链已有数据，跳过加载")
            
            # 6. 加载EnergyCoin交易数据
            if "energy_coin" in demo_data:
                # 处理一些区块，确保EnergyCoin系统已初始化
                self.process_energy_coin_blocks()
                
                print(f"已加载 EnergyCoin 统计信息")
            
            print("演示数据加载完成！")
            return demo_data
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"加载演示数据失败: {str(e)}，将生成新数据")
            return self.generate_complete_demo_data()

if __name__ == "__main__":
    # 创建演示数据生成器
    generator = DemoDataGenerator()
    
    # 生成并保存演示数据
    generator.save_demo_data()