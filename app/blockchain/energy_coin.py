"""
EnergyCoin - 基于PoW与PoS混合共识机制的加密货币
为分布式光伏发电点对点交易系统设计
"""
import hashlib
import time
import json
import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

@dataclass
class StakeInfo:
    """质押信息"""
    address: str  # 质押者地址
    amount: float  # 质押金额
    timestamp: float  # 质押时间
    duration: int  # 质押期限(天)
    active: bool = True  # 是否处于激活状态
    
    @property
    def weight(self) -> float:
        """
        计算质押权重
        权重 = 质押金额 * 质押时长(天) * 衰减因子
        """
        # 质押时长(天)
        days_staked = min((time.time() - self.timestamp) / 86400, self.duration)
        # 质押时长增益因子(最大1.5)
        duration_factor = min(1.0 + days_staked / 100, 1.5)
        # 基础权重
        return self.amount * duration_factor if self.active else 0


class EnergyCoinBlock:
    """EnergyCoin区块"""
    def __init__(self, index: int, transactions: List[Dict[str, Any]], 
                timestamp: float, previous_hash: str, 
                validator: str = "", nonce: int = 0, difficulty: int = 4):
        """
        初始化EnergyCoin区块
        
        Args:
            index: 区块索引/高度
            transactions: 区块中包含的交易列表
            timestamp: 区块创建的时间戳
            previous_hash: 前一个区块的哈希
            validator: 验证者地址(PoS)
            nonce: 工作量证明的随机数(PoW)
            difficulty: 挖矿难度
        """
        self.index = index
        self.transactions = transactions
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.validator = validator
        self.nonce = nonce
        self.difficulty = difficulty
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """计算区块哈希值"""
        block_string = json.dumps({
            "index": self.index,
            "transactions": self.transactions,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "validator": self.validator,
            "nonce": self.nonce,
            "difficulty": self.difficulty
        }, sort_keys=True).encode()
        
        return hashlib.sha256(block_string).hexdigest()
    
    def mine_block(self, difficulty: int) -> None:
        """
        PoW挖矿 - 找到满足难度要求的哈希值
        
        Args:
            difficulty: 挖矿难度(前导0的个数)
        """
        self.difficulty = difficulty
        target = "0" * difficulty
        
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
    
    def to_dict(self) -> Dict[str, Any]:
        """将区块转换为字典格式"""
        return {
            "index": self.index,
            "transactions": self.transactions,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "validator": self.validator,
            "nonce": self.nonce,
            "difficulty": self.difficulty,
            "hash": self.hash
        }


class EnergyCoin:
    """EnergyCoin加密货币系统"""
    
    def __init__(self, pow_difficulty: int = 4, block_time: int = 60, initial_supply: float = 1000000.0, mining_reward: float = 50.0, staking_reward_percentage: float = 0.05):
        """
        初始化EnergyCoin系统
        
        Args:
            pow_difficulty: PoW挖矿难度
            block_time: 目标出块时间(秒)
            initial_supply: 初始供应量
            mining_reward: 挖矿奖励
            staking_reward_percentage: 质押奖励百分比
        """
        self.chain: List[EnergyCoinBlock] = []
        self.pending_transactions: List[Dict[str, Any]] = []
        self.pow_difficulty = pow_difficulty
        self.block_time = block_time
        self.initial_supply = initial_supply
        self.mining_reward = mining_reward
        self.staking_reward_percentage = staking_reward_percentage
        
        # 为了与smart_contract.py兼容，添加别名
        self.pow_reward = mining_reward  
        self.pos_reward = staking_reward_percentage
        
        # 账户余额
        self.accounts = {
            "system": initial_supply  # 系统持有所有初始代币
        }
        self.account_nonces = {}  # 账户交易nonce，防止重放攻击
        
        # 质押信息
        self.staking_info = {}  # 存储质押信息
        self.staking_pool = 0.0  # 质押池余额
        
        # 交易历史
        self.transactions = []
        
        # 矿工列表
        self.miners = set()
        
        self.create_genesis_block()
        
    def create_genesis_block(self) -> None:
        """创建并添加创世区块"""
        genesis_block = EnergyCoinBlock(
            0, 
            [{"from": "system", "to": "genesis", "amount": 1000, "type": "init"}], 
            time.time(), 
            "0"
        )
        genesis_block.hash = genesis_block.calculate_hash()
        self.chain.append(genesis_block)
    
    def get_latest_block(self) -> EnergyCoinBlock:
        """获取最新区块"""
        return self.chain[-1]
    
    def add_transaction(self, from_address: str, to_address: str, 
                      amount: float, tx_type: str = "transfer") -> int:
        """
        添加交易到待处理列表
        
        Args:
            from_address: 发送方地址
            to_address: 接收方地址
            amount: 交易金额
            tx_type: 交易类型
            
        Returns:
            将包含此交易的区块索引
        """
        # 验证交易
        if from_address != "system":  # 系统生成的交易不需要验证余额
            balance = self.get_balance(from_address)
            if balance < amount:
                raise ValueError(f"余额不足: {balance} < {amount}")
        
        transaction = {
            "from": from_address,
            "to": to_address,
            "amount": amount,
            "timestamp": time.time(),
            "type": tx_type,
            "signature": ""  # 实际应用中应该添加数字签名
        }
        
        self.pending_transactions.append(transaction)
        self.transactions.append(transaction)
        return self.get_latest_block().index + 1
    
    def mine_pending_transactions(self, miner_address: str) -> EnergyCoinBlock:
        """
        使用PoW挖掘待处理的交易
        
        Args:
            miner_address: 矿工地址
            
        Returns:
            新挖出的区块
        """
        # 调整难度
        self._adjust_difficulty()
        
        # 添加挖矿奖励交易
        self.pending_transactions.append({
            "from": "system",
            "to": miner_address,
            "amount": self.mining_reward,
            "timestamp": time.time(),
            "type": "mining_reward",
            "signature": ""
        })
        
        # 创建新区块
        block = EnergyCoinBlock(
            index=len(self.chain),
            transactions=self.pending_transactions,
            timestamp=time.time(),
            previous_hash=self.get_latest_block().hash
        )
        
        # 执行PoW挖矿
        block.mine_block(self.pow_difficulty)
        
        # 将新区块添加到链上
        self.chain.append(block)
        
        # 处理区块中的交易
        self._process_block_transactions(block)
        
        # 清空待处理的交易，为下一个区块准备
        self.pending_transactions = []
        
        return block
    
    def _process_block_transactions(self, block: EnergyCoinBlock) -> None:
        """
        处理区块中的所有交易
        
        Args:
            block: 包含交易的区块
        """
        for tx in block.transactions:
            from_address = tx["from"]
            to_address = tx["to"]
            amount = tx["amount"]
            
            # 更新余额
            if from_address != "system":  # 系统生成的交易不影响系统余额
                if from_address not in self.accounts:
                    self.accounts[from_address] = 0
                self.accounts[from_address] -= amount
            
            if to_address not in self.accounts:
                self.accounts[to_address] = 0
            self.accounts[to_address] += amount
    
    def create_stake(self, address: str, amount: float, duration: int, auto_fund: bool = True) -> bool:
        """
        创建质押
        
        Args:
            address: 质押者地址
            amount: 质押金额
            duration: 质押期限(天)
            auto_fund: 当余额不足时是否自动添加资金（仅用于演示）
            
        Returns:
            是否成功创建质押
        """
        # 验证质押参数
        if amount <= 0:
            raise ValueError("质押金额必须大于0")
        
        if duration < 7:
            raise ValueError("质押期限至少为7天")
        
        # 检查余额
        balance = self.get_balance(address)
        if balance < amount:
            # 如果允许自动添加资金且是演示模式
            if auto_fund:
                # 自动添加资金以满足质押需求
                self.add_transaction(
                    from_address="system",
                    to_address=address,
                    amount=amount - balance + 50,  # 多加50个作为缓冲
                    tx_type="demo_funding"
                )
                
                # 直接处理交易更新余额
                if self.pending_transactions:
                    new_block = EnergyCoinBlock(
                        index=len(self.chain),
                        transactions=self.pending_transactions,
                        timestamp=time.time(),
                        previous_hash=self.get_latest_block().hash
                    )
                    new_block.hash = new_block.calculate_hash()
                    self.chain.append(new_block)
                    self._process_block_transactions(new_block)
                    self.pending_transactions = []
            else:
                raise ValueError(f"质押失败: 余额不足: {balance} < {amount}")
        
        # 创建质押交易
        self.add_transaction(
            from_address=address,
            to_address="stake_pool",
            amount=amount,
            tx_type="stake"
        )
        
        # 记录质押信息
        stake_info = StakeInfo(
            address=address,
            amount=amount,
            timestamp=time.time(),
            duration=duration,
            active=True
        )
        self.staking_info[address] = stake_info
        
        # 矿工列表添加该质押账户
        self.miners.add(address)
        
        return True
    
    def release_stake(self, address: str) -> float:
        """
        释放质押
        
        Args:
            address: 质押者地址
            
        Returns:
            释放的质押金额
        """
        if address not in self.staking_info:
            raise ValueError("没有找到该地址的质押")
        
        stake_info = self.staking_info[address]
        if not stake_info.active:
            raise ValueError("该质押已经释放")
        
        # 检查质押期限是否已到
        staking_time = (time.time() - stake_info.timestamp) / 86400  # 转换为天
        if staking_time < stake_info.duration:
            raise ValueError(f"质押期限未到，还需等待{stake_info.duration - staking_time:.1f}天")
        
        # 释放质押
        stake_info.active = False
        released_amount = stake_info.amount
        
        # 创建释放质押交易
        self.add_transaction(
            from_address="stake_pool",
            to_address=address,
            amount=released_amount,
            tx_type="unstake"
        )
        
        # 计算并发放质押奖励
        daily_rate = 0.0005  # 每日收益率 0.05%
        reward = released_amount * daily_rate * stake_info.duration
        
        if reward > 0:
            self.add_transaction(
                from_address="system",
                to_address=address,
                amount=reward,
                tx_type="stake_reward"
            )
        
        return released_amount + reward
    
    def validate_block_pos(self) -> Tuple[bool, str]:
        """
        使用PoS机制选择验证者
        
        Returns:
            (是否成功, 被选中的验证者地址)
        """
        # 总质押权重
        total_weight = sum(stake.weight for stake in self.staking_info.values() if stake.active)
        if total_weight <= 0:
            return False, ""
        
        # 基于质押权重选择验证者
        validator_candidates = []
        for address, stake in self.staking_info.items():
            if stake.active:
                # 计算选中概率比例
                selection_weight = stake.weight / total_weight
                # 转换为选择点数
                points = int(selection_weight * 10000)
                # 添加到候选人列表
                validator_candidates.extend([address] * points)
        
        # 随机选择验证者
        if validator_candidates:
            validator = random.choice(validator_candidates)
            return True, validator
        
        return False, ""
    
    def create_block_pos(self, validator_address: str) -> EnergyCoinBlock:
        """
        使用PoS创建新区块(由被选中的验证者调用)
        
        Args:
            validator_address: 验证者地址
            
        Returns:
            新创建的区块
        """
        # 验证调用者是否为被选中的验证者
        is_valid, chosen_validator = self.validate_block_pos()
        if not is_valid or validator_address != chosen_validator:
            raise ValueError("非法验证者")
        
        # 添加验证奖励交易
        self.pending_transactions.append({
            "from": "system",
            "to": validator_address,
            "amount": self.mining_reward * 0.1,
            "timestamp": time.time(),
            "type": "staking_reward",
            "signature": ""
        })
        
        # 创建新区块
        block = EnergyCoinBlock(
            index=len(self.chain),
            transactions=self.pending_transactions,
            timestamp=time.time(),
            previous_hash=self.get_latest_block().hash,
            validator=validator_address
        )
        
        # 设置区块哈希(PoS不需要大量计算)
        block.hash = block.calculate_hash()
        
        # 将新区块添加到链上
        self.chain.append(block)
        
        # 处理区块中的交易
        self._process_block_transactions(block)
        
        # 清空待处理的交易
        self.pending_transactions = []
        
        return block
    
    def process_next_block(self, miner_address: str) -> EnergyCoinBlock:
        """
        处理下一个区块(自动选择PoW或PoS)
        
        Args:
            miner_address: 矿工地址
            
        Returns:
            新区块
        """
        # 首先尝试PoS
        is_pos_valid, validator = self.validate_block_pos()
        
        # 如果PoS有效，且选中的验证者恰好是调用者
        if is_pos_valid and validator == miner_address:
            # 使用PoS创建区块
            return self.create_block_pos(miner_address)
        
        # 否则使用PoW
        return self.mine_pending_transactions(miner_address)
    
    def get_balance(self, address: str) -> float:
        """
        计算特定地址的EnergyCoin余额
        
        Args:
            address: 要查询余额的地址
            
        Returns:
            地址的EnergyCoin余额
        """
        return self.accounts.get(address, 0)
    
    def get_transactions_by_address(self, address: str) -> List[Dict[str, Any]]:
        """
        获取特定地址的所有交易记录
        
        Args:
            address: 要查询交易的地址
            
        Returns:
            地址相关的所有交易列表
        """
        transactions = []
        
        # 从交易记录中查找
        for tx in self.transactions:
            if tx.get("from") == address or tx.get("to") == address:
                tx_copy = tx.copy()
                
                # 检查交易是否在区块中
                block_index = -1
                for i, block in enumerate(self.chain):
                    for block_tx in block.transactions:
                        if (block_tx.get("from") == tx.get("from") and 
                            block_tx.get("to") == tx.get("to") and 
                            block_tx.get("amount") == tx.get("amount") and
                            abs(block_tx.get("timestamp", 0) - tx.get("timestamp", 0)) < 1):
                            block_index = i
                            break
                    if block_index >= 0:
                        break
                
                if block_index >= 0:
                    tx_copy["block_index"] = block_index
                    tx_copy["confirmed"] = True
                else:
                    tx_copy["confirmed"] = False
                
                transactions.append(tx_copy)
        
        return transactions
    
    def is_chain_valid(self) -> bool:
        """
        验证整个区块链的完整性
        
        Returns:
            如果区块链有效返回True，否则返回False
        """
        # 从第一个区块开始验证（跳过创世区块）
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # 验证当前区块的哈希是否正确
            if current_block.hash != current_block.calculate_hash():
                return False
            
            # 验证当前区块的previous_hash是否指向前一个区块的哈希
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True
    
    def get_total_supply(self) -> float:
        """
        获取EnergyCoin总供应量
        
        Returns:
            EnergyCoin总供应量
        """
        total_supply = 0
        
        # 查找所有系统发行的代币
        for block in self.chain:
            for transaction in block.transactions:
                if transaction.get("from") == "system":
                    total_supply += transaction.get("amount", 0)
        
        return total_supply
    
    def get_staked_amount(self) -> float:
        """
        获取总质押量
        
        Returns:
            总质押量
        """
        return sum(stake.amount for stake in self.staking_info.values() if stake.active)
    
    def get_staking_info(self, address: str) -> Optional[Dict[str, Any]]:
        """
        获取用户质押信息
        
        Args:
            address: 用户地址
            
        Returns:
            质押信息字典，如果没有质押则返回None
        """
        if address not in self.staking_info:
            return None
            
        stake = self.staking_info[address]
        if not stake.active:
            return None
            
        current_time = time.time()
        days_staked = (current_time - stake.timestamp) / 86400
        days_remaining = max(0, stake.duration - days_staked)
        
        return {
            "amount": stake.amount,
            "duration": stake.duration,
            "time_staked": stake.timestamp,
            "days_staked": days_staked,
            "days_remaining": days_remaining,
            "release_time": stake.timestamp + (stake.duration * 86400),
            "weight": stake.weight
        }
    
    def _adjust_difficulty(self) -> None:
        """动态调整挖矿难度"""
        # 每10个区块调整一次难度
        if len(self.chain) % 10 != 0:
            return
            
        # 计算最近10个区块的平均出块时间
        if len(self.chain) >= 11:
            total_time = self.chain[-1].timestamp - self.chain[-11].timestamp
            average_time = total_time / 10
            
            # 如果平均出块时间太长，降低难度
            if average_time > self.block_time * 1.2:
                self.pow_difficulty = max(1, self.pow_difficulty - 1)
                
            # 如果平均出块时间太短，增加难度
            elif average_time < self.block_time * 0.8:
                self.pow_difficulty += 1
                
        self.last_difficulty_adjustment = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将EnergyCoin区块链转换为字典格式
        
        Returns:
            EnergyCoin的字典表示
        """
        return {
            "chain": [block.to_dict() for block in self.chain],
            "pending_transactions": self.pending_transactions,
            "pow_difficulty": self.pow_difficulty,
            "pow_reward": self.mining_reward,
            "pos_reward": self.staking_reward_percentage,
            "block_time": self.block_time,
            "total_supply": self.get_total_supply(),
            "staked_amount": self.get_staked_amount()
        }
    
    def calculate_pow_probability(self, account_id: str) -> float:
        """计算PoW挖矿成功概率"""
        if account_id not in self.miners:
            return 0.0
            
        # 算力均等分配(简化版)
        return 1.0 / len(self.miners) if self.miners else 0.0
    
    def calculate_pos_probability(self, account_id: str) -> float:
        """计算PoS验证成功概率"""
        if account_id not in self.staking_info:
            return 0.0
            
        # 根据质押比例决定
        total_staking = sum([info["amount"] for info in self.staking_info.values()])
        if total_staking <= 0:
            return 0.0
            
        account_staking = self.staking_info[account_id]["amount"]
        return account_staking / total_staking
    
    def select_miner(self) -> str:
        """选择矿工进行PoW挖矿"""
        if not self.miners:
            return None
            
        # 简化版：随机选择一个矿工
        return random.choice(list(self.miners))
    
    def select_validator(self) -> str:
        """选择验证者进行PoS验证"""
        if not self.staking_info:
            return None
            
        # 根据质押金额加权随机选择
        total_staking = sum([info["amount"] for info in self.staking_info.values()])
        if total_staking <= 0:
            return None
            
        # 权重计算
        weighted_accounts = [(account, info["amount"] / total_staking) 
                            for account, info in self.staking_info.items()]
        
        # 加权随机选择
        accounts, weights = zip(*weighted_accounts)
        return random.choices(accounts, weights=weights, k=1)[0]
    
    def process_block(self, account_id: str) -> Dict[str, Any]:
        """处理区块：混合PoW和PoS共识"""
        # 决定使用哪种共识机制
        has_stake = account_id in self.staking_info
        
        # 如果有质押，有70%的概率使用PoS，30%的概率使用PoW
        # 如果没有质押，只能使用PoW
        use_pos = has_stake and random.random() < 0.7
        
        if use_pos:
            # PoS验证
            # 根据质押金额和其他因素决定是否有权验证
            pos_probability = self.calculate_pos_probability(account_id)
            
            if random.random() < pos_probability:
                return self.validate_pos_block(account_id)
            else:
                raise ValueError("未获得PoS验证权限")
        else:
            # PoW挖矿
            # 根据算力和难度决定是否挖矿成功
            pow_probability = self.calculate_pow_probability(account_id)
            
            if random.random() < pow_probability:
                return self.mine_block(account_id)
            else:
                raise ValueError("PoW挖矿失败")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取EnergyCoin统计信息"""
        # 计算总供应量(已经发行的代币总量)
        total_supply = self.initial_supply - self.accounts.get("system", 0)
        
        # 计算总质押量
        total_staked = self.staking_pool
        
        # 计算质押比例
        staking_ratio = total_staked / total_supply if total_supply > 0 else 0
        
        # 获取活跃矿工数
        active_miners = self.get_active_miner_count()
        
        return {
            "total_supply": total_supply,
            "total_staked": total_staked,
            "staking_ratio": staking_ratio,
            "active_miners": active_miners,
            "block_count": len(self.chain),
            "difficulty": self.pow_difficulty,
            "mining_reward": self.mining_reward,
            "staking_reward": self.staking_reward_percentage,
            "transaction_count": len(self.transactions),
        }
        
    def get_transactions_for_account(self, account_id: str) -> List[Dict[str, Any]]:
        """获取账户相关的所有交易"""
        result = []
        
        for tx in self.transactions:
            if tx["from"] == account_id or tx["to"] == account_id:
                result.append(tx)
                
        return result

    def update_difficulty(self):
        """动态调整挖矿难度"""
        if len(self.chain) < 10:  # 少于10个区块时不调整
            return
            
        # 获取最近10个区块的平均出块时间
        recent_blocks = self.chain[-10:]
        timestamps = [b["timestamp"] for b in recent_blocks]
        
        # 计算平均出块时间(秒)
        avg_block_time = sum([timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]) / (len(timestamps)-1)
        
        # 目标出块时间
        target_block_time = 60  # 60秒一个区块
        
        # 调整难度
        if avg_block_time < target_block_time * 0.8:  # 出块太快，提高难度
            self.pow_difficulty += 1
        elif avg_block_time > target_block_time * 1.2:  # 出块太慢，降低难度
            self.pow_difficulty = max(1, self.pow_difficulty - 1)
            
    def print_chain(self):
        """打印区块链信息"""
        for block in self.chain:
            print(f"区块 #{block['index']}")
            print(f"时间戳: {block['timestamp']}")
            print(f"哈希: {block['hash']}")
            print(f"前一个哈希: {block['previous_hash']}")
            print(f"交易数: {len(block['transactions'])}")
            print("----------------------------")
            
    def add_miner(self, address: str) -> None:
        """
        将地址添加到矿工列表
        
        Args:
            address: 矿工地址
        """
        self.miners.add(address)
        
    def get_active_miner_count(self) -> int:
        """
        获取活跃矿工数量
        
        Returns:
            活跃矿工数量
        """
        return len(self.miners)