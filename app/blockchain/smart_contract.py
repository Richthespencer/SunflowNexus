import time
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from .energy_coin import EnergyCoin


class TradingConditionType(Enum):
    """交易条件类型枚举"""
    PRICE_THRESHOLD = "price_threshold"  # 价格阈值条件
    TIME_CONSTRAINT = "time_constraint"  # 时间约束条件
    ENERGY_AMOUNT = "energy_amount"      # 能源数量条件
    REPUTATION_THRESHOLD = "reputation_threshold"  # 信誉度阈值条件
    GREEN_CERTIFIED = "green_certified"  # 绿色能源认证条件
    PEAK_AVOIDANCE = "peak_avoidance"    # 避开用电高峰条件


@dataclass
class TradingCondition:
    """交易条件基类"""
    condition_type: TradingConditionType
    description: str
    
    def evaluate(self, listing: Dict[str, Any], user: 'User', context: Dict[str, Any]) -> bool:
        """
        评估条件是否满足
        
        Args:
            listing: 能源挂单信息
            user: 相关用户
            context: 评估上下文信息
            
        Returns:
            条件是否满足
        """
        raise NotImplementedError("必须在子类中实现此方法")


@dataclass
class PriceCondition(TradingCondition):
    """价格条件"""
    max_price_per_kwh: float  # 最大接受价格
    min_price_per_kwh: float = 0.0  # 最小接受价格
    
    def __post_init__(self):
        self.condition_type = TradingConditionType.PRICE_THRESHOLD
        self.description = f"价格介于 {self.min_price_per_kwh} - {self.max_price_per_kwh} 元/kWh之间"
    
    def evaluate(self, listing: Dict[str, Any], user: 'User', context: Dict[str, Any]) -> bool:
        """评估价格条件是否满足"""
        price = listing["price_per_kwh"]
        return self.min_price_per_kwh <= price <= self.max_price_per_kwh


@dataclass
class TimeCondition(TradingCondition):
    """时间条件"""
    valid_hours: int  # 挂单有效期要求(小时)
    
    def __post_init__(self):
        self.condition_type = TradingConditionType.TIME_CONSTRAINT
        self.description = f"挂单有效期至少 {self.valid_hours} 小时"
    
    def evaluate(self, listing: Dict[str, Any], user: 'User', context: Dict[str, Any]) -> bool:
        """评估时间条件是否满足"""
        current_time = time.time()
        valid_until = listing["valid_until"]
        return (valid_until - current_time) >= (self.valid_hours * 3600)


@dataclass
class EnergyAmountCondition(TradingCondition):
    """能源数量条件"""
    min_amount: float  # 最小能源数量
    max_amount: float  # 最大能源数量
    
    def __post_init__(self):
        self.condition_type = TradingConditionType.ENERGY_AMOUNT
        self.description = f"能源数量介于 {self.min_amount} - {self.max_amount} kWh之间"
    
    def evaluate(self, listing: Dict[str, Any], user: 'User', context: Dict[str, Any]) -> bool:
        """评估能源数量条件是否满足"""
        amount = listing["amount"]
        return self.min_amount <= amount <= self.max_amount


@dataclass
class ReputationCondition(TradingCondition):
    """信誉度条件"""
    min_reputation: float  # 最低信誉要求
    
    def __post_init__(self):
        self.condition_type = TradingConditionType.REPUTATION_THRESHOLD
        self.description = f"交易对手信誉值不低于 {self.min_reputation}"
    
    def evaluate(self, listing: Dict[str, Any], user: 'User', context: Dict[str, Any]) -> bool:
        """评估信誉度条件是否满足"""
        reputation = context.get("reputation", 0)
        return reputation >= self.min_reputation


@dataclass
class GreenCertifiedCondition(TradingCondition):
    """绿能认证条件"""
    require_certified: bool  # 是否要求绿能认证
    
    def __post_init__(self):
        self.condition_type = TradingConditionType.GREEN_CERTIFIED
        self.description = "要求绿能认证" if self.require_certified else "不要求绿能认证"
    
    def evaluate(self, listing: Dict[str, Any], user: 'User', context: Dict[str, Any]) -> bool:
        """评估绿能认证条件是否满足"""
        is_green_certified = listing.get("green_certified", False)
        return is_green_certified == self.require_certified


@dataclass
class PeakAvoidanceCondition(TradingCondition):
    """避开用电高峰条件"""
    peak_hours: List[int]  # 高峰时段列表 (0-23)
    
    def __post_init__(self):
        self.condition_type = TradingConditionType.PEAK_AVOIDANCE
        self.description = f"避开高峰时段: {', '.join(map(str, self.peak_hours))}点"
    
    def evaluate(self, listing: Dict[str, Any], user: 'User', context: Dict[str, Any]) -> bool:
        """评估是否避开高峰时段"""
        current_hour = context.get("current_hour", 0)
        return current_hour not in self.peak_hours


@dataclass
class AutoTradingStrategy:
    """自动交易策略"""
    user_address: str  # 用户地址
    is_buy_strategy: bool  # 是否为买入策略(True为买入,False为卖出)
    conditions: List[TradingCondition] = field(default_factory=list)  # 交易条件列表
    energy_amount: float = None  # 目标交易能源数量
    max_price_per_kwh: float = None  # 最大接受价格
    min_price_per_kwh: float = None  # 最小接受价格
    valid_hours: int = 24  # 挂单有效期(小时)
    auto_accept_offer: bool = False  # 自动接受报价
    description: str = ""  # 策略描述
    is_active: bool = True  # 策略是否激活
    created_at: float = field(default_factory=time.time)  # 策略创建时间
    
    def add_condition(self, condition: TradingCondition):
        """添加交易条件"""
        self.conditions.append(condition)
        return self
    
    def check_all_conditions(self, listing: Dict[str, Any], user: 'User', contract: 'EnergyTradingContract') -> bool:
        """
        检查所有条件是否满足
        
        Args:
            listing: 能源挂单信息
            user: 用户信息
            contract: 智能合约实例
            
        Returns:
            是否满足所有条件
        """
        # 构建评估上下文
        context = {
            "current_time": time.time(),
            "current_hour": time.localtime().tm_hour,
            "reputation": contract.calculate_reputation(listing["producer_address"]) if self.is_buy_strategy else 0
        }
        
        # 检查所有条件
        for condition in self.conditions:
            if not condition.evaluate(listing, user, context):
                return False
        
        return True


@dataclass
class User:
    """用户信息类"""
    address: str  # 用户地址
    name: str  # 用户名称
    reputation: float = 100.0  # 用户信誉值，初始为100
    is_producer: bool = False  # 是否为发电方
    is_consumer: bool = False  # 是否为用电方
    is_green_certified: bool = False  # 是否获得绿能认证
    green_energy_subsidy: float = 0.0  # 电力补贴余额
    active_score: float = 0.0  # 节点活跃度
    trading_strategies: List[AutoTradingStrategy] = field(default_factory=list)  # 用户的自动交易策略列表
    energy_coin_address: str = ""  # EnergyCoin钱包地址
    is_mining: bool = False  # 是否参与挖矿

    def add_trading_strategy(self, strategy: AutoTradingStrategy):
        """添加交易策略"""
        strategy.user_address = self.address
        self.trading_strategies.append(strategy)
        return strategy


class EnergyTradingContract:
    """光伏发电交易智能合约"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}  # 用户列表
        self.transactions: List[Dict[str, Any]] = []  # 交易记录
        self.energy_listings: List[Dict[str, Any]] = []  # 电力挂单列表
        self.energy_bids: List[Dict[str, Any]] = []  # 电力竞标列表
        self.completed_trades: List[Dict[str, Any]] = []  # 已完成交易
        self.auto_trading_logs: List[Dict[str, Any]] = []  # 自动交易日志
        self.green_certification_requirements = {
            "min_reputation": 85.0,  # 最低信誉值
            "min_active_score": 70.0,  # 最低活跃度
            "min_energy_production": 100.0,  # 最低发电量(kWh)
        }
        # 初始化EnergyCoin系统
        self.energy_coin = EnergyCoin(pow_difficulty=3, block_time=60)
        
    def register_user(self, address: str, name: str, is_producer: bool = False, 
                      is_consumer: bool = False) -> User:
        """
        注册新用户
        
        Args:
            address: 用户地址
            name: 用户名称
            is_producer: 是否为发电方
            is_consumer: 是否为用电方
            
        Returns:
            注册的用户对象
        """
        if address in self.users:
            raise ValueError(f"用户地址 {address} 已存在")
        
        user = User(
            address=address,
            name=name,
            is_producer=is_producer,
            is_consumer=is_consumer
        )
        
        # 创建EnergyCoin钱包地址
        user.energy_coin_address = f"EC_{address}"
        
        # 为新用户发放初始EnergyCoin
        initial_amount = 5.0  # 初始赠送5个EnergyCoin
        self.energy_coin.add_transaction(
            from_address="system",
            to_address=user.energy_coin_address,
            amount=initial_amount,
            tx_type="initial_grant"
        )
        
        # 创建创世区块，将交易打包
        if len(self.energy_coin.chain) <= 1 and len(self.energy_coin.pending_transactions) > 0:
            self.energy_coin.process_next_block("system")
        
        self.users[address] = user
        return user
    
    def list_energy_for_sale(self, producer_address: str, amount: float, 
                          price_per_kwh: float, valid_until: float) -> Dict[str, Any]:
        """
        发电方挂单出售电力
        
        Args:
            producer_address: 发电方地址
            amount: 电力数量 (kWh)
            price_per_kwh: 每千瓦时价格
            valid_until: 有效期截止时间戳
            
        Returns:
            电力挂单信息
        """
        if producer_address not in self.users:
            raise ValueError("用户不存在")
            
        user = self.users[producer_address]
        if not user.is_producer:
            raise ValueError("该用户不是发电方")
        
        listing = {
            "id": len(self.energy_listings) + 1,
            "producer_address": producer_address,
            "amount": amount,
            "price_per_kwh": price_per_kwh,
            "valid_until": valid_until,
            "created_at": time.time(),
            "status": "active",
            "green_certified": user.is_green_certified  # 是否为绿能认证
        }
        
        self.energy_listings.append(listing)
        return listing
    
    def place_bid(self, consumer_address: str, listing_id: int, amount: float) -> Dict[str, Any]:
        """
        用电方对电力挂单进行投标
        
        Args:
            consumer_address: 用电方地址
            listing_id: 电力挂单ID
            amount: 电力数量 (kWh)，不能超过挂单数量
            
        Returns:
            投标信息
        """
        if consumer_address not in self.users:
            raise ValueError("用户不存在")
            
        user = self.users[consumer_address]
        if not user.is_consumer:
            raise ValueError("该用户不是用电方")
            
        # 查找对应的挂单
        listing = None
        for l in self.energy_listings:
            if l["id"] == listing_id and l["status"] == "active":
                listing = l
                break
                
        if not listing:
            raise ValueError(f"无效的挂单ID: {listing_id}")
            
        if amount > listing["amount"]:
            raise ValueError(f"投标数量不能超过挂单数量")
            
        if time.time() > listing["valid_until"]:
            raise ValueError("该挂单已过期")
            
        bid = {
            "id": len(self.energy_bids) + 1,
            "consumer_address": consumer_address,
            "listing_id": listing_id,
            "amount": amount,
            "price_per_kwh": listing["price_per_kwh"],
            "created_at": time.time(),
            "status": "pending"
        }
        
        self.energy_bids.append(bid)
        return bid
    
    def accept_bid(self, producer_address: str, bid_id: int) -> Dict[str, Any]:
        """
        发电方接受投标
        
        Args:
            producer_address: 发电方地址
            bid_id: 投标ID
            
        Returns:
            交易信息
        """
        # 查找对应的投标
        bid = None
        for b in self.energy_bids:
            if b["id"] == bid_id and b["status"] == "pending":
                bid = b
                break
                
        if not bid:
            raise ValueError(f"无效的投标ID: {bid_id}")
            
        # 查找对应的挂单
        listing = None
        for l in self.energy_listings:
            if l["id"] == bid["listing_id"] and l["status"] == "active":
                listing = l
                break
                
        if not listing:
            raise ValueError("关联的挂单不存在或已关闭")
            
        if listing["producer_address"] != producer_address:
            raise ValueError("您不是该挂单的所有者")
            
        # 完成交易
        transaction = {
            "id": len(self.transactions) + 1,
            "producer_address": producer_address,
            "consumer_address": bid["consumer_address"],
            "amount": bid["amount"],
            "price_per_kwh": bid["price_per_kwh"],
            "total_price": bid["amount"] * bid["price_per_kwh"],
            "created_at": time.time(),
            "listing_id": listing["id"],
            "bid_id": bid["id"],
            "status": "contracted",  # 已签订合约但未确认实际用电量
            "actual_amount": None  # 实际用电量，后续确认
        }
        
        # 更新投标状态
        bid["status"] = "accepted"
        
        # 如果挂单的电量被完全买走，则关闭挂单
        remaining = listing["amount"] - bid["amount"]
        if remaining <= 0:
            listing["status"] = "completed"
        else:
            listing["amount"] = remaining
            
        self.transactions.append(transaction)
        return transaction
    
    def confirm_energy_delivery(self, transaction_id: int, actual_amount: float) -> Dict[str, Any]:
        """
        确认实际电力交付情况
        
        Args:
            transaction_id: 交易ID
            actual_amount: 实际交付的电力数量 (kWh)
            
        Returns:
            更新后的交易信息
        """
        # 查找对应的交易
        transaction = None
        for tx in self.transactions:
            if tx["id"] == transaction_id and tx["status"] == "contracted":
                transaction = tx
                break
                
        if not transaction:
            raise ValueError(f"无效的交易ID: {transaction_id}")
            
        producer = self.users[transaction["producer_address"]]
        consumer = self.users[transaction["consumer_address"]]
        
        # 计算实际交付率
        contracted_amount = transaction["amount"]
        delivery_ratio = actual_amount / contracted_amount
        
        # 更新交易状态
        transaction["actual_amount"] = actual_amount
        transaction["delivery_ratio"] = delivery_ratio
        transaction["final_price"] = actual_amount * transaction["price_per_kwh"]
        transaction["status"] = "completed"
        
        # 支付EnergyCoin
        final_price_coin = transaction["final_price"] / 10  # 1 EnergyCoin = 10元电费
        
        try:
            # 从用电方转账给发电方
            self.energy_coin.add_transaction(
                from_address=self.users[transaction["consumer_address"]].energy_coin_address,
                to_address=self.users[transaction["producer_address"]].energy_coin_address,
                amount=final_price_coin,
                tx_type="energy_payment"
            )
            
            # 添加交易记录
            transaction["energy_coin_payment"] = {
                "amount": final_price_coin,
                "status": "pending",
                "timestamp": time.time()
            }
            
            # 如果有足够的待处理交易，自动生成新区块
            if len(self.energy_coin.pending_transactions) >= 3:
                # 从挖矿用户中选择一个来处理区块
                mining_users = [user for user in self.users.values() if user.is_mining]
                if mining_users:
                    import random
                    miner = random.choice(mining_users)
                    self.energy_coin.process_next_block(miner.energy_coin_address)
                    transaction["energy_coin_payment"]["status"] = "confirmed"
        except Exception as e:
            transaction["energy_coin_payment_error"] = str(e)
        
        # 更新用户信誉值
        if delivery_ratio < 0.9:  # 交付率低于90%
            # 发电方违约，扣减信誉值
            penalty = (0.9 - delivery_ratio) * 10 * 10  # 缺口越大，惩罚越重
            producer.reputation -= penalty
            
            # 绿能补贴惩罚
            if producer.is_green_certified:
                subsidy_penalty = penalty * 0.5
                producer.green_energy_subsidy -= subsidy_penalty
        else:
            # 交付率良好，提升信誉值
            producer.reputation += min(2.0, (delivery_ratio - 0.9) * 5)
            consumer.reputation += 1.0
            
            # 绿能补贴奖励
            if producer.is_green_certified:
                producer.green_energy_subsidy += actual_amount * 0.01  # 每kWh奖励0.01单位补贴
                
        # 更新活跃度
        producer.active_score += 1.0
        consumer.active_score += 1.0
        
        # 将交易添加到已完成列表
        self.completed_trades.append(transaction)
        
        return transaction
    
    def apply_for_green_certification(self, producer_address: str, 
                                    total_production: float) -> bool:
        """
        申请绿能认证
        
        Args:
            producer_address: 发电方地址
            total_production: 总发电量
            
        Returns:
            是否获得认证
        """
        if producer_address not in self.users:
            raise ValueError("用户不存在")
            
        user = self.users[producer_address]
        if not user.is_producer:
            raise ValueError("只有发电方可以申请绿能认证")
            
        requirements = self.green_certification_requirements
        
        # 检查是否满足认证条件
        is_qualified = (
            user.reputation >= requirements["min_reputation"] and
            user.active_score >= requirements["min_active_score"] and
            total_production >= requirements["min_energy_production"]
        )
        
        if is_qualified:
            user.is_green_certified = True
            user.green_energy_subsidy += 20.0  # 初始绿能补贴
            
        return is_qualified
    
    def calculate_reputation(self, user_address: str) -> float:
        """
        计算用户综合信誉值
        
        Args:
            user_address: 用户地址
            
        Returns:
            用户综合信誉值
        """
        if user_address not in self.users:
            raise ValueError("用户不存在")
            
        user = self.users[user_address]
        
        # 计算交易完成率
        completed_tx_count = 0
        successful_tx_count = 0
        
        for tx in self.completed_trades:
            if tx["producer_address"] == user_address or tx["consumer_address"] == user_address:
                completed_tx_count += 1
                if tx.get("delivery_ratio", 0) >= 0.9:  # 交付率≥90%视为成功交易
                    successful_tx_count += 1
                    
        # 交易完成率权重
        if completed_tx_count > 0:
            completion_rate = successful_tx_count / completed_tx_count
            weighted_rate = completion_rate * 0.6  # 60%权重
        else:
            weighted_rate = 0
            
        # 节点活跃度权重
        active_weight = min(0.3, user.active_score / 100 * 0.3)  # 最多30%权重
        
        # 基础信誉值权重
        base_reputation_weight = user.reputation / 100 * 0.1  # 10%权重
        
        # 综合信誉值
        total_reputation = (weighted_rate + active_weight + base_reputation_weight) * 100
        
        return min(100.0, total_reputation)  # 最高100分
    
    def create_trading_strategy(self, user_address: str, is_buy_strategy: bool,
                             description: str, energy_amount: float = None, 
                             max_price_per_kwh: float = None, min_price_per_kwh: float = None,
                             valid_hours: int = 24, auto_accept_offer: bool = False) -> AutoTradingStrategy:
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
            创建的自动交易策略
        """
        if user_address not in self.users:
            raise ValueError("用户不存在")
            
        user = self.users[user_address]
        
        # 检查用户类型与策略类型是否匹配
        if is_buy_strategy and not user.is_consumer:
            raise ValueError("非用电方用户不能创建买入策略")
        if not is_buy_strategy and not user.is_producer:
            raise ValueError("非发电方用户不能创建卖出策略")
        
        # 创建交易策略
        strategy = AutoTradingStrategy(
            user_address=user_address,
            is_buy_strategy=is_buy_strategy,
            description=description,
            energy_amount=energy_amount,
            max_price_per_kwh=max_price_per_kwh,
            min_price_per_kwh=min_price_per_kwh,
            valid_hours=valid_hours,
            auto_accept_offer=auto_accept_offer
        )
        
        # 添加基本价格条件
        if max_price_per_kwh is not None:
            price_condition = PriceCondition(
                condition_type=TradingConditionType.PRICE_THRESHOLD,
                description="",  # __post_init__会设置
                max_price_per_kwh=max_price_per_kwh,
                min_price_per_kwh=min_price_per_kwh or 0.0
            )
            strategy.add_condition(price_condition)
        
        # 添加策略到用户
        user.add_trading_strategy(strategy)
        
        # 记录日志
        self.auto_trading_logs.append({
            "type": "strategy_created",
            "user_address": user_address,
            "strategy_type": "buy" if is_buy_strategy else "sell",
            "timestamp": time.time(),
            "description": description
        })
        
        return strategy
    
    def add_strategy_condition(self, user_address: str, strategy: AutoTradingStrategy, 
                            condition: TradingCondition) -> AutoTradingStrategy:
        """
        添加交易条件到策略
        
        Args:
            user_address: 用户地址
            strategy: 交易策略
            condition: 要添加的交易条件
            
        Returns:
            更新后的交易策略
        """
        if user_address not in self.users:
            raise ValueError("用户不存在")
            
        if strategy.user_address != user_address:
            raise ValueError("无权修改此策略")
        
        strategy.add_condition(condition)
        
        # 记录日志
        self.auto_trading_logs.append({
            "type": "condition_added",
            "user_address": user_address,
            "strategy_type": "buy" if strategy.is_buy_strategy else "sell",
            "condition_type": condition.condition_type.value,
            "timestamp": time.time(),
            "description": condition.description
        })
        
        return strategy
    
    def activate_strategy(self, user_address: str, strategy: AutoTradingStrategy, activate: bool = True) -> AutoTradingStrategy:
        """
        激活或停用交易策略
        
        Args:
            user_address: 用户地址
            strategy: 交易策略
            activate: 是否激活
            
        Returns:
            更新后的交易策略
        """
        if user_address not in self.users:
            raise ValueError("用户不存在")
            
        if strategy.user_address != user_address:
            raise ValueError("无权修改此策略")
        
        strategy.is_active = activate
        
        # 记录日志
        self.auto_trading_logs.append({
            "type": "strategy_status_changed",
            "user_address": user_address,
            "strategy_type": "buy" if strategy.is_buy_strategy else "sell",
            "is_active": activate,
            "timestamp": time.time()
        })
        
        return strategy
    
    def execute_auto_trading(self) -> List[Dict[str, Any]]:
        """
        执行所有活跃的自动交易策略
        
        Returns:
            自动执行的交易列表
        """
        executed_transactions = []
        
        # 处理所有活跃的买入策略
        for user_address, user in self.users.items():
            for strategy in user.trading_strategies:
                # 跳过非活跃策略
                if not strategy.is_active:
                    continue
                
                if strategy.is_buy_strategy:
                    # 买入策略: 查找匹配的挂单并下单
                    executed_tx = self._execute_buy_strategy(user, strategy)
                    if executed_tx:
                        executed_transactions.extend(executed_tx)
                else:
                    # 卖出策略: 创建新挂单或接受匹配的投标
                    executed_tx = self._execute_sell_strategy(user, strategy)
                    if executed_tx:
                        executed_transactions.extend(executed_tx)
        
        return executed_transactions
    
    def _execute_buy_strategy(self, user: User, strategy: AutoTradingStrategy) -> List[Dict[str, Any]]:
        """
        执行买入策略
        
        Args:
            user: 用户对象
            strategy: 买入策略
            
        Returns:
            执行的交易列表
        """
        if not user.is_consumer:
            return []  # 非用电方不能执行买入
            
        transactions = []
        remaining_energy = strategy.energy_amount or float('inf')
        
        # 查找匹配的挂单
        matching_listings = []
        for listing in self.energy_listings:
            # 跳过非活跃挂单
            if listing["status"] != "active" or listing["valid_until"] < time.time():
                continue
                
            # 检查是否满足所有条件
            if strategy.check_all_conditions(listing, user, self):
                matching_listings.append(listing)
        
        # 按价格排序(从低到高)
        matching_listings.sort(key=lambda l: l["price_per_kwh"])
        
        for listing in matching_listings:
            # 如果已达到目标能量,停止交易
            if remaining_energy <= 0:
                break
                
            # 确定购买数量
            bid_amount = min(remaining_energy, listing["amount"])
            
            try:
                # 创建投标
                bid = self.place_bid(user.address, listing["id"], bid_amount)
                
                # 记录日志
                self.auto_trading_logs.append({
                    "type": "auto_bid_placed",
                    "user_address": user.address,
                    "listing_id": listing["id"],
                    "amount": bid_amount,
                    "price_per_kwh": listing["price_per_kwh"],
                    "timestamp": time.time()
                })
                
                # 如果卖家设置了自动接受
                producer = self.users.get(listing["producer_address"])
                auto_accept = False
                
                for producer_strategy in producer.trading_strategies:
                    if (not producer_strategy.is_buy_strategy and producer_strategy.is_active 
                        and producer_strategy.auto_accept_offer):
                        auto_accept = True
                        break
                
                if auto_accept:
                    transaction = self.accept_bid(listing["producer_address"], bid["id"])
                    transactions.append(transaction)
                    
                    # 记录日志
                    self.auto_trading_logs.append({
                        "type": "auto_bid_accepted",
                        "producer_address": listing["producer_address"],
                        "consumer_address": user.address,
                        "amount": bid_amount,
                        "price_per_kwh": listing["price_per_kwh"],
                        "timestamp": time.time()
                    })
                    
                    # 更新剩余能量
                    remaining_energy -= bid_amount
            except Exception as e:
                # 记录错误
                self.auto_trading_logs.append({
                    "type": "auto_trading_error",
                    "user_address": user.address,
                    "error_message": str(e),
                    "timestamp": time.time()
                })
        
        return transactions
    
    def _execute_sell_strategy(self, user: User, strategy: AutoTradingStrategy) -> List[Dict[str, Any]]:
        """
        执行卖出策略
        
        Args:
            user: 用户对象
            strategy: 卖出策略
            
        Returns:
            执行的交易列表
        """
        if not user.is_producer:
            return []  # 非发电方不能执行卖出
            
        transactions = []
        
        # 检查是否已有活跃挂单
        user_has_active_listing = False
        for listing in self.energy_listings:
            if listing["producer_address"] == user.address and listing["status"] == "active":
                user_has_active_listing = True
                break
        
        # 如果没有活跃挂单且策略指定了数量和价格,创建新挂单
        if not user_has_active_listing and strategy.energy_amount is not None and strategy.min_price_per_kwh is not None:
            try:
                valid_until = time.time() + strategy.valid_hours * 3600
                
                listing = self.list_energy_for_sale(
                    user.address,
                    strategy.energy_amount,
                    strategy.min_price_per_kwh,
                    valid_until
                )
                
                # 记录日志
                self.auto_trading_logs.append({
                    "type": "auto_listing_created",
                    "user_address": user.address,
                    "amount": strategy.energy_amount,
                    "price_per_kwh": strategy.min_price_per_kwh,
                    "valid_until": valid_until,
                    "timestamp": time.time()
                })
            except Exception as e:
                # 记录错误
                self.auto_trading_logs.append({
                    "type": "auto_trading_error",
                    "user_address": user.address,
                    "error_message": str(e),
                    "timestamp": time.time()
                })
        
        # 如果设置了自动接受报价,检查有没有合适的投标
        if strategy.auto_accept_offer:
            pending_bids = []
            
            # 查找对该用户挂单的所有待处理投标
            for bid in self.energy_bids:
                if bid["status"] != "pending":
                    continue
                    
                # 查找对应的挂单
                for listing in self.energy_listings:
                    if (listing["id"] == bid["listing_id"] and 
                        listing["producer_address"] == user.address and
                        listing["status"] == "active"):
                        
                        # 检查投标价格是否满足条件
                        if strategy.min_price_per_kwh is None or bid["price_per_kwh"] >= strategy.min_price_per_kwh:
                            pending_bids.append(bid)
            
            # 按价格排序(从高到低)
            pending_bids.sort(key=lambda b: b["price_per_kwh"], reverse=True)
            
            # 接受最好的报价
            for bid in pending_bids:
                try:
                    transaction = self.accept_bid(user.address, bid["id"])
                    transactions.append(transaction)
                    
                    # 记录日志
                    self.auto_trading_logs.append({
                        "type": "auto_bid_accepted",
                        "producer_address": user.address,
                        "consumer_address": bid["consumer_address"],
                        "amount": bid["amount"],
                        "price_per_kwh": bid["price_per_kwh"],
                        "timestamp": time.time()
                    })
                    
                    # 只接受一个报价
                    break
                except Exception as e:
                    # 记录错误
                    self.auto_trading_logs.append({
                        "type": "auto_trading_error",
                        "user_address": user.address,
                        "error_message": str(e),
                        "timestamp": time.time()
                    })
        
        return transactions
    
    def get_auto_trading_logs(self, user_address: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取自动交易日志
        
        Args:
            user_address: 可选,按用户筛选
            limit: 返回的最大记录数
            
        Returns:
            自动交易日志列表
        """
        if user_address is not None:
            # 筛选特定用户的日志
            filtered_logs = [log for log in self.auto_trading_logs 
                          if log.get("user_address") == user_address or 
                             log.get("producer_address") == user_address or
                             log.get("consumer_address") == user_address]
        else:
            filtered_logs = self.auto_trading_logs
            
        # 按时间倒序排序并限制数量
        return sorted(filtered_logs, key=lambda log: log["timestamp"], reverse=True)[:limit]

    # EnergyCoin相关方法
    def start_mining(self, user_address: str) -> bool:
        """
        开始挖矿
        
        Args:
            user_address: 用户地址
            
        Returns:
            是否成功开始挖矿
        """
        if user_address not in self.users:
            raise ValueError("用户不存在")
        
        user = self.users[user_address]
        user.is_mining = True
        
        return True
    
    def stop_mining(self, user_address: str) -> bool:
        """
        停止挖矿
        
        Args:
            user_address: 用户地址
            
        Returns:
            是否成功停止挖矿
        """
        if user_address not in self.users:
            raise ValueError("用户不存在")
        
        user = self.users[user_address]
        user.is_mining = False
        
        return True
    
    def process_block(self, user_address: str) -> Dict[str, Any]:
        """
        处理区块(挖矿或验证)
        
        Args:
            user_address: 处理区块的用户地址
            
        Returns:
            处理的区块信息
        """
        if user_address not in self.users:
            raise ValueError("用户不存在")
        
        user = self.users[user_address]
        if not user.is_mining:
            raise ValueError("用户未开启挖矿功能")
        
        # 处理新区块(自动选择PoW或PoS)
        block = self.energy_coin.process_next_block(user.energy_coin_address)
        
        return {
            "block_index": block.index,
            "transactions_count": len(block.transactions),
            "timestamp": block.timestamp,
            "hash": block.hash,
            "validator": block.validator or "PoW挖矿"
        }
    
    def create_stake(self, user_address: str, amount: float, duration: int) -> Dict[str, Any]:
        """
        创建EnergyCoin质押
        
        Args:
            user_address: 用户地址
            amount: 质押金额
            duration: 质押期限(天)
            
        Returns:
            质押信息
        """
        if user_address not in self.users:
            raise ValueError("用户不存在")
        
        user = self.users[user_address]
        
        try:
            # 创建质押
            self.energy_coin.create_stake(user.energy_coin_address, amount, duration)
            
            # 获取质押信息
            stake_info = self.energy_coin.get_staking_info(user.energy_coin_address)
            
            return {
                "user_address": user_address,
                "energy_coin_address": user.energy_coin_address,
                "stake_amount": amount,
                "duration": duration,
                "timestamp": time.time(),
                "stake_info": stake_info
            }
        except Exception as e:
            raise ValueError(f"质押失败: {str(e)}")
    
    def release_stake(self, user_address: str) -> Dict[str, Any]:
        """
        释放EnergyCoin质押
        
        Args:
            user_address: 用户地址
            
        Returns:
            释放信息
        """
        if user_address not in self.users:
            raise ValueError("用户不存在")
        
        user = self.users[user_address]
        
        try:
            # 释放质押
            released_amount = self.energy_coin.release_stake(user.energy_coin_address)
            
            return {
                "user_address": user_address,
                "energy_coin_address": user.energy_coin_address,
                "released_amount": released_amount,
                "timestamp": time.time()
            }
        except Exception as e:
            raise ValueError(f"释放质押失败: {str(e)}")
    
    def get_energy_coin_balance(self, user_address: str) -> float:
        """
        获取用户EnergyCoin余额
        
        Args:
            user_address: 用户地址
            
        Returns:
            EnergyCoin余额
        """
        if user_address not in self.users:
            raise ValueError("用户不存在")
        
        user = self.users[user_address]
        
        return self.energy_coin.get_balance(user.energy_coin_address)
    
    def get_energy_coin_transactions(self, user_address: str) -> List[Dict[str, Any]]:
        """
        获取用户EnergyCoin交易记录
        
        Args:
            user_address: 用户地址
            
        Returns:
            交易记录列表
        """
        if user_address not in self.users:
            raise ValueError("用户不存在")
        
        user = self.users[user_address]
        
        return self.energy_coin.get_transactions_by_address(user.energy_coin_address)
    
    def transfer_energy_coin(self, from_address: str, to_address: str, amount: float) -> Dict[str, Any]:
        """
        转账EnergyCoin
        
        Args:
            from_address: 发送方用户地址
            to_address: 接收方用户地址
            amount: 转账金额
            
        Returns:
            转账交易信息
        """
        if from_address not in self.users:
            raise ValueError("发送方用户不存在")
        
        if to_address not in self.users:
            raise ValueError("接收方用户不存在")
        
        from_user = self.users[from_address]
        to_user = self.users[to_address]
        
        try:
            # 创建转账交易
            self.energy_coin.add_transaction(
                from_address=from_user.energy_coin_address,
                to_address=to_user.energy_coin_address,
                amount=amount,
                tx_type="transfer"
            )
            
            return {
                "from_address": from_address,
                "to_address": to_address,
                "amount": amount,
                "timestamp": time.time(),
                "status": "pending"
            }
        except Exception as e:
            raise ValueError(f"转账失败: {str(e)}")
    
    def get_energy_coin_stats(self) -> Dict[str, Any]:
        """
        获取EnergyCoin统计信息
        
        Returns:
            统计信息
        """
        total_supply = self.energy_coin.get_total_supply()
        total_staked = self.energy_coin.get_staked_amount()
        active_miners = sum(1 for user in self.users.values() if user.is_mining)
        
        return {
            "total_supply": total_supply,
            "total_staked": total_staked,
            "staking_ratio": total_staked / total_supply if total_supply > 0 else 0,
            "active_miners": active_miners,
            "block_count": len(self.energy_coin.chain),
            "difficulty": self.energy_coin.pow_difficulty,
            "mining_reward": self.energy_coin.pow_reward,
            "staking_reward": self.energy_coin.pos_reward
        }
    
    def pay_energy_bill(self, user_address: str, amount: float) -> Dict[str, Any]:
        """
        使用EnergyCoin支付电费
        
        Args:
            user_address: 用户地址
            amount: 电费金额(EnergyCoin)
            
        Returns:
            支付信息
        """
        if user_address not in self.users:
            raise ValueError("用户不存在")
        
        user = self.users[user_address]
        
        try:
            # 创建支付电费交易
            self.energy_coin.add_transaction(
                from_address=user.energy_coin_address,
                to_address="utility_company",
                amount=amount,
                tx_type="bill_payment"
            )
            
            return {
                "user_address": user_address,
                "amount": amount,
                "timestamp": time.time(),
                "status": "pending"
            }
        except Exception as e:
            raise ValueError(f"支付失败: {str(e)}")