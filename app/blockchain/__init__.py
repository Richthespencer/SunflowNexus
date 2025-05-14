import time
import hashlib
import json
from typing import List, Dict, Any, Optional


class Block:
    def __init__(self, index: int, previous_hash: str, timestamp: float, data: Dict[str, Any], nonce: int = 0):
        """
        区块初始化
        
        Args:
            index: 区块索引
            previous_hash: 前一个区块的哈希值
            timestamp: 时间戳
            data: 交易数据
            nonce: 用于工作量证明的随机数
        """
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.nonce = nonce
        self.hash = self.calculate_hash()
        
    def calculate_hash(self) -> str:
        """计算当前区块的哈希值"""
        block_string = json.dumps({
            "index": self.index,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "data": self.data,
            "nonce": self.nonce
        }, sort_keys=True).encode()
        
        return hashlib.sha256(block_string).hexdigest()
    
    def mine_block(self, difficulty: int) -> None:
        """
        挖矿过程 - 工作量证明
        
        Args:
            difficulty: 难度系数，表示哈希前几位为0
        """
        target = '0' * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
            
    def to_dict(self) -> Dict[str, Any]:
        """将区块转换为字典格式"""
        return {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "data": self.data,
            "nonce": self.nonce,
            "hash": self.hash
        }


class Blockchain:
    def __init__(self, difficulty: int = 4):
        """
        区块链初始化
        
        Args:
            difficulty: 挖矿难度
        """
        self.chain: List[Block] = []
        self.difficulty = difficulty
        self.pending_transactions: List[Dict[str, Any]] = []
        self.mining_reward = 100  # 挖矿奖励
        
        # 创建创世区块
        self.create_genesis_block()
        
    def create_genesis_block(self) -> None:
        """创建创世区块"""
        genesis_block = Block(0, "0", time.time(), {"message": "Genesis Block"})
        genesis_block.mine_block(self.difficulty)
        self.chain.append(genesis_block)
        
    def get_latest_block(self) -> Block:
        """获取最新区块"""
        return self.chain[-1]
    
    def mine_pending_transactions(self, mining_reward_address: str) -> Block:
        """
        挖掘待处理的交易
        
        Args:
            mining_reward_address: 接收挖矿奖励的地址
        
        Returns:
            新挖出的区块
        """
        # 创建包含所有待处理交易的新区块
        block = Block(
            index=len(self.chain),
            previous_hash=self.get_latest_block().hash,
            timestamp=time.time(),
            data={
                "transactions": self.pending_transactions
            }
        )
        
        block.mine_block(self.difficulty)
        
        # 将新区块添加到链中
        self.chain.append(block)
        
        # 重置待处理交易列表并发送挖矿奖励
        self.pending_transactions = [{
            "from": "system",
            "to": mining_reward_address,
            "amount": self.mining_reward,
            "type": "mining_reward",
            "timestamp": time.time()
        }]
        
        return block
    
    def create_transaction(self, transaction: Dict[str, Any]) -> int:
        """
        创建新交易
        
        Args:
            transaction: 交易数据
            
        Returns:
            将包含此交易的区块索引
        """
        self.pending_transactions.append(transaction)
        return self.get_latest_block().index + 1
    
    def get_balance(self, address: str) -> float:
        """
        获取地址余额
        
        Args:
            address: 用户地址
            
        Returns:
            地址余额
        """
        balance = 0
        
        # 遍历所有区块
        for block in self.chain:
            if 'transactions' in block.data:
                for tx in block.data['transactions']:
                    # 如果地址是发送方，减少余额
                    if tx.get('from') == address:
                        balance -= tx.get('amount', 0)
                    
                    # 如果地址是接收方，增加余额
                    if tx.get('to') == address:
                        balance += tx.get('amount', 0)
        
        return balance
    
    def is_chain_valid(self) -> bool:
        """
        验证区块链是否有效
        
        Returns:
            区块链是否有效
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # 验证当前区块哈希是否正确
            if current_block.hash != current_block.calculate_hash():
                return False
            
            # 验证当前区块的前一个哈希是否指向前一个区块的哈希
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True
    
    def to_dict(self) -> List[Dict[str, Any]]:
        """将区块链转换为字典列表格式"""
        return [block.to_dict() for block in self.chain]