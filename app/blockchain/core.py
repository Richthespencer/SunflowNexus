import hashlib
import json
import time
from typing import List, Dict, Any


class Block:
    """区块类，表示区块链中的一个区块"""
    def __init__(self, index: int, transactions: List[Dict[str, Any]], timestamp: float, previous_hash: str, nonce: int = 0):
        """
        初始化一个新的区块
        
        Args:
            index: 区块索引/高度
            transactions: 区块中包含的交易列表
            timestamp: 区块创建的时间戳
            previous_hash: 前一个区块的哈希
            nonce: 用于工作量证明的随机数
        """
        self.index = index
        self.transactions = transactions
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """
        计算当前区块的哈希值
        
        Returns:
            str: 区块的哈希值
        """
        block_string = json.dumps({
            "index": self.index,
            "transactions": self.transactions,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True).encode()
        
        return hashlib.sha256(block_string).hexdigest()
    
    def mine_block(self, difficulty: int) -> None:
        """
        挖矿 - 找到满足难度要求的哈希值
        
        Args:
            difficulty: 工作量证明的难度 (前导0的个数)
        """
        target = "0" * difficulty
        
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将区块转换为字典格式
        
        Returns:
            Dict: 区块的字典表示
        """
        return {
            "index": self.index,
            "transactions": self.transactions,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash": self.hash
        }


class Blockchain:
    """区块链类，管理区块链的核心操作"""
    def __init__(self, difficulty: int = 4):
        """
        初始化一个新的区块链
        
        Args:
            difficulty: 挖矿难度
        """
        self.chain = []
        self.pending_transactions = []
        self.difficulty = difficulty
        self.mining_reward = 10  # 挖矿奖励
        self.create_genesis_block()
    
    def create_genesis_block(self) -> None:
        """创建并添加创世区块"""
        genesis_block = Block(0, [], time.time(), "0")
        genesis_block.hash = genesis_block.calculate_hash()
        self.chain.append(genesis_block)
    
    def get_latest_block(self) -> Block:
        """
        获取最新区块
        
        Returns:
            Block: 区块链中的最新区块
        """
        return self.chain[-1]
    
    def mine_pending_transactions(self, miner_address: str) -> Block:
        """
        挖掘待处理的交易并创建新区块
        
        Args:
            miner_address: 接收挖矿奖励的地址
            
        Returns:
            Block: 新挖出的区块
        """
        # 添加挖矿奖励交易
        self.pending_transactions.append({
            "from": "network",
            "to": miner_address,
            "amount": self.mining_reward,
            "timestamp": time.time(),
            "type": "mining_reward"
        })
        
        # 创建新区块
        block = Block(
            index=len(self.chain),
            transactions=self.pending_transactions,
            timestamp=time.time(),
            previous_hash=self.get_latest_block().hash
        )
        
        # 挖矿
        block.mine_block(self.difficulty)
        
        # 将新区块添加到链上
        self.chain.append(block)
        
        # 清空待处理的交易，为下一个区块准备
        self.pending_transactions = []
        
        return block
    
    def add_transaction(self, transaction: Dict[str, Any]) -> int:
        """
        添加交易到待处理列表
        
        Args:
            transaction: 要添加的交易
            
        Returns:
            int: 将包含此交易的区块索引
        """
        # 验证交易
        if not transaction.get("from") or not transaction.get("to") or not transaction.get("amount"):
            return -1
        
        # 为交易添加时间戳（如果没有）
        if "timestamp" not in transaction:
            transaction["timestamp"] = time.time()
        
        # 添加到待处理交易
        self.pending_transactions.append(transaction)
        
        # 返回交易将被添加到的区块的索引
        return self.get_latest_block().index + 1
    
    def is_chain_valid(self) -> bool:
        """
        验证整个区块链的完整性
        
        Returns:
            bool: 如果区块链有效返回True，否则返回False
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
    
    def get_balance(self, address: str) -> float:
        """
        计算特定地址的余额
        
        Args:
            address: 要查询余额的地址
            
        Returns:
            float: 地址的当前余额
        """
        balance = 0
        
        # 遍历所有区块中的交易
        for block in self.chain:
            for transaction in block.transactions:
                if transaction.get("to") == address:
                    balance += transaction.get("amount", 0)
                if transaction.get("from") == address:
                    balance -= transaction.get("amount", 0)
        
        return balance
    
    def get_transactions_by_address(self, address: str) -> List[Dict[str, Any]]:
        """
        获取特定地址的所有交易
        
        Args:
            address: 要查询交易的地址
            
        Returns:
            List: 地址相关的所有交易列表
        """
        transactions = []
        
        # 遍历所有区块中的交易
        for block in self.chain:
            for transaction in block.transactions:
                if transaction.get("from") == address or transaction.get("to") == address:
                    tx = transaction.copy()
                    tx["block"] = block.index
                    tx["block_hash"] = block.hash
                    transactions.append(tx)
        
        return transactions
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将整个区块链转换为字典格式
        
        Returns:
            Dict: 区块链的字典表示
        """
        return {
            "chain": [block.to_dict() for block in self.chain],
            "pending_transactions": self.pending_transactions,
            "difficulty": self.difficulty,
            "mining_reward": self.mining_reward
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Blockchain':
        """
        从字典创建区块链
        
        Args:
            data: 区块链字典表示
            
        Returns:
            Blockchain: 新创建的区块链实例
        """
        blockchain = cls(difficulty=data.get("difficulty", 4))
        blockchain.mining_reward = data.get("mining_reward", 10)
        
        # 清除默认创建的创世区块
        blockchain.chain = []
        
        # 重建区块链
        for block_data in data.get("chain", []):
            block = Block(
                index=block_data.get("index"),
                transactions=block_data.get("transactions"),
                timestamp=block_data.get("timestamp"),
                previous_hash=block_data.get("previous_hash"),
                nonce=block_data.get("nonce")
            )
            block.hash = block_data.get("hash") or block.calculate_hash()
            blockchain.chain.append(block)
        
        blockchain.pending_transactions = data.get("pending_transactions", [])
        
        return blockchain