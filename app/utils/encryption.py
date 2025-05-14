import os
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64
from typing import Dict, Tuple, Any, Optional, Union, List


class AsymmetricEncryption:
    """非对称加密模块"""
    
    def __init__(self, key_size: int = 2048):
        """
        初始化非对称加密模块
        
        Args:
            key_size: RSA密钥大小，默认2048位
        """
        self.key_size = key_size
        
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """
        生成RSA密钥对
        
        Returns:
            Tuple[bytes, bytes]: (私钥, 公钥)
        """
        # 生成私钥
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size,
            backend=default_backend()
        )
        
        # 获取公钥
        public_key = private_key.public_key()
        
        # 将密钥序列化为PEM格式
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def encrypt(self, public_key_pem: bytes, data: bytes) -> bytes:
        """
        使用公钥加密数据
        
        Args:
            public_key_pem: PEM格式的公钥
            data: 要加密的数据
            
        Returns:
            加密后的数据
        """
        # 加载公钥
        public_key = serialization.load_pem_public_key(
            public_key_pem,
            backend=default_backend()
        )
        
        # 由于RSA加密有大小限制，这里使用分段加密
        if len(data) > self.key_size // 16:
            # 对于大数据，使用对称加密，然后用RSA加密对称密钥
            aes_key = os.urandom(32)  # 256位AES密钥
            iv = os.urandom(16)  # 初始化向量
            
            # 使用AES加密数据
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            
            # PKCS7填充
            padded_data = self._pad_data(data, 16)
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # 使用RSA加密AES密钥
            encrypted_key = public_key.encrypt(
                aes_key + iv,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # 组合加密密钥和加密数据
            result = base64.b64encode(encrypted_key) + b'.' + base64.b64encode(encrypted_data)
            return result
        else:
            # 小数据直接使用RSA
            encrypted = public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return base64.b64encode(encrypted)
    
    def decrypt(self, private_key_pem: bytes, encrypted_data: bytes) -> bytes:
        """
        使用私钥解密数据
        
        Args:
            private_key_pem: PEM格式的私钥
            encrypted_data: 加密数据
            
        Returns:
            解密后的数据
        """
        # 加载私钥
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None,
            backend=default_backend()
        )
        
        # 检查是否是混合加密
        if b'.' in encrypted_data:
            # 混合加密模式
            encrypted_key_b64, encrypted_data_b64 = encrypted_data.split(b'.')
            encrypted_key = base64.b64decode(encrypted_key_b64)
            encrypted_data_content = base64.b64decode(encrypted_data_b64)
            
            # 解密AES密钥和IV
            key_iv = private_key.decrypt(
                encrypted_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            aes_key = key_iv[:32]
            iv = key_iv[32:]
            
            # 使用AES解密数据
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(encrypted_data_content) + decryptor.finalize()
            
            # 去除PKCS7填充
            return self._unpad_data(padded_data)
        else:
            # 直接RSA加密
            encrypted = base64.b64decode(encrypted_data)
            decrypted = private_key.decrypt(
                encrypted,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return decrypted
        
    def _pad_data(self, data: bytes, block_size: int) -> bytes:
        """
        对数据进行PKCS7填充
        
        Args:
            data: 原始数据
            block_size: 块大小
            
        Returns:
            填充后的数据
        """
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _unpad_data(self, padded_data: bytes) -> bytes:
        """
        去除PKCS7填充
        
        Args:
            padded_data: 填充后的数据
            
        Returns:
            去除填充后的数据
        """
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    

class DataEncryptor:
    """数据加密器，用于处理系统中的数据加密需求"""
    
    def __init__(self):
        """初始化数据加密器"""
        self.asymmetric_encryption = AsymmetricEncryption()
        self.keypairs: Dict[str, Tuple[bytes, bytes]] = {}  # 用户地址 -> (私钥, 公钥)
        
    def generate_user_keypair(self, user_address: str) -> bytes:
        """
        为用户生成密钥对
        
        Args:
            user_address: 用户地址
            
        Returns:
            用户公钥
        """
        private_key, public_key = self.asymmetric_encryption.generate_keypair()
        self.keypairs[user_address] = (private_key, public_key)
        return public_key
    
    def encrypt_energy_data(self, user_address: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        加密能源数据
        
        Args:
            user_address: 用户地址
            data: 能源数据
            
        Returns:
            加密后的数据
        """
        if user_address not in self.keypairs:
            raise ValueError(f"用户 {user_address} 未生成密钥对")
            
        _, public_key = self.keypairs[user_address]
        
        encrypted_data = {}
        for key, value in data.items():
            # 将值转换为字符串并编码
            value_str = str(value).encode('utf-8')
            # 加密值
            encrypted_value = self.asymmetric_encryption.encrypt(public_key, value_str)
            encrypted_data[key] = encrypted_value.decode('utf-8')
            
        return encrypted_data
    
    def decrypt_energy_data(self, user_address: str, encrypted_data: Dict[str, str]) -> Dict[str, Any]:
        """
        解密能源数据
        
        Args:
            user_address: 用户地址
            encrypted_data: 加密的数据
            
        Returns:
            解密后的数据
        """
        if user_address not in self.keypairs:
            raise ValueError(f"用户 {user_address} 未生成密钥对")
            
        private_key, _ = self.keypairs[user_address]
        
        decrypted_data = {}
        for key, value in encrypted_data.items():
            # 解码加密值
            encrypted_value = value.encode('utf-8')
            # 解密值
            decrypted_value = self.asymmetric_encryption.decrypt(private_key, encrypted_value)
            
            # 尝试将解密后的值转换为原始类型
            value_str = decrypted_value.decode('utf-8')
            try:
                # 尝试转换为数字
                if '.' in value_str:
                    value_converted = float(value_str)
                else:
                    value_converted = int(value_str)
            except ValueError:
                # 如果不是数字，保持字符串
                value_converted = value_str
                
            decrypted_data[key] = value_converted
            
        return decrypted_data
    
    def sign_transaction(self, user_address: str, transaction_data: Dict[str, Any]) -> str:
        """
        对交易进行签名
        
        Args:
            user_address: 用户地址
            transaction_data: 交易数据
            
        Returns:
            签名字符串
        """
        if user_address not in self.keypairs:
            raise ValueError(f"用户 {user_address} 未生成密钥对")
            
        private_key_pem, _ = self.keypairs[user_address]
        
        # 加载私钥
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None,
            backend=default_backend()
        )
        
        # 将交易数据转换为字符串并编码
        tx_str = str(transaction_data).encode('utf-8')
        
        # 计算签名
        signature = private_key.sign(
            tx_str,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # 返回Base64编码的签名
        return base64.b64encode(signature).decode('utf-8')
    
    def verify_signature(self, user_address: str, transaction_data: Dict[str, Any], 
                       signature: str) -> bool:
        """
        验证交易签名
        
        Args:
            user_address: 用户地址
            transaction_data: 交易数据
            signature: 签名字符串
            
        Returns:
            签名是否有效
        """
        if user_address not in self.keypairs:
            raise ValueError(f"用户 {user_address} 未生成密钥对")
            
        _, public_key_pem = self.keypairs[user_address]
        
        # 加载公钥
        public_key = serialization.load_pem_public_key(
            public_key_pem,
            backend=default_backend()
        )
        
        # 将交易数据转换为字符串并编码
        tx_str = str(transaction_data).encode('utf-8')
        
        # 解码签名
        signature_bytes = base64.b64decode(signature)
        
        try:
            # 验证签名
            public_key.verify(
                signature_bytes,
                tx_str,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False