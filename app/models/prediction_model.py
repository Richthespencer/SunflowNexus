import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Dict, Any, Optional
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class LSTMModel(nn.Module):
    """LSTM神经网络模型用于时间序列预测"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        """
        初始化LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: LSTM层数
            output_size: 输出维度
            dropout: Dropout比率，防止过拟合
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_size]
            
        Returns:
            输出张量，形状为 [batch_size, output_size]
        """
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向计算
        out, _ = self.lstm(x, (h0, c0))
        
        # 只使用最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


class TimeSeriesForecaster:
    """时间序列预测器，用于光伏发电和用电负荷预测"""
    
    def __init__(self, sequence_length: int = 24, forecast_horizon: int = 24):
        """
        初始化预测器
        
        Args:
            sequence_length: 用于预测的历史数据长度（小时）
            forecast_horizon: 预测时域（小时）
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列的输入-输出对
        
        Args:
            data: 原始时间序列数据
            
        Returns:
            输入序列和对应的目标值
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            # 输入序列
            X.append(data[i:i+self.sequence_length])
            # 目标序列
            y.append(data[i+self.sequence_length:i+self.sequence_length+self.forecast_horizon])
            
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame, feature_columns: List[str], target_column: str, 
              epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001, 
              hidden_size: int = 64, num_layers: int = 2) -> Dict[str, Any]:
        """
        训练LSTM模型
        
        Args:
            data: 包含特征和目标的DataFrame
            feature_columns: 特征列名
            target_column: 目标列名
            epochs: 训练轮数
            batch_size: 批量大小
            learning_rate: 学习率
            hidden_size: LSTM隐藏层维度
            num_layers: LSTM层数
            
        Returns:
            训练历史记录
        """
        # 准备特征和目标数据
        features = data[feature_columns].values
        targets = data[target_column].values.reshape(-1, 1)
        
        # 特征标准化
        scaled_features = self.scaler.fit_transform(features)
        scaled_targets = self.scaler.fit_transform(targets)
        
        # 创建序列
        X, y = self.create_sequences(scaled_targets)
        
        # 转换为Pytorch张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # 创建数据加载器
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 初始化模型
        input_size = 1  # 如果只使用目标变量历史值
        output_size = self.forecast_horizon
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size
        ).to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练历史记录
        history = {
            'loss': []
        }
        
        # 训练模型
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                # 前向传播
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y.reshape(-1, self.forecast_horizon))
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # 记录每轮的损失
            avg_loss = epoch_loss / len(dataloader)
            history['loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
                
        return history
    
    def predict(self, data: pd.DataFrame, feature_columns: List[str], target_column: str) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        Args:
            data: 包含特征和目标的DataFrame
            feature_columns: 特征列名
            target_column: 目标列名
            
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        try:
            print(f"开始预测处理, 数据点数: {len(data)}")
            # 准备数据
            targets = data[target_column].values.reshape(-1, 1)
            
            if len(targets) < self.sequence_length:
                raise ValueError(f"数据点不足，需要至少 {self.sequence_length} 个数据点，但只有 {len(targets)} 个")
            
            # 标准化数据
            print("标准化数据...")
            scaled_targets = self.scaler.fit_transform(targets)
            
            # 获取最后一个序列作为输入
            print(f"准备预测输入序列, 长度={self.sequence_length}")
            last_sequence = scaled_targets[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            
            # 验证输入形状
            if last_sequence.shape != (1, self.sequence_length, 1):
                print(f"警告：输入形状不符 {last_sequence.shape}，尝试重新调整")
                last_sequence = last_sequence.reshape(1, self.sequence_length, 1)
                
            X_tensor = torch.FloatTensor(last_sequence).to(self.device)
            
            # 预测
            print("执行模型预测...")
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(X_tensor).cpu().numpy()
                
            # 检查预测结果是否有效
            if np.isnan(prediction).any() or np.isinf(prediction).any():
                raise ValueError("预测结果包含NaN或Inf值")
                
            # 反标准化
            print("处理预测结果...")
            prediction_reshaped = prediction.reshape(-1, 1)
            prediction_original = self.scaler.inverse_transform(prediction_reshaped)
            
            # 确保预测值在合理范围内
            prediction_final = np.clip(prediction_original.flatten(), 0, None)  # 负载和发电量不可能为负
            
            print(f"预测完成，结果包含 {len(prediction_final)} 个数据点")
            return prediction_final
        except Exception as e:
            import traceback
            print(f"预测过程出错: {str(e)}")
            traceback.print_exc()
            raise RuntimeError(f"预测失败: {str(e)}")
    
    def plot_prediction(self, actual: np.ndarray, predicted: np.ndarray, title: str = "预测结果") -> plt.Figure:
        """
        可视化预测结果
        
        Args:
            actual: 实际值
            predicted: 预测值
            title: 图表标题
            
        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制实际值和预测值
        ax.plot(actual, label="实际值", color="blue")
        ax.plot(range(len(actual) - len(predicted), len(actual)), predicted, label="预测值", color="red")
        
        # 添加标题和标签
        ax.set_title(title)
        ax.set_xlabel("时间")
        ax.set_ylabel("数值")
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def save_model(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon
        }, path)
        
    def load_model(self, path: str) -> None:
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path)
        
        self.sequence_length = checkpoint['sequence_length']
        self.forecast_horizon = checkpoint['forecast_horizon']
        self.scaler = checkpoint['scaler']
        
        # 重新创建模型
        input_size = 1
        output_size = self.forecast_horizon
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            output_size=output_size
        ).to(self.device)
        
        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()


class SolarPowerForecaster(TimeSeriesForecaster):
    """光伏发电预测器，继承自TimeSeriesForecaster"""
    
    def __init__(self, sequence_length: int = 24, forecast_horizon: int = 24):
        super().__init__(sequence_length, forecast_horizon)
        
    def prepare_weather_features(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        准备天气特征数据
        
        Args:
            weather_data: 天气数据DataFrame，包含温度、辐照度、云量等
            
        Returns:
            处理后的特征DataFrame
        """
        # 添加时间特征
        weather_data['hour'] = weather_data.index.hour
        weather_data['day_of_year'] = weather_data.index.dayofyear
        weather_data['month'] = weather_data.index.month
        
        # 计算太阳高度角（简化版）
        weather_data['solar_angle'] = np.sin(2 * np.pi * weather_data['day_of_year'] / 365) * \
                                     np.sin(np.pi * weather_data['hour'] / 12)
        
        return weather_data
        
    def train_with_weather(self, power_data: pd.DataFrame, weather_data: pd.DataFrame,
                         power_column: str = 'power_output', **kwargs) -> Dict[str, Any]:
        """
        结合天气数据训练光伏发电预测模型
        
        Args:
            power_data: 发电量数据
            weather_data: 天气数据
            power_column: 发电量列名
            **kwargs: 其他训练参数
            
        Returns:
            训练历史记录
        """
        try:
            print(f"开始训练模型，发电数据点: {len(power_data)}，天气数据点: {len(weather_data)}")
            
            # 准备天气特征
            weather_features = self.prepare_weather_features(weather_data)
            
            # 合并发电量和天气数据
            combined_data = pd.merge(power_data, weather_features, left_index=True, right_index=True, how='inner')
            
            if len(combined_data) < self.sequence_length + self.forecast_horizon:
                raise ValueError(f"合并后的数据不足，需要至少 {self.sequence_length + self.forecast_horizon} 个数据点")
                
            print(f"数据合并后保留 {len(combined_data)} 个有效数据点")
            
            # 确定特征列 - 只使用实际存在的列
            feature_columns = []
            # 检查并添加基本特征
            for col in ['temperature', 'irradiance']:
                if col in combined_data.columns:
                    feature_columns.append(col)
            
            # 添加计算出的特征
            for col in ['hour', 'day_of_year', 'month', 'solar_angle']:
                if col in combined_data.columns:
                    feature_columns.append(col)
            
            if not feature_columns:
                raise ValueError("没有可用的特征列进行训练")
            
            print(f"使用特征列: {feature_columns}")
            
            # 设置默认训练参数，如果没有提供
            if 'epochs' not in kwargs:
                kwargs['epochs'] = 5  # 默认减少到5轮
            if 'batch_size' not in kwargs:
                kwargs['batch_size'] = 16
            if 'learning_rate' not in kwargs:
                kwargs['learning_rate'] = 0.001
                
            print(f"使用参数: epochs={kwargs.get('epochs')}, batch_size={kwargs.get('batch_size')}, learning_rate={kwargs.get('learning_rate')}")
            
            # 调用父类的训练方法
            return self.train(combined_data, feature_columns, power_column, **kwargs)
        except Exception as e:
            import traceback
            print(f"训练过程出错: {str(e)}")
            traceback.print_exc()
            raise RuntimeError(f"模型训练失败: {str(e)}")


class LoadForecaster(TimeSeriesForecaster):
    """用电负荷预测器，继承自TimeSeriesForecaster"""
    
    def __init__(self, sequence_length: int = 24, forecast_horizon: int = 24):
        super().__init__(sequence_length, forecast_horizon)
        
    def prepare_load_features(self, load_data: pd.DataFrame) -> pd.DataFrame:
        """
        准备负荷特征数据
        
        Args:
            load_data: 负荷数据DataFrame
            
        Returns:
            处理后的特征DataFrame
        """
        # 添加时间特征
        load_data['hour'] = load_data.index.hour
        load_data['day_of_week'] = load_data.index.dayofweek
        load_data['is_weekend'] = load_data['day_of_week'].isin([5, 6]).astype(int)
        load_data['month'] = load_data.index.month
        
        # 计算移动平均
        load_data['load_ma_24h'] = load_data['load'].rolling(window=24, min_periods=1).mean()
        
        return load_data
        
    def train_with_user_profile(self, load_data: pd.DataFrame, user_profile: Dict[str, Any],
                              load_column: str = 'load', **kwargs) -> Dict[str, Any]:
        """
        结合用户画像训练负荷预测模型
        
        Args:
            load_data: 负荷数据
            user_profile: 用户画像数据
            load_column: 负荷列名
            **kwargs: 其他训练参数
            
        Returns:
            训练历史记录
        """
        try:
            print(f"开始训练负荷预测模型，数据点: {len(load_data)}")
            
            # 准备负荷特征
            load_features = self.prepare_load_features(load_data)
            
            if len(load_features) < self.sequence_length + self.forecast_horizon:
                raise ValueError(f"数据不足，需要至少 {self.sequence_length + self.forecast_horizon} 个数据点")
            
            # 处理用户画像特征
            processed_user_profile = {}
            for key, value in user_profile.items():
                # 处理字符串类型的用户类型
                if key == 'user_type':
                    # 将用户类型转换为数值
                    if value == "住宅":
                        processed_user_profile['user_type_residential'] = 1
                        processed_user_profile['user_type_commercial'] = 0
                        processed_user_profile['user_type_industrial'] = 0
                    elif value == "商业":
                        processed_user_profile['user_type_residential'] = 0
                        processed_user_profile['user_type_commercial'] = 1
                        processed_user_profile['user_type_industrial'] = 0
                    elif value == "工业":
                        processed_user_profile['user_type_residential'] = 0
                        processed_user_profile['user_type_commercial'] = 0
                        processed_user_profile['user_type_industrial'] = 1
                    else:
                        # 默认为住宅
                        processed_user_profile['user_type_residential'] = 1
                        processed_user_profile['user_type_commercial'] = 0
                        processed_user_profile['user_type_industrial'] = 0
                else:
                    # 对于其他特征，确保它们是数值类型
                    try:
                        processed_user_profile[key] = float(value)
                    except (ValueError, TypeError):
                        print(f"警告：跳过非数值特征 {key}={value}")
            
            # 添加处理后的用户画像特征
            for key, value in processed_user_profile.items():
                load_features[key] = value
                
            print(f"处理后的数据包含 {len(load_features)} 个有效数据点")
                
            # 确定特征列（更新为处理后的特征名称）
            feature_columns = ['hour', 'day_of_week', 'is_weekend', 'load_ma_24h']
            feature_columns.extend(list(processed_user_profile.keys()))
            
            # 输出使用的特征
            print(f"使用特征列: {feature_columns}")
            
            # 设置默认训练参数，如果没有提供
            if 'epochs' not in kwargs:
                kwargs['epochs'] = 5  # 默认减少到5轮
            if 'batch_size' not in kwargs:
                kwargs['batch_size'] = 16
            if 'learning_rate' not in kwargs:
                kwargs['learning_rate'] = 0.001
                
            print(f"使用参数: epochs={kwargs.get('epochs')}, batch_size={kwargs.get('batch_size')}, learning_rate={kwargs.get('learning_rate')}")
            
            # 调用父类的训练方法
            return self.train(load_features, feature_columns, load_column, **kwargs)
        except Exception as e:
            import traceback
            print(f"训练过程出错: {str(e)}")
            traceback.print_exc()
            raise RuntimeError(f"负荷预测模型训练失败: {str(e)}")