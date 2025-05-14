import pandas as pd
import numpy as np
import time
import datetime
import random
from typing import Dict, List, Any, Optional, Tuple


class EnergyDataGenerator:
    """能源数据生成器，模拟从物联网设备采集的发电和用电数据"""

    def __init__(self, location: str = "北京", seed: int = None):
        """
        初始化能源数据生成器
        
        Args:
            location: 位置信息，影响光照模式
            seed: 随机种子，用于可重复生成
        """
        self.location = location
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        
        # 位置对应的光照强度参数
        self.location_params = {
            "北京": {"max_irradiance": 1000, "summer_peak": 0.9, "winter_peak": 0.5},
            "上海": {"max_irradiance": 950, "summer_peak": 0.85, "winter_peak": 0.55},
            "广州": {"max_irradiance": 1050, "summer_peak": 0.95, "winter_peak": 0.7},
            "成都": {"max_irradiance": 900, "summer_peak": 0.8, "winter_peak": 0.45}
        }
        
        if location not in self.location_params:
            # 默认使用北京参数
            self.location_params[location] = self.location_params["北京"]
            
    def generate_solar_irradiance(self, date_time: datetime.datetime) -> float:
        """
        生成太阳辐照度数据
        
        Args:
            date_time: 日期时间
            
        Returns:
            辐照度值 (W/m²)
        """
        params = self.location_params.get(self.location, self.location_params["北京"])
        max_irradiance = params["max_irradiance"]
        
        # 基于日期时间计算季节因子
        day_of_year = date_time.timetuple().tm_yday
        season_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # 夏季和冬季峰值
        if season_factor > 0.5:  # 夏季
            peak_factor = params["summer_peak"]
        else:  # 冬季
            peak_factor = params["winter_peak"]
            
        # 基于小时计算日内辐照度
        hour = date_time.hour + date_time.minute / 60
        if 6 <= hour <= 18:  # 白天
            hour_factor = np.sin(np.pi * (hour - 6) / 12)
            
            # 加入随机波动（云层等天气影响）
            cloud_factor = 0.7 + 0.3 * np.random.random()
            
            irradiance = max_irradiance * season_factor * peak_factor * hour_factor * cloud_factor
            return max(0, irradiance)
        else:
            return 0.0  # 夜间无太阳辐照
        
    def generate_temperature(self, date_time: datetime.datetime) -> float:
        """
        生成温度数据
        
        Args:
            date_time: 日期时间
            
        Returns:
            温度值 (°C)
        """
        # 基于日期时间计算季节因子
        day_of_year = date_time.timetuple().tm_yday
        season_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # 基于季节设定基础温度
        if season_factor > 0.75:  # 夏季
            base_temp = 30
        elif season_factor > 0.5:  # 春季
            base_temp = 20
        elif season_factor > 0.25:  # 秋季
            base_temp = 15
        else:  # 冬季
            base_temp = 0
            
        # 基于小时计算日内温度变化
        hour = date_time.hour + date_time.minute / 60
        daily_variation = 5 * np.sin(np.pi * (hour - 4) / 12)
        
        # 加入随机波动
        random_factor = 2 * np.random.random() - 1  # -1到1之间
        
        temperature = base_temp + daily_variation + random_factor
        return temperature
        
    def generate_solar_power_output(self, date_time: datetime.datetime, 
                                  panel_capacity: float = 5.0, efficiency: float = 0.18) -> float:
        """
        生成光伏发电输出
        
        Args:
            date_time: 日期时间
            panel_capacity: 光伏板容量 (kW)
            efficiency: 光伏板效率
            
        Returns:
            发电输出 (kW)
        """
        irradiance = self.generate_solar_irradiance(date_time)
        temperature = self.generate_temperature(date_time)
        
        # 考虑温度对效率的影响
        temp_coefficient = -0.004  # 每增加1度降低0.4%的效率
        temperature_effect = 1 + temp_coefficient * (temperature - 25)  # 25°C为参考温度
        
        # 计算实际发电功率
        power = panel_capacity * (irradiance / 1000) * efficiency * temperature_effect
        
        # 加入随机波动（设备损耗、灰尘等）
        random_factor = 0.9 + 0.1 * np.random.random()
        
        # 确保功率不为负
        return max(0, power * random_factor)
    
    def generate_load_profile(self, date_time: datetime.datetime, user_type: str = "住宅",
                           base_load: float = 2.0) -> float:
        """
        生成用电负荷曲线
        
        Args:
            date_time: 日期时间
            user_type: 用户类型 ("住宅", "商业", "工业")
            base_load: 基础负荷 (kW)
            
        Returns:
            负荷值 (kW)
        """
        hour = date_time.hour
        day_of_week = date_time.weekday()  # 0-6, 0是周一
        is_weekend = day_of_week >= 5
        
        # 不同用户类型的负荷模式
        if user_type == "住宅":
            if is_weekend:
                # 周末模式：早上和晚上用电高峰
                if 7 <= hour <= 10:  # 早上
                    factor = 0.8 + 0.4 * np.sin(np.pi * (hour - 7) / 3)
                elif 17 <= hour <= 23:  # 晚上
                    factor = 0.9 + 0.6 * np.sin(np.pi * (hour - 17) / 6)
                elif 0 <= hour <= 6:  # 深夜
                    factor = 0.3
                else:  # 其他时间
                    factor = 0.6
            else:
                # 工作日模式：早上和晚上用电高峰
                if 6 <= hour <= 9:  # 早上
                    factor = 0.9 + 0.5 * np.sin(np.pi * (hour - 6) / 3)
                elif 17 <= hour <= 22:  # 晚上
                    factor = 1.0 + 0.7 * np.sin(np.pi * (hour - 17) / 5)
                elif 0 <= hour <= 5:  # 深夜
                    factor = 0.2
                else:  # 其他时间
                    factor = 0.5
        elif user_type == "商业":
            # 商业负荷：工作日上班时间高，周末较低
            if is_weekend:
                if 10 <= hour <= 18:  # 营业时间
                    factor = 0.8
                else:
                    factor = 0.3
            else:
                if 8 <= hour <= 18:  # 工作时间
                    factor = 1.2
                else:
                    factor = 0.4
        else:  # 工业
            # 工业负荷：比较平稳，晚上略低
            if 7 <= hour <= 19:  # 白班
                factor = 0.9 + 0.1 * np.sin(np.pi * (hour - 7) / 12)
            else:
                factor = 0.7
        
        # 随机波动
        random_factor = 0.9 + 0.2 * np.random.random()
        
        # 计算最终负荷
        load = base_load * factor * random_factor
        return load
    
    def generate_daily_data(self, date: datetime.datetime.date, user_config: Dict[str, Any]) -> pd.DataFrame:
        """
        生成一天的发电和用电数据
        
        Args:
            date: 日期
            user_config: 用户配置信息，包括光伏容量、用户类型等
            
        Returns:
            包含一天数据的DataFrame
        """
        panel_capacity = user_config.get("panel_capacity", 5.0)
        user_type = user_config.get("user_type", "住宅")
        base_load = user_config.get("base_load", 2.0)
        
        data = []
        for hour in range(24):
            for minute in range(0, 60, 15):  # 15分钟一个数据点
                dt = datetime.datetime.combine(date, datetime.time(hour, minute))
                
                # 生成数据点
                irradiance = self.generate_solar_irradiance(dt)
                temperature = self.generate_temperature(dt)
                solar_power = self.generate_solar_power_output(dt, panel_capacity)
                load = self.generate_load_profile(dt, user_type, base_load)
                
                data.append({
                    "timestamp": dt,
                    "irradiance": irradiance,
                    "temperature": temperature,
                    "solar_power": solar_power,
                    "load": load,
                    "net_load": load - solar_power  # 净负荷（负值表示供电大于用电）
                })
                
        return pd.DataFrame(data)
    
    def generate_historical_data(self, start_date: datetime.datetime.date, days: int, 
                              user_config: Dict[str, Any]) -> pd.DataFrame:
        """
        生成历史数据
        
        Args:
            start_date: 起始日期
            days: 天数
            user_config: 用户配置
            
        Returns:
            历史数据DataFrame
        """
        all_data = []
        
        for day in range(days):
            date = start_date + datetime.timedelta(days=day)
            daily_data = self.generate_daily_data(date, user_config)
            all_data.append(daily_data)
            
        return pd.concat(all_data, ignore_index=True)
    
    
class EnergyDataCollector:
    """能源数据采集器，模拟从实际设备采集数据并处理"""
    
    def __init__(self, data_generator: EnergyDataGenerator = None):
        """
        初始化数据采集器
        
        Args:
            data_generator: 能源数据生成器，用于模拟真实数据
        """
        self.data_generator = data_generator or EnergyDataGenerator()
        self.cached_data: Dict[str, pd.DataFrame] = {}  # 缓存的用户数据
        self.user_configs: Dict[str, Dict[str, Any]] = {}  # 用户配置信息，用作用户画像
        
    def collect_current_data(self, user_address: str, user_config: Dict[str, Any]) -> Dict[str, float]:
        """
        采集当前能源数据
        
        Args:
            user_address: 用户地址
            user_config: 用户配置
            
        Returns:
            当前能源数据
        """
        # 使用实际时间
        now = datetime.datetime.now()
        
        # 使用数据生成器模拟数据采集
        irradiance = self.data_generator.generate_solar_irradiance(now)
        temperature = self.data_generator.generate_temperature(now)
        solar_power = self.data_generator.generate_solar_power_output(
            now, 
            user_config.get("panel_capacity", 5.0)
        )
        load = self.data_generator.generate_load_profile(
            now,
            user_config.get("user_type", "住宅"),
            user_config.get("base_load", 2.0)
        )
        
        return {
            "timestamp": now.timestamp(),
            "irradiance": irradiance,
            "temperature": temperature,
            "solar_power": solar_power,
            "load": load,
            "net_load": load - solar_power
        }
        
    def collect_and_store_data(self, user_address: str, user_config: Dict[str, Any], 
                            storing_interval: int = 15) -> None:
        """
        连续采集并存储数据
        
        Args:
            user_address: 用户地址
            user_config: 用户配置
            storing_interval: 存储间隔（分钟）
        """
        print(f"开始为用户 {user_address} 采集数据...")
        
        try:
            while True:
                data_point = self.collect_current_data(user_address, user_config)
                
                # 如果是第一次采集，创建新的DataFrame
                if user_address not in self.cached_data:
                    self.cached_data[user_address] = pd.DataFrame([data_point])
                else:
                    # 追加数据
                    self.cached_data[user_address] = self.cached_data[user_address].append(
                        data_point, ignore_index=True
                    )
                
                print(f"采集到数据: 发电量={data_point['solar_power']:.2f}kW, 用电量={data_point['load']:.2f}kW")
                
                # 按指定间隔休眠
                time.sleep(storing_interval * 60)
                
        except KeyboardInterrupt:
            print("数据采集已停止")
            
    def get_user_profile(self, user_address: str) -> Dict[str, Any]:
        """
        获取用户画像数据，用于负荷预测模型
        
        Args:
            user_address: 用户地址
            
        Returns:
            用户画像字典，包含用户类型、面板容量、基础负荷等信息
        """
        # 检查是否有保存的用户配置
        if user_address in self.user_configs:
            return self.user_configs[user_address]
            
        # 如果没有保存的配置，从历史数据中提取用户类型
        if user_address in self.cached_data and not self.cached_data[user_address].empty:
            # 分析历史数据，提取用户画像特征
            data = self.cached_data[user_address]
            
            # 估计基础负荷（夜间最小负荷的平均值）
            night_load = data.loc[pd.to_datetime(data['timestamp']).dt.hour.between(1, 5), 'load']
            base_load = night_load.mean() if not night_load.empty else 2.0
            
            # 判断用户类型
            # 根据一周内工作日和周末的负荷差异来判断
            try:
                data['datetime'] = pd.to_datetime(data['timestamp'])
                data['day_of_week'] = data['datetime'].dt.dayofweek
                weekday_load = data.loc[data['day_of_week'] < 5, 'load'].mean()
                weekend_load = data.loc[data['day_of_week'] >= 5, 'load'].mean()
                
                load_ratio = weekend_load / weekday_load if weekday_load > 0 else 1
                
                if load_ratio > 1.2:
                    user_type = "住宅"  # 周末负荷高于工作日，可能是住宅
                elif load_ratio < 0.8:
                    user_type = "商业"  # 工作日负荷高于周末，可能是商业
                else:
                    user_type = "工业"  # 负荷比较稳定，可能是工业
            except:
                user_type = "住宅"  # 默认为住宅类型
            
            # 估计光伏板容量
            max_solar = data['solar_power'].max()
            panel_capacity = max(5.0, max_solar)
            
            # 创建并保存用户配置
            user_config = {
                "panel_capacity": panel_capacity,
                "user_type": user_type,
                "base_load": base_load
            }
            self.user_configs[user_address] = user_config
            
            return user_config
        else:
            # 如果没有数据，返回默认用户画像
            default_profile = {
                "panel_capacity": 5.0,
                "user_type": "住宅",
                "base_load": 2.0
            }
            self.user_configs[user_address] = default_profile
            return default_profile
    
    def get_user_data(self, user_address: str) -> pd.DataFrame:
        """
        获取用户的能源数据
        
        Args:
            user_address: 用户地址
            
        Returns:
            用户能源数据DataFrame
        """
        if user_address in self.cached_data:
            return self.cached_data[user_address].copy()
        else:
            return pd.DataFrame()
    
    def generate_sample_data(self, user_address: str, user_config: Dict[str, Any], 
                          days_back: int = 30) -> pd.DataFrame:
        """
        生成样本历史数据
        
        Args:
            user_address: 用户地址
            user_config: 用户配置
            days_back: 回溯的天数
            
        Returns:
            历史数据DataFrame
        """
        end_date = datetime.datetime.now().date()
        start_date = end_date - datetime.timedelta(days=days_back)
        
        historical_data = self.data_generator.generate_historical_data(
            start_date, days_back, user_config
        )
        
        # 缓存数据
        self.cached_data[user_address] = historical_data
        
        return historical_data
    
    def export_user_data(self, user_address: str, file_path: str) -> None:
        """
        导出用户数据到文件
        
        Args:
            user_address: 用户地址
            file_path: 文件路径
        """
        if user_address not in self.cached_data:
            raise ValueError(f"没有用户 {user_address} 的数据")
            
        self.cached_data[user_address].to_csv(file_path, index=False)
        print(f"数据已导出到 {file_path}")