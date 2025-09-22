"""
配置管理器
"""
import yaml
import os
from typing import Any, Dict

class ConfigManager:
    """
    配置管理器类，用于加载和管理项目配置
    """
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        
        Args:
            config_path (str): 配置文件路径
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        with open(self.config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项的值
        
        Args:
            key (str): 配置项键名
            default (Any): 默认值
            
        Returns:
            Any: 配置项的值
        """
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置项的值
        
        Args:
            key (str): 配置项键名
            value (Any): 配置项的值
        """
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value

# 全局配置管理器实例
config_manager = ConfigManager()