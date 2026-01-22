"""
Configuration Expert - Singleton configuration manager

Handles loading and caching of YAML configuration files.
Provides easy access to configuration values throughout the application.
"""

import yaml
from pathlib import Path
from typing import Any, Optional, Dict

class ConfigExpert:
    """
    Singleton configuration manager
    
    Loads YAML configuration once and provides easy access via get() method.
    
    Usage:
        # Initialize with config file
        config = ConfigExpert.get_instance("experiments/example.yaml")
        
        # Access values anywhere in code
        test_mode = ConfigExpert.get_instance().get("test_mode")
        models = ConfigExpert.get_instance().get("models")
        
        # Nested values with dot notation
        name = ConfigExpert.get_instance().get("experiment.name")
        
        # With default values
        timeout = ConfigExpert.get_instance().get("timeout", default=30)
    """
    
    _instance = None
    _config = None
    _config_path = None
    
    def __new__(cls, config_path: Optional[str] = None):
        """Singleton pattern: only one instance allowed"""
        if cls._instance is None:
            cls._instance = super(ConfigExpert, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration (only once)
        
        Args:
            config_path: Path to YAML configuration file
        """
        # If config already loaded and no new path provided, skip
        if self._config is not None and config_path is None:
            return
        
        # Load new config if path provided
        if config_path is not None:
            self._load_config(config_path)
    
    def _load_config(self, config_path: str):
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to YAML file
        """
        config_path = str(Path(config_path).resolve())
        
        # Skip if already loaded from this path
        if self._config_path == config_path and self._config is not None:
            print(f"✅️ Config already loaded from: {config_path}")
            return
        
        print(f"📖 Loading configuration from: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            
            self._config_path = config_path
            print(f"✅️ Configuration loaded successfully")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise Exception(f"Error loading configuration: {e}")
    
    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> 'ConfigExpert':
        """
        Get singleton instance
        
        Args:
            config_path: Optional path to config file (only needed on first call)
            
        Returns:
            ConfigExpert instance
        """
        if cls._instance is None:
            if config_path is None:
                raise ValueError(
                    "ConfigExpert must be initialized with config_path on first call. "
                    "Usage: ConfigExpert.get_instance('path/to/config.yaml')"
                )
            cls._instance = cls(config_path)
        elif config_path is not None:
            # Update config if new path provided
            cls._instance._load_config(config_path)
        
        return cls._instance
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key
        
        Supports dot notation for nested values.
        
        Args:
            key: Configuration key (supports dot notation, e.g., "experiment.name")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Examples:
            config.get("models")
            config.get("experiment.name")
            config.get("timeout", default=30)
        """
        if self._config is None:
            raise RuntimeError(
                "Configuration not loaded. "
                "Initialize with: ConfigExpert.get_instance('path/to/config.yaml')"
            )
        
        # Handle dot notation for nested keys
        if '.' in key:
            keys = key.split('.')
            value = self._config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
        
        # Simple key lookup
        return self._config.get(key, default)
    
    def get_all(self) -> Dict:
        """
        Get entire configuration dictionary
        
        Returns:
            Complete configuration as dictionary
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        
        return self._config.copy()
    
    def has(self, key: str) -> bool:
        """
        Check if configuration key exists
        
        Args:
            key: Configuration key (supports dot notation)
            
        Returns:
            True if key exists, False otherwise
        """
        if self._config is None:
            return False
        
        # Handle dot notation
        if '.' in key:
            keys = key.split('.')
            value = self._config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return False
            
            return True
        
        return key in self._config
    
    def reload(self, config_path: Optional[str] = None):
        """
        Reload configuration from file
        
        Args:
            config_path: Optional new config path, or reload from current path
        """
        path = config_path or self._config_path
        
        if path is None:
            raise ValueError("No config path available for reload")
        
        # Force reload by clearing config first
        self._config = None
        self._load_config(path)
    
    def get_config_path(self) -> Optional[str]:
        """Get path of currently loaded configuration file"""
        return self._config_path
    
    @classmethod
    def reset(cls):
        """Reset singleton (useful for testing)"""
        cls._instance = None
        cls._config = None
        cls._config_path = None