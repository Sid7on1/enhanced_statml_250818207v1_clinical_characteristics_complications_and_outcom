import logging
import os
import yaml
from typing import Dict, List, Optional
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
CONFIG_FILE = 'config.yaml'
DEFAULT_CONFIG = {
    'agent': {
        'name': 'default_agent',
        'version': '1.0'
    },
    'environment': {
        'name': 'default_environment',
        'version': '1.0'
    }
}

class ConfigError(Exception):
    """Base class for configuration errors"""
    pass

class ConfigMissingError(ConfigError):
    """Raised when a required configuration is missing"""
    pass

class ConfigInvalidError(ConfigError):
    """Raised when a configuration is invalid"""
    pass

class ConfigLoadError(ConfigError):
    """Raised when a configuration cannot be loaded"""
    pass

class ConfigSaveError(ConfigError):
    """Raised when a configuration cannot be saved"""
    pass

class ConfigType(Enum):
    """Enum for configuration types"""
    AGENT = 'agent'
    ENVIRONMENT = 'environment'

class Config(ABC):
    """Abstract base class for configuration"""
    def __init__(self, config_type: ConfigType):
        self.config_type = config_type
        self.config = self.load_config()

    @abstractmethod
    def load_config(self) -> Dict:
        """Load configuration from file or default values"""
        pass

    def save_config(self) -> None:
        """Save configuration to file"""
        try:
            with open(CONFIG_FILE, 'w') as f:
                yaml.dump(self.config, f)
        except Exception as e:
            raise ConfigSaveError(f"Failed to save configuration: {e}")

    def get_config(self) -> Dict:
        """Get the current configuration"""
        return self.config

class AgentConfig(Config):
    """Configuration for the agent"""
    def load_config(self) -> Dict:
        config = DEFAULT_CONFIG['agent']
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config is not None:
                    config.update(loaded_config['agent'])
        except FileNotFoundError:
            logger.info(f"Configuration file not found, using default values")
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"Failed to load configuration: {e}")
        return config

class EnvironmentConfig(Config):
    """Configuration for the environment"""
    def load_config(self) -> Dict:
        config = DEFAULT_CONFIG['environment']
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config is not None:
                    config.update(loaded_config['environment'])
        except FileNotFoundError:
            logger.info(f"Configuration file not found, using default values")
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"Failed to load configuration: {e}")
        return config

class ConfigManager:
    """Manager for configuration"""
    def __init__(self):
        self.agent_config = AgentConfig(ConfigType.AGENT)
        self.environment_config = EnvironmentConfig(ConfigType.ENVIRONMENT)

    def get_agent_config(self) -> Dict:
        """Get the agent configuration"""
        return self.agent_config.get_config()

    def get_environment_config(self) -> Dict:
        """Get the environment configuration"""
        return self.environment_config.get_config()

    def save_config(self) -> None:
        """Save the configuration to file"""
        self.agent_config.save_config()
        self.environment_config.save_config()

@contextmanager
def config_context(config_manager: ConfigManager):
    """Context manager for configuration"""
    try:
        yield config_manager
    finally:
        config_manager.save_config()

def main():
    config_manager = ConfigManager()
    with config_context(config_manager) as cm:
        agent_config = cm.get_agent_config()
        environment_config = cm.get_environment_config()
        logger.info(f"Agent configuration: {agent_config}")
        logger.info(f"Environment configuration: {environment_config}")

if __name__ == '__main__':
    main()