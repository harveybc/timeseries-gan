# filepath: /home/harveybc/Documents/GitHub/timeseries-gan/tsg_plugins/plugin_base.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class PluginBase(ABC):
    """
    Base class for all plugins.
    Defines the common interface for plugins.
    """
    plugin_params: Dict[str, Any] = {}

    def __init__(self, config: Dict[str, Any]):
        self.params = self.plugin_params.copy()
        if config:
            self.main_config = config.copy() # Store main_config
            # Update params with prefixed and non-prefixed keys from config
            for key, value in config.items():
                if key in self.plugin_params:
                    self.params[key] = value
                else:
                    # Attempt to match prefixed keys (e.g., generator_param_name)
                    plugin_name_prefix = getattr(self, 'plugin_name_prefix', None) # e.g., 'generator_'
                    if plugin_name_prefix and key.startswith(plugin_name_prefix):
                        param_key = key[len(plugin_name_prefix):]
                        if param_key in self.plugin_params:
                            self.params[param_key] = value
        else:
            self.main_config = {}


    @abstractmethod
    def set_params(self, **kwargs) -> None:
        """
        Set plugin parameters.
        This method should update self.params and potentially re-initialize
        or rebuild components if critical parameters change.
        """
        # Default implementation: update self.params
        # Plugins should override if more complex logic is needed (e.g., model rebuilding)
        
        # Update main_config first if it exists
        if hasattr(self, 'main_config') and self.main_config is not None:
            self.main_config.update(kwargs)
        else:
            self.main_config = kwargs.copy()

        # Update plugin-specific params, handling prefixes
        plugin_name_prefix = getattr(self, 'plugin_name_prefix', None)

        for param_key_default in self.plugin_params.keys():
            # Check for prefixed key first
            if plugin_name_prefix and f"{plugin_name_prefix}{param_key_default}" in kwargs:
                self.params[param_key_default] = kwargs[f"{plugin_name_prefix}{param_key_default}"]
            # Then check for non-prefixed key
            elif param_key_default in kwargs:
                self.params[param_key_default] = kwargs[param_key_default]
        
        # Optionally, log or handle unexpected parameters
        # for key_arg, val_arg in kwargs.items():
        #     if not (key_arg in self.plugin_params or (plugin_name_prefix and key_arg.startswith(plugin_name_prefix))):
        #         print(f"Warning: Unknown parameter '{key_arg}' passed to {self.__class__.__name__}.set_params")


    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information from the plugin.
        """
        # Basic implementation, plugins can extend this
        debug_info = {}
        plugin_debug_vars = getattr(self, 'plugin_debug_vars', [])
        for var in plugin_debug_vars:
            if var in self.params:
                prefix = getattr(self, 'plugin_name_prefix', self.__class__.__name__.lower() + '_')
                debug_info[f"{prefix}{var}"] = self.params[var]
        return debug_info

    def add_debug_info(self, debug_dict: Dict[str, Any]) -> None:
        """
        Add plugin-specific debug information to a dictionary.
        """
        debug_dict.update(self.get_debug_info())

    @property
    def name(self) -> str:
        """Return the name of the plugin (class name by default)."""
        return self.__class__.__name__

    # Example of a method that might be common to many plugins
    # def load_model(self, model_path: str) -> Any:
    #     """Load a model from the given path."""
    #     raise NotImplementedError("Plugins that use models must implement load_model.")

    # def save_model(self, model: Any, save_path: str) -> None:
    #     """Save a model to the given path."""
    #     raise NotImplementedError("Plugins that use models must implement save_model.")

