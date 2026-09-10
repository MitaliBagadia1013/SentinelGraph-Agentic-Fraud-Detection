import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigDict(dict):

    def __init__(self, data: Dict[str, Any]):
        super().__init__(data)
        for key, value in data.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)
            elif isinstance(value, list):
                self[key] = [
                    ConfigDict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                self[key] = value

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Configuration key '{key}' not found")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_nested(self, path: str, default: Any = None) -> Any:
        keys = path.split(".")
        value = self
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


class Config:
    _instance: Optional["Config"] = None
    _config: Optional[ConfigDict] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if self._config is not None:
            return
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        self.config_path = Path(config_path)
        self._load_config()
        self._override_from_env()
        self._validate_config()

    def _load_config(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\nPlease create config.yaml in the project root."
            )
        try:
            with open(self.config_path, "r") as f:
                raw_config = yaml.safe_load(f)
            self._config = ConfigDict(raw_config)
            logger.info(f"Configuration loaded from {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

    def _override_from_env(self) -> None:
        if "llm" in self._config and "api_key_env" in self._config.llm:
            env_var_name = self._config.llm.api_key_env
            api_key = os.getenv(env_var_name)
            if api_key:
                self._config.llm["api_key"] = api_key
                logger.info(f"Loaded {env_var_name} from environment")
            else:
                logger.warning(
                    f"{env_var_name} not found in environment. Agent functionality may be limited."
                )
        sagemaker_endpoint = os.getenv("SAGEMAKER_ENDPOINT_NAME")
        if sagemaker_endpoint:
            self._config.sagemaker.endpoint["name"] = sagemaker_endpoint
            logger.info(f"Using SageMaker endpoint: {sagemaker_endpoint}")
        feature_store_bucket = os.getenv("FEATURE_STORE_BUCKET")
        if feature_store_bucket:
            self._config.feature_store.offline_store["s3_bucket"] = feature_store_bucket

    def _validate_config(self) -> None:
        required_sections = ["model", "thresholds", "agents", "feature_store"]
        for section in required_sections:
            if section not in self._config:
                raise ValueError(
                    f"Missing required configuration section: '{section}'\nPlease check config.yaml"
                )
        thresholds = self._config.thresholds
        for threshold_name in ["high_risk", "medium_risk", "low_risk"]:
            value = thresholds.get(threshold_name)
            if value is None or not 0 <= value <= 1:
                raise ValueError(
                    f"Threshold '{threshold_name}' must be between 0 and 1, got: {value}"
                )
        logger.info("Configuration validation passed")

    def reload(self) -> None:
        logger.info("Reloading configuration...")
        self._config = None
        self._load_config()
        self._override_from_env()
        self._validate_config()
        logger.info("Configuration reloaded successfully")

    @property
    def config(self) -> ConfigDict:
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        return self._config

    def get(self, path: str, default: Any = None) -> Any:
        return self.config.get_nested(path, default)

    def __repr__(self) -> str:
        return f"<Config loaded from {self.config_path}>"


_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> ConfigDict:
    global _global_config
    if _global_config is None:
        _global_config = Config(config_path)
    return _global_config.config


def reload_config() -> None:
    global _global_config
    if _global_config is not None:
        _global_config.reload()
    else:
        _global_config = Config()


def get_model_config() -> ConfigDict:
    return get_config().model


def get_agent_config() -> ConfigDict:
    return get_config().agents


def get_threshold_config() -> ConfigDict:
    return get_config().thresholds


def get_sagemaker_config() -> ConfigDict:
    return get_config().sagemaker


def get_feature_store_config() -> ConfigDict:
    return get_config().feature_store


def get_hitl_config() -> ConfigDict:
    return get_config().hitl


if __name__ == "__main__":
    "\n Test the configuration loader.\n \n Run this file directly to verify config.yaml loads correctly:\n python config_loader.py\n"
    import json

    logging.basicConfig(level=logging.INFO)
    try:
        config = get_config()
        print("\n" + "=" * 70)
        print("CONFIGURATION LOADED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nModel: {config.model.name} v{config.model.version}")
        print(f"- Trees: {config.model.params.n_estimators}")
        print(f"- Max Depth: {config.model.params.max_depth}")
        print(f"- Learning Rate: {config.model.params.learning_rate}")
        print(f"\nThresholds:")
        print(f"- High Risk: {config.thresholds.high_risk}")
        print(f"- Medium Risk: {config.thresholds.medium_risk}")
        print(f"- Low Risk: {config.thresholds.low_risk}")
        print(f"- HITL Uncertainty: {config.thresholds.hitl_uncertainty}")
        print(f"\nAgents:")
        print(f"- Enabled: {config.agents.enabled}")
        print(f"- Max Iterations: {config.agents.max_iterations}")
        print(f"- Detective: {config.agents.detective.name}")
        print(f"- Analyst: {config.agents.analyst.name}")
        print(f"- Verifier: {config.agents.verifier.name}")
        print(f"\nSageMaker:")
        print(f"- Endpoint: {config.sagemaker.endpoint.name}")
        print(f"- Instance Type: {config.sagemaker.endpoint.instance_type}")
        print(f"- Auto-scaling: {config.sagemaker.endpoint.auto_scaling.enabled}")
        print(f"\nFeature Store:")
        print(f"- Online Store: {config.feature_store.online_store.enabled}")
        print(f"- Cache: {config.feature_store.cache.enabled}")
        print(f"- Cache TTL: {config.feature_store.cache.ttl_seconds}s")
        print(f"\nHuman-in-the-Loop:")
        print(f"- Enabled: {config.hitl.enabled}")
        print(f"- Sample Rate: {config.hitl.sample_rate * 100}%")
        print(f"- UI Port: {config.hitl.ui.port}")
        print("\n" + "=" * 70)
        print("All configuration tests passed!")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"\nConfiguration Error: {e}\n")
        raise
