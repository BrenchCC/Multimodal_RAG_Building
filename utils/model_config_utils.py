import os
import sys
import yaml
import logging


sys.path.append(os.getcwd())
logger = logging.getLogger("ModelConfigs")

@staticmethod
class ModelConfigs:

    def __init__(
        self,
        config_path: str = "configs/api.yaml"
    ):
        self.config_path = config_path
        self._config = self._load_model_config()
        self._get_openai_model_config()
        self._get_dashscope_model_config()
        self._get_volces_model_config()
        self._get_silicon_model_config()
        self._get_deepseek_model_config()
        self._get_glm_model_config()
        self._get_moonshot_model_config()

    def _load_model_config(self):
        if not os.path.exists(os.path.abspath(self.config_path)):
            raise FileNotFoundError(f"Model config file not found: {self.config_path}")
        if not self.config_path.endswith('.yaml'):
            raise ValueError(f"Model config file must be a YAML file: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded model config from {self.config_path}")
        return config

    def _get_openai_model_config(self):
        openai_model_config = self._config.get('OpenAI', {})
        openai_base_url = openai_model_config.get('base_url', '')
        openai_api_key = openai_model_config.get('api_key', '')

        self.OPENAI_BASE_URL = openai_base_url
        self.OPENAI_API_KEY = openai_api_key
        return openai_base_url, openai_api_key

    def _get_dashscope_model_config(self):
        dashscope_model_config = self._config.get('DashScope', {})
        dashscope_base_url = dashscope_model_config.get('base_url', '')
        dashscope_api_key = dashscope_model_config.get('api_key', '')

        self.DASHSCOPE_BASE_URL = dashscope_base_url
        self.DASHSCOPE_API_KEY = dashscope_api_key
        return dashscope_base_url, dashscope_api_key
    
    def _get_volces_model_config(self):
        volces_model_config = self._config.get('Volces', {})
        volces_base_url = volces_model_config.get('base_url', '')
        volces_api_key = volces_model_config.get('api_key', '')

        self.VOLCES_BASE_URL = volces_base_url
        self.VOLCES_API_KEY = volces_api_key
        return volces_base_url, volces_api_key
    
    def _get_silicon_model_config(self):
        silicon_model_config = self._config.get('Silicon', {})
        silicon_base_url = silicon_model_config.get('base_url', '')
        silicon_api_key = silicon_model_config.get('api_key', '')

        self.SILICON_BASE_URL = silicon_base_url
        self.SILICON_API_KEY = silicon_api_key
        return silicon_base_url, silicon_api_key
    
    def _get_deepseek_model_config(self):
        deepseek_model_config = self._config.get('DeepSeek', {})
        deepseek_base_url = deepseek_model_config.get('base_url', '')
        deepseek_api_key = deepseek_model_config.get('api_key', '')

        self.DEEPSEEK_BASE_URL = deepseek_base_url
        self.DEEPSEEK_API_KEY = deepseek_api_key
        return deepseek_base_url, deepseek_api_key
    
    def _get_glm_model_config(self):
        glm_model_config = self._config.get('GLM', {})
        glm_base_url = glm_model_config.get('base_url', '')
        glm_api_key = glm_model_config.get('api_key', '')

        self.GLM_BASE_URL = glm_base_url
        self.GLM_API_KEY = glm_api_key
        return glm_base_url, glm_api_key
    
    def _get_moonshot_model_config(self):
        moonshot_model_config = self._config.get('MoonShot', {})
        moonshot_base_url = moonshot_model_config.get('base_url', '')
        moonshot_api_key = moonshot_model_config.get('api_key', '')

        self.MOONSHOT_BASE_URL = moonshot_base_url
        self.MOONSHOT_API_KEY = moonshot_api_key
        return moonshot_base_url, moonshot_api_key

    def _get_local_model_config(self):
        local_model_config = self._config.get('Local', {})
        local_base_url = local_model_config.get('base_url', '')
        local_api_key = local_model_config.get('api_key', '')

        self.LOCAL_BASE_URL = local_base_url
        self.LOCAL_API_KEY = local_api_key
        return local_base_url, local_api_key
    
MODEL_CONFIGS = ModelConfigs()

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [
            logging.StreamHandler()
        ]
    )
    model_configs = ModelConfigs()
