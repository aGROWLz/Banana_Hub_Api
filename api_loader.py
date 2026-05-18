import json
import os
from typing import Dict, List, Any


class APIProvider:
    """API 提供商配置类"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.name = self.config.get("name", "Unknown")
        self.description = self.config.get("description", "")
        self.endpoints = self.config.get("endpoints", {})
        self.hosts = self.config.get("hosts", {})
        self.default_host = self.config.get("default_host", "china")
        self.request_format = self.config.get("request_format", {})
        self.response_format = self.config.get("response_format", {})
        self.models = self.config.get("models", [])
        self.model_mapping = self.config.get("model_mapping", {})  # 模型映射
        self.image_sizes = self.config.get("image_sizes", [])
        self.aspect_ratios = self.config.get("aspect_ratios", [])
    
    def get_host(self, host_type: str = None) -> str:
        """获取 API 主机地址"""
        if host_type is None:
            host_type = self.default_host
        return self.hosts.get(host_type, list(self.hosts.values())[0])
    
    def get_endpoint(self, endpoint_name: str) -> str:
        """获取端点路径"""
        return self.endpoints.get(endpoint_name, "")
    
    def map_model(self, model: str) -> str:
        """映射模型名称，如果有映射则返回映射后的名称，否则返回原名称"""
        return self.model_mapping.get(model, model)
    
    def build_request(self, endpoint_name: str, **kwargs) -> Dict[str, Any]:
        """构建请求"""
        format_config = self.request_format.get(endpoint_name, {})
        content_type = format_config.get("content_type", "application/json")
        
        # 构建 headers
        headers = {}
        for key, value in format_config.get("headers", {}).items():
            headers[key] = self._replace_placeholders(value, kwargs)
        
        # 只有在非 multipart/form-data 时才添加 Content-Type
        if content_type != "multipart/form-data":
            headers["Content-Type"] = content_type
        
        # 构建 body（支持嵌套对象）
        body_template = format_config.get("body", {})
        body = self._build_body_recursive(body_template, kwargs)
        
        return {
            "method": format_config.get("method", "POST"),
            "headers": headers,
            "body": body,
            "content_type": content_type
        }
    
    def _build_body_recursive(self, template: Any, values: Dict) -> Any:
        """递归构建请求体，支持嵌套对象"""
        if isinstance(template, dict):
            result = {}
            for key, value in template.items():
                built_value = self._build_body_recursive(value, values)
                # 只添加非 None 的值
                if built_value is not None:
                    result[key] = built_value
            return result
        elif isinstance(template, list):
            return [self._build_body_recursive(item, values) for item in template]
        elif isinstance(template, str) and template.startswith("{") and template.endswith("}"):
            # 占位符
            placeholder_key = template[1:-1]
            if placeholder_key in values:
                param_value = values[placeholder_key]
                # 只返回非 None、非空值
                if param_value is None:
                    return None
                if param_value or param_value == 0 or param_value is False:
                    return param_value
            return None
        else:
            # 普通值
            return template
    
    def parse_response(self, endpoint_name: str, response_data: Dict) -> Dict[str, Any]:
        """解析响应"""
        format_config = self.response_format.get(endpoint_name, {})
        result = {}
        
        for key, path in format_config.items():
            if isinstance(path, str):
                result[key] = self._get_nested_value(response_data, path)
            else:
                result[key] = path
        
        return result
    
    def _replace_placeholders(self, template: str, values: Dict) -> Any:
        """替换占位符"""
        if not isinstance(template, str):
            return template
        
        # 检查是否是占位符格式 {key}
        if template.startswith("{") and template.endswith("}"):
            key = template[1:-1]
            return values.get(key, template)
        
        # 替换字符串中的所有占位符
        result = template
        for key, value in values.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        
        return result
    
    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """获取嵌套字典的值，支持路径如 'data.results[0].url'"""
        keys = path.split('.')
        value = data
        
        for key in keys:
            if '[' in key and ']' in key:
                # 处理数组索引，如 results[0]
                key_name = key[:key.index('[')]
                index = int(key[key.index('[') + 1:key.index(']')])
                value = value.get(key_name, [])[index] if isinstance(value.get(key_name), list) else None
            else:
                value = value.get(key) if isinstance(value, dict) else None
            
            if value is None:
                return None
        
        return value


class APILoader:
    """API 配置加载器"""
    
    def __init__(self, api_dir: str):
        self.api_dir = api_dir
        self.providers: Dict[str, APIProvider] = {}
        self._load_providers()
    
    def _load_providers(self):
        """加载所有 API 提供商配置"""
        if not os.path.exists(self.api_dir):
            return
        
        for filename in os.listdir(self.api_dir):
            if filename.endswith('.json'):
                config_path = os.path.join(self.api_dir, filename)
                try:
                    provider = APIProvider(config_path)
                    provider_id = filename.replace('.json', '')
                    self.providers[provider_id] = provider
                except Exception as e:
                    print(f"加载 API 配置失败 {filename}: {e}")
    
    def get_provider(self, provider_id: str) -> APIProvider:
        """获取指定的 API 提供商"""
        return self.providers.get(provider_id)
    
    def get_provider_names(self) -> List[str]:
        """获取所有提供商名称"""
        return [provider.name for provider in self.providers.values()]
    
    def get_provider_ids(self) -> List[str]:
        """获取所有提供商 ID"""
        return list(self.providers.keys())
