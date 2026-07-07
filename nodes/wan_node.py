import requests
import torch
import numpy as np
from PIL import Image
import io
import base64
import json
import os
from datetime import datetime
from comfy_api.latest import io as comfy_io
import folder_paths
from ..utils import APILoader


class WanImageGenerationNode(comfy_io.ComfyNode):
    """
    Wan Image Generation API 节点
    支持 Vector Engine wan2.7-image-pro 图像生成
    """

    # 初始化 API 加载器
    api_loader = None

    @classmethod
    def _init_api_loader(cls):
        """初始化 API 加载器"""
        if cls.api_loader is None:
            api_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api")
            cls.api_loader = APILoader(api_dir)

    @classmethod
    def _load_config(cls):
        """加载用户配置文件"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
        default_config = {
            "api_keys": {}
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config
            except Exception as e:
                print(f"加载配置文件失败: {e}，使用默认配置")
                return default_config
        return default_config

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        cls._init_api_loader()
        config = cls._load_config()
        
        # 获取 wan_api 提供商
        provider = cls.api_loader.get_provider("wan_api")
        if not provider:
            # 默认配置
            image_sizes = ["1024x1024", "1024x1792", "1792x1024", "512x512"]
        else:
            image_sizes = provider.image_sizes
        
        return comfy_io.Schema(
            node_id="WanImageGeneration",
            display_name="Wan Image Generation API",
            category="Banana",
            inputs=[
                comfy_io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                ),
                comfy_io.Combo.Input(
                    "image_size",
                    options=image_sizes,
                    default="1024x1024" if "1024x1024" in image_sizes else image_sizes[0]
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xffffffffffffffff,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Int.Input(
                    "timeout",
                    default=300,
                    min=10,
                    max=600,
                    step=10,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
            ],
            outputs=[
                comfy_io.Image.Output("result_image"),
                comfy_io.String.Output("log"),
            ],
        )

    @classmethod
    def execute(cls, prompt, image_size, seed, timeout) -> comfy_io.NodeOutput:
        
        cls._init_api_loader()
        log_messages = []
        
        def log(msg, icon="", console_only=False):
            """添加日志并立即打印到控制台"""
            full_msg = f"{icon} {msg}" if icon else msg
            if not console_only:
                log_messages.append(full_msg)
            print(f"[Wan Image API] {full_msg}")
        
        try:
            # 加载配置和 API 提供商
            config = cls._load_config()
            provider = cls.api_loader.get_provider("wan_api")
            
            if not provider:
                error_msg = "未找到 Wan Image API 提供商配置"
                log(error_msg, "❌")
                raise ValueError(error_msg)
            
            log(f"使用 API 提供商: {provider.name}", "🔌")
            
            # 确定使用的 API host
            api_host = provider.get_host("china")
            
            # 从配置文件获取 API key
            api_keys = config.get("api_keys", {})
            api_key = api_keys.get("wan_api", "")
            
            if not api_key:
                error_msg = "错误: 未设置 API Key，请在配置文件的 api_keys.wan_api 中设置"
                log(error_msg, "❌")
                raise ValueError(error_msg)
            
            api_host = api_host.rstrip('/')
            
            log(f"使用 API Host: {api_host}", "🌐")
            log(f"使用模型: wan2.7-image-pro", "🎨")
            log(f"图片尺寸: {image_size}", "📐")
            log(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}", "✍️")
            
            # 构建请求
            draw_endpoint = provider.get_endpoint("draw")
            draw_url = f"{api_host}{draw_endpoint}"
            
            request_body = {
                "model": "wan2.7-image-pro",
                "prompt": prompt,
                "size": image_size,
                "n": 1,
                "watermark": False,
                "prompt_extend": False
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            log(f"发送图像生成请求到: {draw_url}", "📤")
            
            # 发送请求
            try:
                response = requests.post(
                    draw_url,
                    headers=headers,
                    json=request_body,
                    timeout=timeout
                )
            except requests.exceptions.Timeout:
                error_msg = f"请求超时 ({timeout}秒)，请检查网络连接或增加 timeout 参数"
                log(error_msg, "⏰")
                raise TimeoutError(error_msg)
            except requests.exceptions.RequestException as e:
                error_msg = f"请求失败: {str(e)}"
                log(error_msg, "❌")
                raise RuntimeError(error_msg)
            
            log(f"收到响应，状态码: {response.status_code}", "📨", console_only=True)
            
            if response.status_code != 200:
                error_msg = f"API 请求失败: {response.status_code} - {response.text}"
                log(error_msg, "❌")
                raise RuntimeError(error_msg)
            
            # 尝试解析 JSON
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                error_msg = f"API 返回的不是有效的 JSON 格式"
                log(error_msg, "❌")
                log(f"响应内容: {response.text[:500]}", "ℹ️")
                raise RuntimeError(f"{error_msg}\n响应内容: {response.text[:200]}")
            
            log(f"请求响应: {json.dumps(result, ensure_ascii=False)[:500]}...", "📥")
            
            # 处理响应
            image_data = None
            image_url = None
            
            # 尝试从 data[0].url 获取
            if "data" in result and len(result["data"]) > 0:
                first_item = result["data"][0]
                if "url" in first_item:
                    image_url = first_item["url"]
                elif "b64_json" in first_item:
                    image_data = first_item["b64_json"]
            
            if image_url:
                log(f"获取到图像 URL: {image_url[:50]}...", "🎨")
                log("正在下载图像...", "⬇️", console_only=True)
                
                img_response = requests.get(image_url, timeout=timeout)
                if img_response.status_code == 200:
                    result_img = Image.open(io.BytesIO(img_response.content))
                    result_img = result_img.convert("RGB")
                    
                    img_width, img_height = result_img.size
                    log(f"图像尺寸: {img_width}x{img_height}", "📏")
                    
                    # 保存图片
                    try:
                        output_dir = folder_paths.get_output_directory()
                        wan_dir = os.path.join(output_dir, "wan")
                        os.makedirs(wan_dir, exist_ok=True)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"wan_{timestamp}.png"
                        filepath = os.path.join(wan_dir, filename)
                        
                        result_img.save(filepath, "PNG")
                        log(f"图片已保存: {filepath}", "💾")
                    except Exception as save_error:
                        log(f"保存图片失败: {str(save_error)}", "⚠️")
                    
                    img_array = np.array(result_img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array)[None,]
                    
                    log("处理完成", "✅")
                    log_text = "\n".join(log_messages)
                    
                    return comfy_io.NodeOutput(img_tensor, log_text)
                else:
                    error_msg = f"下载图片失败: {img_response.status_code}"
                    log(error_msg, "❌")
                    raise RuntimeError(error_msg)
            
            elif image_data:
                log("获取到 base64 编码的图像", "🎨")
                
                try:
                    img_bytes = base64.b64decode(image_data)
                    result_img = Image.open(io.BytesIO(img_bytes))
                    result_img = result_img.convert("RGB")
                    
                    img_width, img_height = result_img.size
                    log(f"图像尺寸: {img_width}x{img_height}", "📏")
                    
                    # 保存图片
                    try:
                        output_dir = folder_paths.get_output_directory()
                        wan_dir = os.path.join(output_dir, "wan")
                        os.makedirs(wan_dir, exist_ok=True)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"wan_{timestamp}.png"
                        filepath = os.path.join(wan_dir, filename)
                        
                        result_img.save(filepath, "PNG")
                        log(f"图片已保存: {filepath}", "💾")
                    except Exception as save_error:
                        log(f"保存图片失败: {str(save_error)}", "⚠️")
                    
                    img_array = np.array(result_img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array)[None,]
                    
                    log("处理完成", "✅")
                    log_text = "\n".join(log_messages)
                    
                    return comfy_io.NodeOutput(img_tensor, log_text)
                except Exception as decode_error:
                    error_msg = f"解码 base64 图片失败: {str(decode_error)}"
                    log(error_msg, "❌")
                    raise RuntimeError(error_msg)
            
            else:
                error_msg = f"响应中未找到图像数据，响应: {result}"
                log(error_msg, "❌")
                raise RuntimeError(error_msg)
            
        except Exception as e:
            error_msg = f"发生错误: {str(e)}"
            log(error_msg, "❌")
            raise
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        返回 seed 值，让 ComfyUI 知道输入变化了
        当 seed 改变时，强制重新执行节点，避免使用缓存
        """
        return kwargs.get("seed", 0)
