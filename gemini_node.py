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
from .api_loader import APILoader


class GeminiVisionNode(comfy_io.ComfyNode):
    """
    Gemini Vision API 节点
    支持图片和视频URL输入，返回文本分析结果
    使用 bltai_api 的 API Key
    """

    FIXED_API_PROVIDER = "gemini_api"
    FIXED_MODEL = "gemini-3.1-flash-lite-preview"
    API_KEY_SOURCE = "bltai_api"

    @classmethod
    def _get_api_loader(cls):
        api_dir = os.path.join(os.path.dirname(__file__), "api")
        return APILoader(api_dir)

    @classmethod
    def _load_config(cls):
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        default_config = {"api_keys": {}}
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载配置文件失败: {e}，使用默认配置")
                return default_config
        return default_config

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="GeminiVision",
            display_name="Gemini Vision API",
            category="Banana",
            inputs=[
                comfy_io.Image.Input("image", optional=True),
                comfy_io.String.Input(
                    "video_url",
                    default="",
                    multiline=False,
                    placeholder="https://example.com/video.mp4 (可选)"
                ),
                comfy_io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    placeholder="请输入分析问题或指令"
                ),
                comfy_io.Combo.Input(
                    "host_type",
                    options=["china", "overseas", "custom"],
                    default="china"
                ),
                comfy_io.Float.Input(
                    "temperature",
                    default=0.7,
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Int.Input(
                    "max_tokens",
                    default=4000,
                    min=100,
                    max=8000,
                    step=100,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Int.Input(
                    "timeout",
                    default=60,
                    min=10,
                    max=300,
                    step=10,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Combo.Input(
                    "save_response",
                    options=["启用", "禁用"],
                    default="禁用"
                ),
            ],
            outputs=[
                comfy_io.String.Output("response_text"),
                comfy_io.String.Output("log"),
            ],
        )

    @classmethod
    def execute(cls, host_type, prompt, temperature, max_tokens, timeout, save_response,
                image=None, video_url="") -> comfy_io.NodeOutput:
        
        api_loader = cls._get_api_loader()
        log_messages = []
        
        def log(msg, icon="", console_only=False):
            full_msg = f"{icon} {msg}" if icon else msg
            if not console_only:
                log_messages.append(full_msg)
            print(f"[Gemini Vision] {full_msg}")
        
        try:
            config = cls._load_config()
            provider = api_loader.get_provider(cls.FIXED_API_PROVIDER)
            
            if not provider:
                error_msg = f"未找到 API 提供商: {cls.FIXED_API_PROVIDER}"
                log(error_msg, "❌")
                raise ValueError(error_msg)
            
            log(f"使用 API 提供商: {provider.name}", "🔌")
            
            api_host = provider.get_host(host_type)
            api_keys = config.get("api_keys", {})
            api_key = api_keys.get(cls.API_KEY_SOURCE, "")
            
            if not api_key:
                error_msg = f"错误: 未设置 API Key，请在配置文件的 api_keys.{cls.API_KEY_SOURCE} 中设置"
                log(error_msg, "❌")
                raise ValueError(error_msg)
            
            api_host = api_host.rstrip('/')
            draw_endpoint = provider.get_endpoint("draw")
            draw_url = f"{api_host}{draw_endpoint}"
            
            log(f"使用 API Host: {api_host}", "🌐")
            log(f"使用模型: {cls.FIXED_MODEL}", "🤖")
            log(f"Temperature: {temperature}, Max Tokens: {max_tokens}", "⚙️")
            
            messages = [{"role": "user", "content": []}]
            
            if image is not None:
                if len(image.shape) == 4:
                    image = image[0]
                
                height, width, channels = image.shape
                img_np = (image.cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                
                buffered = io.BytesIO()
                pil_img.save(buffered, format="PNG")
                buffered.seek(0)
                img_base64 = base64.b64encode(buffered.read()).decode('utf-8')
                
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                })
                
                log(f"输入图片: {width}x{height}", "📸")
            
            if video_url and video_url.strip():
                video_url = video_url.strip()
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": video_url
                    }
                })
                log(f"输入视频 URL: {video_url}", "🎥")
            
            if prompt and prompt.strip():
                messages[0]["content"].append({
                    "type": "text",
                    "text": prompt.strip()
                })
                log(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}", "✍️")
            
            if image is None and not video_url:
                error_msg = "请至少提供图片或视频 URL 之一"
                log(error_msg, "❌")
                raise ValueError(error_msg)
            
            request_body = {
                "model": cls.FIXED_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            log(f"发送请求到: {draw_url}", "📤")
            log(f"内容: {len(messages[0]['content'])} 个元素", "📋", console_only=True)
            
            try:
                response = requests.post(
                    draw_url,
                    headers=headers,
                    json=request_body,
                    timeout=timeout
                )
            except requests.exceptions.Timeout:
                error_msg = f"请求超时 ({timeout}秒)"
                log(error_msg, "⏰")
                raise TimeoutError(error_msg)
            except requests.exceptions.RequestException as e:
                error_msg = f"请求失败: {str(e)}"
                log(error_msg, "❌")
                raise RuntimeError(error_msg)
            
            log(f"收到响应，状态码: {response.status_code}", "📨", console_only=True)
            
            if response.status_code != 200:
                error_msg = f"API 请求失败: {response.status_code} - {response.text[:200]}"
                log(error_msg, "❌")
                raise RuntimeError(error_msg)
            
            try:
                result = response.json()
            except json.JSONDecodeError:
                error_msg = "API 返回的不是有效的 JSON 格式"
                log(error_msg, "❌")
                log(f"响应内容: {response.text[:200]}", "ℹ️")
                raise RuntimeError(error_msg)
            
            response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if response_text:
                log(f"收到响应，长度: {len(response_text)} 字符", "📥")
                log(f"响应内容: {response_text[:200]}..." if len(response_text) > 200 else f"响应内容: {response_text}", "💬")
                
                if save_response == "启用":
                    try:
                        output_dir = folder_paths.get_output_directory()
                        banana_dir = os.path.join(output_dir, "banana")
                        os.makedirs(banana_dir, exist_ok=True)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"gemini_{timestamp}.txt"
                        filepath = os.path.join(banana_dir, filename)
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(response_text)
                        
                        log(f"响应已保存: {filepath}", "💾")
                    except Exception as save_error:
                        log(f"保存响应失败: {str(save_error)}", "⚠️")
            else:
                response_text = "未收到有效响应"
                log(response_text, "⚠️")
            
            log("处理完成", "✅")
            log_text = "\n".join(log_messages)
            
            return comfy_io.NodeOutput(response_text, log_text)
            
        except Exception as e:
            error_msg = f"发生错误: {str(e)}"
            log(error_msg, "❌")
            raise
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        image = kwargs.get("image")
        if image is not None:
            # 将 tensor 转换为可哈希的值（使用 id 或形状信息）
            return hash(image.cpu().numpy().tobytes())
        return 0
