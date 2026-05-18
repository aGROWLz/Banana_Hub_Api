import base64
import io
import json
import os
from datetime import datetime

import requests
from PIL import Image
from comfy_api.latest import io as comfy_io

import folder_paths
from .api_loader import APILoader


class GeminiVisionNode(comfy_io.ComfyNode):
    """Gemini image/video understanding node."""

    FIXED_API_PROVIDER = "gemini_api"
    FIXED_MODEL = "gemini-3.1-flash-lite-preview"
    API_KEY_SOURCE = "gemini"
    LEGACY_API_KEY_SOURCES = ("gemini_api", "bltai_api")

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
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载配置文件失败: {e}，使用默认配置")
        return default_config

    @classmethod
    def _resolve_api_key(cls, api_keys):
        if api_keys.get(cls.API_KEY_SOURCE):
            return cls.API_KEY_SOURCE, api_keys[cls.API_KEY_SOURCE]
        for key_name in cls.LEGACY_API_KEY_SOURCES:
            if api_keys.get(key_name):
                return key_name, api_keys[key_name]
        return cls.API_KEY_SOURCE, ""

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="GeminiVision",
            display_name="Gemini Vision API",
            category="Banana",
            inputs=[
                comfy_io.Image.Input("image1", optional=True),
                comfy_io.Image.Input("image2", optional=True),
                comfy_io.Image.Input("image3", optional=True),
                comfy_io.Image.Input("image4", optional=True),
                comfy_io.Image.Input("image5", optional=True),
                comfy_io.String.Input(
                    "video_url",
                    default="",
                    multiline=False,
                    placeholder="https://example.com/video.mp4",
                ),
                comfy_io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    placeholder="请输入分析问题或指令",
                ),
                comfy_io.Combo.Input(
                    "host_type",
                    options=["china", "overseas", "custom"],
                    default="china",
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
                    max=65536,
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
                    default="禁用",
                ),
            ],
            outputs=[
                comfy_io.String.Output("response_text"),
                comfy_io.String.Output("log"),
            ],
        )

    @classmethod
    def _tensor_to_data_url(cls, image):
        if len(image.shape) == 4:
            image = image[0]

        height, width, _ = image.shape
        img_np = (image.cpu().numpy() * 255).astype("uint8")
        pil_img = Image.fromarray(img_np)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.read()).decode("utf-8")
        return f"data:image/png;base64,{img_base64}", width, height

    @classmethod
    def _build_messages(cls, prompt, images, video_url, log):
        content = []

        if prompt and prompt.strip():
            content.append({"type": "text", "text": prompt.strip()})
            text_for_log = prompt[:100] + "..." if len(prompt) > 100 else prompt
            log(f"Prompt: {text_for_log}", "i")

        for idx, image in images:
            data_url, width, height = cls._tensor_to_data_url(image)
            content.append({"type": "image_url", "image_url": {"url": data_url}})
            log(f"输入图片 {idx}: {width}x{height}", "i")

        if video_url and video_url.strip():
            cleaned_url = video_url.strip()
            content.append({"type": "video_url", "video_url": {"url": cleaned_url}})
            log(f"输入视频 URL: {cleaned_url}", "i")

        return [{"role": "user", "content": content}]

    @classmethod
    def execute(
        cls,
        host_type,
        prompt,
        temperature,
        max_tokens,
        timeout,
        save_response,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        video_url="",
    ) -> comfy_io.NodeOutput:
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
                raise ValueError(f"未找到 API 提供商 {cls.FIXED_API_PROVIDER}")

            api_key_source, api_key = cls._resolve_api_key(config.get("api_keys", {}))
            if not api_key:
                raise ValueError(
                    f"错误: 未设置 API Key，请在配置文件的 api_keys.{cls.API_KEY_SOURCE} 中设置"
                )

            images = [(idx, img) for idx, img in enumerate([image1, image2, image3, image4, image5], 1) if img is not None]
            has_video = bool(video_url and video_url.strip())
            if not images and not has_video:
                raise ValueError("请至少提供图片或视频 URL 之一")

            endpoint_name = "video_understanding" if has_video else "image_understanding"
            api_host = provider.get_host(host_type).rstrip("/")
            draw_url = f"{api_host}{provider.get_endpoint(endpoint_name)}"
            messages = cls._build_messages(prompt, images, video_url, log)

            request_body = {
                "model": cls.FIXED_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            log(f"使用 API 提供商: {provider.name}", "i")
            log(f"使用 API Host: {api_host}", "i")
            log(f"使用 API Key: {api_key_source}", "i")
            log(f"使用模型: {cls.FIXED_MODEL}", "i")
            log(f"请求类型: {endpoint_name}", "i")
            log(f"发送请求到: {draw_url}", "i")
            log(f"内容数量: {len(messages[0]['content'])}", "i", console_only=True)

            response = requests.post(draw_url, headers=headers, json=request_body, timeout=timeout)
            log(f"收到响应，状态码: {response.status_code}", "i", console_only=True)
            if response.status_code != 200:
                raise RuntimeError(f"API 请求失败: {response.status_code} - {response.text[:500]}")

            try:
                result = response.json()
            except json.JSONDecodeError:
                raise RuntimeError(f"API 返回的不是有效的 JSON 格式\n响应内容: {response.text[:200]}")

            response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(response_text, list):
                response_text = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item) for item in response_text
                )
            if not response_text:
                response_text = "未收到有效响应"

            log(f"响应内容: {response_text[:200]}..." if len(response_text) > 200 else f"响应内容: {response_text}", "i")

            if save_response == "启用":
                try:
                    output_dir = folder_paths.get_output_directory()
                    banana_dir = os.path.join(output_dir, "banana")
                    os.makedirs(banana_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filepath = os.path.join(banana_dir, f"gemini_{timestamp}.txt")
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(response_text)
                    log(f"响应已保存: {filepath}", "i")
                except Exception as save_error:
                    log(f"保存响应失败: {str(save_error)}", "!")

            log("处理完成", "OK")
            return comfy_io.NodeOutput(response_text, "\n".join(log_messages))
        except requests.exceptions.Timeout:
            error_msg = f"请求超时 ({timeout} 秒)"
            log(error_msg, "!")
            raise TimeoutError(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"请求失败: {str(e)}"
            log(error_msg, "X")
            raise RuntimeError(error_msg)
        except Exception as e:
            log(f"发生错误: {str(e)}", "X")
            raise

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        hashes = []
        for key in ("image1", "image2", "image3", "image4", "image5"):
            image = kwargs.get(key)
            if image is not None:
                hashes.append(hash(image.cpu().numpy().tobytes()))
        if kwargs.get("video_url"):
            hashes.append(hash(kwargs["video_url"]))
        return hash(tuple(hashes)) if hashes else 0
