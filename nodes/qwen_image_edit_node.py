import base64
import io
import json
import os
from datetime import datetime

import requests
import torch
import numpy as np
from PIL import Image
from comfy_api.latest import io as comfy_io

import folder_paths
from ..utils import APILoader


class QwenImageEditNode(comfy_io.ComfyNode):
    """阿里云百炼 千问图像编辑 API 节点，支持 1-3 张图片输入，输出编辑后的图片"""

    FIXED_API_PROVIDER = "qwen_image_edit_api"
    API_KEY_SOURCE = "qwen_image_edit_api"

    # 比例预设 -> 推荐分辨率（宽*高），总像素接近 2048*2048（2.0/3.0 系列均适用，宽高比均在 1:8~8:1 内）
    SIZE_PRESETS = {
        "1:1": "2048*2048",
        "2:3": "1664*2496",
        "3:2": "2496*1664",
        "3:4": "1728*2304",
        "4:3": "2304*1728",
        "9:16": "1440*2560",
        "16:9": "2560*1440",
        "21:9": "3024*1296",
    }
    AUTO_SIZE = "自动"
    CUSTOM_SIZE = "自定义"

    @classmethod
    def _get_api_loader(cls):
        api_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api")
        return APILoader(api_dir)

    @classmethod
    def _load_config(cls):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
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
        return cls.API_KEY_SOURCE, api_keys.get(cls.API_KEY_SOURCE, "")

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        provider = cls._get_api_loader().get_provider(cls.FIXED_API_PROVIDER)
        models = provider.models if provider else ["qwen-image-2.0-pro"]
        size_options = [cls.AUTO_SIZE] + list(cls.SIZE_PRESETS.keys()) + [cls.CUSTOM_SIZE]

        return comfy_io.Schema(
            node_id="QwenImageEdit",
            display_name="Qwen Image Edit API",
            category="Banana",
            inputs=[
                comfy_io.Image.Input("image1", optional=True),
                comfy_io.Image.Input("image2", optional=True),
                comfy_io.Image.Input("image3", optional=True),
                comfy_io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    placeholder="请输入编辑指令，如：将图中的人物换成戴帽子的",
                ),
                comfy_io.Combo.Input(
                    "input_mode",
                    options=["仅图生图", "支持文生图"],
                    default="仅图生图",
                ),
                comfy_io.Combo.Input(
                    "model",
                    options=models,
                    default=models[0] if models else "qwen-image-2.0-pro",
                ),
                comfy_io.Combo.Input(
                    "host_type",
                    options=["china", "overseas", "custom"],
                    default="china",
                ),
                comfy_io.Combo.Input(
                    "image_size",
                    options=size_options,
                    default="1:1",
                ),
                comfy_io.Int.Input(
                    "width",
                    default=1024,
                    min=512,
                    max=4096,
                    step=16,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Int.Input(
                    "height",
                    default=1024,
                    min=512,
                    max=4096,
                    step=16,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.String.Input(
                    "negative_prompt",
                    default="",
                    multiline=True,
                    placeholder="可选，描述不希望出现的内容",
                ),
                comfy_io.Combo.Input(
                    "prompt_extend",
                    options=["启用", "禁用"],
                    default="启用",
                ),
                comfy_io.Combo.Input(
                    "watermark",
                    options=["启用", "禁用"],
                    default="禁用",
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=-1,
                    min=-1,
                    max=2147483647,
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
                comfy_io.Combo.Input(
                    "save_to_output",
                    options=["启用", "禁用"],
                    default="启用",
                ),
            ],
            outputs=[
                comfy_io.Image.Output("result_image"),
                comfy_io.String.Output("log"),
            ],
        )

    @classmethod
    def _tensor_to_data_url(cls, image):
        if len(image.shape) == 4:
            image = image[0]
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.read()).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"

    @classmethod
    def execute(
        cls,
        prompt,
        input_mode,
        model,
        host_type,
        image_size,
        width,
        height,
        negative_prompt,
        prompt_extend,
        watermark,
        seed,
        timeout,
        save_to_output,
        image1=None,
        image2=None,
        image3=None,
    ) -> comfy_io.NodeOutput:
        api_loader = cls._get_api_loader()
        log_messages = []

        def log(msg, icon="", console_only=False):
            full_msg = f"{icon} {msg}" if icon else msg
            if not console_only:
                log_messages.append(full_msg)
            print(f"[Qwen Image Edit] {full_msg}")

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

            # 收集输入图片（最多 3 张）
            images = [
                (idx, img)
                for idx, img in enumerate([image1, image2, image3], 1)
                if img is not None
            ]
            if not images:
                if input_mode == "仅图生图":
                    raise ValueError(
                        "请至少提供 1 张输入图片（或将 input_mode 切换为「支持文生图」以允许纯文生图）"
                    )
                log("未输入图片，使用纯文生图模式", "💭")

            if not prompt or not prompt.strip():
                raise ValueError("请输入编辑指令 prompt")

            # 构建 content 数组：图片在前，文本指令在后
            content = []
            for idx, img in images:
                data_url = cls._tensor_to_data_url(img)
                content.append({"image": data_url})
                log(f"输入图片 {idx}: 已编码", "📸")

            content.append({"text": prompt.strip()})
            log(
                f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}",
                "✍️",
            )

            # 构建 size：自动（不传 size，由模型根据提示词推荐）/ 预设映射 / 自定义宽高（校验总像素与宽高比）
            if image_size == cls.AUTO_SIZE:
                size = None
            elif image_size == cls.CUSTOM_SIZE:
                total_pixels = width * height
                min_pixels = 512 * 512
                max_pixels = 2048 * 2048
                if not (min_pixels <= total_pixels <= max_pixels):
                    raise ValueError(
                        f"自定义分辨率 {width}*{height} 的总像素 {total_pixels} 超出范围"
                        f"（需在 {min_pixels}~{max_pixels} 之间）"
                    )
                if max(width, height) / min(width, height) > 8:
                    raise ValueError(
                        f"自定义分辨率 {width}*{height} 的宽高比需在 1:8 至 8:1 之间"
                    )
                size = f"{width}*{height}"
            else:
                size = cls.SIZE_PRESETS.get(image_size)
            log(f"输出分辨率: {size or '自动（由模型根据提示词推荐）'}", "📐")

            # 布尔参数映射
            prompt_extend_bool = prompt_extend == "启用"
            watermark_bool = watermark == "启用"

            api_host = provider.get_host(host_type).rstrip("/")
            draw_url = f"{api_host}{provider.get_endpoint('draw')}"
            mapped_model = provider.map_model(model)

            request = provider.build_request(
                "draw",
                api_key=api_key,
                model=mapped_model,
                content=content,
                negative_prompt=negative_prompt.strip() if negative_prompt.strip() else None,
                size=size,
                prompt_extend=prompt_extend_bool,
                watermark=watermark_bool,
                seed=seed if seed >= 0 else None,
            )

            log(f"使用 API 提供商: {provider.name}", "🔌")
            log(f"使用 API Host: {api_host}", "🌐")
            log(f"使用模型: {mapped_model}", "🤖")
            log(f"发送请求到: {draw_url}", "📤")
            log(f"输入图片数量: {len(images)}", "🖼️")

            # 发送请求
            response = requests.post(
                draw_url,
                headers=request["headers"],
                json=request["body"],
                timeout=timeout,
            )

            log(f"收到响应，状态码: {response.status_code}", "📨", console_only=True)
            if response.status_code != 200:
                raise RuntimeError(
                    f"API 请求失败: {response.status_code} - {response.text[:500]}"
                )

            try:
                result = response.json()
            except json.JSONDecodeError:
                raise RuntimeError(
                    f"API 返回的不是有效的 JSON 格式\n响应内容: {response.text[:200]}"
                )

            # 从响应中提取图片（兼容 content 为字符串 URL 或 parts 数组两种格式）
            response_format = provider.response_format.get("draw", {})
            content_value = provider._get_nested_value(
                result, response_format.get("image_url_path", "")
            )
            image_url = None
            if isinstance(content_value, list):
                # DashScope 返回 parts 数组，如 [{"image": "https://..."}]
                for part in content_value:
                    if isinstance(part, dict) and part.get("image"):
                        image_url = part["image"]
                        break
            elif isinstance(content_value, str):
                image_url = content_value

            if image_url:
                log_url = (
                    f"{image_url[:10]}..."
                    if image_url.startswith("data:") and len(image_url) > 10
                    else image_url
                )
                log(f"获取到结果图片 URL: {log_url}", "🎨")
                log("正在下载图片...", "⬇️", console_only=True)

                img_response = requests.get(image_url, timeout=timeout)
                if img_response.status_code != 200:
                    raise RuntimeError(f"下载图片失败: {img_response.status_code}")
                result_img = Image.open(io.BytesIO(img_response.content)).convert("RGB")
            else:
                raise RuntimeError(
                    f"响应中未找到图片数据: {json.dumps(result, ensure_ascii=False)[:500]}"
                )

            img_width, img_height = result_img.size
            log(f"图片尺寸: {img_width}x{img_height}", "📏")

            # 保存图片
            if save_to_output == "启用":
                try:
                    output_dir = folder_paths.get_output_directory()
                    banana_dir = os.path.join(output_dir, "banana")
                    os.makedirs(banana_dir, exist_ok=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"qwen_{timestamp}.png"
                    filepath = os.path.join(banana_dir, filename)

                    result_img.save(filepath, "PNG")
                    log(f"图片已保存: {filepath}", "💾")
                except Exception as save_error:
                    log(f"保存图片失败: {str(save_error)}", "⚠️")
            else:
                log("已跳过保存图片（保存功能已禁用）", "ℹ️")

            img_array = np.array(result_img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array)[None,]

            log("处理完成", "✅")
            return comfy_io.NodeOutput(img_tensor, "\n".join(log_messages))

        except requests.exceptions.Timeout:
            error_msg = f"请求超时 ({timeout} 秒)"
            log(error_msg, "⏰")
            raise TimeoutError(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"请求失败: {str(e)}"
            log(error_msg, "❌")
            raise RuntimeError(error_msg)
        except Exception as e:
            log(f"发生错误: {str(e)}", "❌")
            raise

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return kwargs.get("seed", -1)