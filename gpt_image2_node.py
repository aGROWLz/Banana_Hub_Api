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


class GPTImage2ReverseNode(comfy_io.ComfyNode):
    """
    GPT-Image-2 图片编辑节点（逆向分组）
    使用 bltai 域名，支持独立的 API Key 配置
    支持最多5张参考图片输入
    仅支持参数：model、prompt、size、image
    """

    # 初始化 API 加载器
    api_loader = None

    @classmethod
    def _init_api_loader(cls):
        """初始化 API 加载器"""
        if cls.api_loader is None:
            api_dir = os.path.join(os.path.dirname(__file__), "api")
            cls.api_loader = APILoader(api_dir)

    @classmethod
    def _load_config(cls):
        """加载用户配置文件"""
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
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

        # 获取 gpt_image2 提供商
        provider = cls.api_loader.get_provider("gpt_image2_api")

        if provider:
            models = provider.models if provider.models else ["gpt-image-2"]
            image_sizes = provider.image_sizes if provider.image_sizes else ["auto", "1024x1024", "1536x1024", "1024x1536"]
        else:
            models = ["gpt-image-2"]
            image_sizes = ["auto", "1024x1024", "1536x1024", "1024x1536", "2048x2048", "2048x1152", "3840x2160", "2160x3840"]

        return comfy_io.Schema(
            node_id="GPTImage2Reverse",
            display_name="GPT-Image-2 Edit (逆向)",
            category="Banana",
            inputs=[
                comfy_io.Image.Input("image1", optional=True),
                comfy_io.Image.Input("image2", optional=True),
                comfy_io.Image.Input("image3", optional=True),
                comfy_io.Image.Input("image4", optional=True),
                comfy_io.Image.Input("image5", optional=True),
                comfy_io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                ),
                comfy_io.Combo.Input(
                    "host_type",
                    options=["china", "overseas", "custom"],
                    default="china"
                ),
                comfy_io.Combo.Input(
                    "model",
                    options=models,
                    default="gpt-image-2"
                ),
                comfy_io.Combo.Input(
                    "size",
                    options=image_sizes,
                    default="auto"
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
                comfy_io.Combo.Input(
                    "save_to_output",
                    options=["启用", "禁用"],
                    default="启用"
                ),
            ],
            outputs=[
                comfy_io.Image.Output("result_image"),
                comfy_io.String.Output("log"),
            ],
        )

    @classmethod
    def execute(cls, host_type, prompt, model, size, seed, timeout, save_to_output,
                image1=None, image2=None, image3=None, image4=None, image5=None) -> comfy_io.NodeOutput:

        cls._init_api_loader()
        log_messages = []

        def log(msg, icon="", console_only=False):
            """添加日志并立即打印到控制台"""
            full_msg = f"{icon} {msg}" if icon else msg
            if not console_only:
                log_messages.append(full_msg)
            print(f"[GPT-Image-2] {full_msg}")

        try:
            # 加载配置
            config = cls._load_config()
            provider = cls.api_loader.get_provider("gpt_image2_api")

            if not provider:
                error_msg = "未找到 GPT-Image-2 API 配置"
                log(error_msg, "❌")
                raise ValueError(error_msg)

            log(f"使用 API: {provider.name}", "🔌")

            # 确定使用的 API host
            api_host = provider.get_host(host_type)

            # 从配置文件获取 API key - 使用独立的 gpt_image2_api key
            api_keys = config.get("api_keys", {})
            api_key = api_keys.get("gpt_image2_api", "")

            if not api_key:
                error_msg = "错误: 未设置 API Key，请在配置文件的 api_keys.gpt_image2_api 中设置"
                log(error_msg, "❌")
                raise ValueError(error_msg)

            api_host = api_host.rstrip('/')

            log(f"使用 API Host: {api_host}", "🌐")
            log(f"使用模型: {model}", "🤖")
            log(f"图片尺寸: {size}", "📐")
            log(f"随机种子: {seed}", "🎲")

            # 收集所有输入的图片
            input_images = []
            for idx, img in enumerate([image1, image2, image3, image4, image5], 1):
                if img is not None:
                    input_images.append((idx, img))

            if not input_images:
                error_msg = "图片编辑模式需要至少输入一张图片"
                log(error_msg, "❌")
                raise ValueError(error_msg)

            # 处理输入图片 - 为 multipart/form-data 准备
            image_files = []  # 用于 multipart/form-data
            image_buffers = []  # 保持 BytesIO 对象的引用

            log(f"检测到 {len(input_images)} 张输入图片，将使用图片编辑模式", "🖼️")

            for img_idx, img_tensor in input_images:
                if len(img_tensor.shape) == 4:
                    img_tensor = img_tensor[0]

                height, width, channels = img_tensor.shape
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)

                # 为 multipart 创建 BytesIO
                buffered_file = io.BytesIO()
                pil_img.save(buffered_file, format="PNG")
                buffered_file.seek(0)
                image_buffers.append(buffered_file)  # 保持引用
                image_files.append(('image', (f'image_{img_idx}.png', buffered_file, 'image/png')))

                log(f"图片 {img_idx}: {width}x{height}, 已编码", "📸")

            # 构建请求
            draw_endpoint = provider.get_endpoint("draw")
            draw_url = f"{api_host}{draw_endpoint}"

            log(f"发送请求到: {draw_url}", "📤")
            log(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}", "✍️")

            # 构建 multipart/form-data 请求
            headers = {
                "Authorization": f"Bearer {api_key}"
            }

            # 构建 data 字段 - 逆向分组仅支持 model、prompt、size 参数
            data = {
                "model": model,
                "prompt": prompt,
                "size": size
            }

            log(f"发送 multipart 请求，文件数: {len(image_files)}, 数据字段: {list(data.keys())}", "📋", console_only=True)

            try:
                response = requests.post(
                    draw_url,
                    headers=headers,
                    files=image_files,
                    data=data,
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

            # 解析响应
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                error_msg = "API 返回的不是有效的 JSON 格式"
                log(error_msg, "❌")
                log(f"HTTP 状态码: {response.status_code}", "ℹ️")
                log(f"响应内容: {response.text[:500]}", "ℹ️")
                raise RuntimeError(f"{error_msg}\n响应内容: {response.text[:200]}")

            log(f"响应: {json.dumps(result, ensure_ascii=False)[:500]}...", "📥")

            # 解析响应获取图片 - 支持 url 和 b64_json 两种格式
            response_format = provider.response_format.get("draw", {})

            image_url = None
            b64_json = None

            if "image_url_path" in response_format:
                image_url = provider._get_nested_value(result, response_format["image_url_path"])

            if "b64_json_path" in response_format:
                b64_json = provider._get_nested_value(result, response_format["b64_json_path"])

            # 优先尝试从 URL 下载
            if image_url:
                log(f"获取到结果图片 URL", "🎨")
                log("正在下载图片...", "⬇️", console_only=True)

                try:
                    img_response = requests.get(image_url, timeout=timeout)
                    if img_response.status_code == 200:
                        result_img = Image.open(io.BytesIO(img_response.content))
                        result_img = result_img.convert("RGB")

                        img_width, img_height = result_img.size
                        log(f"图片尺寸: {img_width}x{img_height}", "📏")

                        # 保存图片
                        if save_to_output == "启用":
                            try:
                                output_dir = folder_paths.get_output_directory()
                                banana_dir = os.path.join(output_dir, "banana")
                                os.makedirs(banana_dir, exist_ok=True)

                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"gpt_image2_{timestamp}.png"
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
                        log_text = "\n".join(log_messages)

                        return comfy_io.NodeOutput(img_tensor, log_text)
                    else:
                        log(f"下载图片失败: {img_response.status_code}，尝试 base64 解码", "⚠️")
                except Exception as e:
                    log(f"下载图片失败: {str(e)}，尝试 base64 解码", "⚠️")

            # 如果 URL 下载失败或没有 URL，尝试 base64
            if b64_json:
                log("获取到 base64 编码的图片", "🎨")

                try:
                    img_data = base64.b64decode(b64_json)
                    result_img = Image.open(io.BytesIO(img_data))
                    result_img = result_img.convert("RGB")

                    img_width, img_height = result_img.size
                    log(f"图片尺寸: {img_width}x{img_height}", "📏")

                    # 保存图片
                    if save_to_output == "启用":
                        try:
                            output_dir = folder_paths.get_output_directory()
                            banana_dir = os.path.join(output_dir, "banana")
                            os.makedirs(banana_dir, exist_ok=True)

                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"gpt_image2_{timestamp}.png"
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
                    log_text = "\n".join(log_messages)

                    return comfy_io.NodeOutput(img_tensor, log_text)
                except Exception as decode_error:
                    error_msg = f"解码 base64 图片失败: {str(decode_error)}"
                    log(error_msg, "❌")
                    raise RuntimeError(error_msg)

            # 两种格式都没有找到
            error_msg = "响应中未找到图片数据（URL 或 base64）"
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


class GPTImage2OfficialNode(comfy_io.ComfyNode):
    """
    GPT-Image-2 图片编辑节点（官转组）
    使用 bltai 域名，支持独立的 API Key 配置
    支持最多5张参考图片输入
    兼容 edits 接口所有参数透传
    """

    # 初始化 API 加载器
    api_loader = None

    @classmethod
    def _init_api_loader(cls):
        """初始化 API 加载器"""
        if cls.api_loader is None:
            api_dir = os.path.join(os.path.dirname(__file__), "api")
            cls.api_loader = APILoader(api_dir)

    @classmethod
    def _load_config(cls):
        """加载用户配置文件"""
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
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

        # 获取 gpt_image2 提供商
        provider = cls.api_loader.get_provider("gpt_image2_api")

        if provider:
            models = provider.models if provider.models else ["gpt-image-2"]
            image_sizes = provider.image_sizes if provider.image_sizes else ["auto", "1024x1024", "1536x1024", "1024x1536", "2048x2048", "2048x1152", "3840x2160", "2160x3840"]
        else:
            models = ["gpt-image-2"]
            image_sizes = ["auto", "1024x1024", "1536x1024", "1024x1536", "2048x2048", "2048x1152", "3840x2160", "2160x3840"]

        return comfy_io.Schema(
            node_id="GPTImage2Official",
            display_name="GPT-Image-2 Edit (官转)",
            category="Banana",
            inputs=[
                comfy_io.Image.Input("image1", optional=True),
                comfy_io.Image.Input("image2", optional=True),
                comfy_io.Image.Input("image3", optional=True),
                comfy_io.Image.Input("image4", optional=True),
                comfy_io.Image.Input("image5", optional=True),
                comfy_io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                ),
                comfy_io.Combo.Input(
                    "host_type",
                    options=["china", "overseas", "custom"],
                    default="china"
                ),
                comfy_io.Combo.Input(
                    "model",
                    options=models,
                    default="gpt-image-2"
                ),
                comfy_io.Combo.Input(
                    "size",
                    options=image_sizes,
                    default="auto"
                ),
                comfy_io.Combo.Input(
                    "quality",
                    options=["auto", "low", "medium", "high"],
                    default="auto"
                ),
                comfy_io.Combo.Input(
                    "response_format",
                    options=["", "url", "b64_json"],
                    default=""
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
                comfy_io.Combo.Input(
                    "save_to_output",
                    options=["启用", "禁用"],
                    default="启用"
                ),
            ],
            outputs=[
                comfy_io.Image.Output("result_image"),
                comfy_io.String.Output("log"),
            ],
        )

    @classmethod
    def execute(cls, host_type, prompt, model, size, quality, response_format, seed, timeout, save_to_output,
                image1=None, image2=None, image3=None, image4=None, image5=None) -> comfy_io.NodeOutput:

        cls._init_api_loader()
        log_messages = []

        def log(msg, icon="", console_only=False):
            """添加日志并立即打印到控制台"""
            full_msg = f"{icon} {msg}" if icon else msg
            if not console_only:
                log_messages.append(full_msg)
            print(f"[GPT-Image-2-Official] {full_msg}")

        try:
            # 加载配置
            config = cls._load_config()
            provider = cls.api_loader.get_provider("gpt_image2_api")

            if not provider:
                error_msg = "未找到 GPT-Image-2 API 配置"
                log(error_msg, "❌")
                raise ValueError(error_msg)

            log(f"使用 API: {provider.name}", "🔌")

            # 确定使用的 API host
            api_host = provider.get_host(host_type)

            # 从配置文件获取 API key - 使用独立的 gpt_image2_api key
            api_keys = config.get("api_keys", {})
            api_key = api_keys.get("gpt_image2_api", "")

            if not api_key:
                error_msg = "错误: 未设置 API Key，请在配置文件的 api_keys.gpt_image2_api 中设置"
                log(error_msg, "❌")
                raise ValueError(error_msg)

            api_host = api_host.rstrip('/')

            log(f"使用 API Host: {api_host}", "🌐")
            log(f"使用模型: {model}", "🤖")
            log(f"图片尺寸: {size}", "📐")
            log(f"质量: {quality}", "⭐")
            log(f"响应格式: {response_format if response_format else '默认'}", "📄")
            log(f"随机种子: {seed}", "🎲")

            # 收集所有输入的图片
            input_images = []
            for idx, img in enumerate([image1, image2, image3, image4, image5], 1):
                if img is not None:
                    input_images.append((idx, img))

            if not input_images:
                error_msg = "图片编辑模式需要至少输入一张图片"
                log(error_msg, "❌")
                raise ValueError(error_msg)

            # 处理输入图片 - 为 multipart/form-data 准备
            image_files = []  # 用于 multipart/form-data
            image_buffers = []  # 保持 BytesIO 对象的引用

            log(f"检测到 {len(input_images)} 张输入图片，将使用图片编辑模式", "🖼️")

            for img_idx, img_tensor in input_images:
                if len(img_tensor.shape) == 4:
                    img_tensor = img_tensor[0]

                height, width, channels = img_tensor.shape
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)

                # 为 multipart 创建 BytesIO
                buffered_file = io.BytesIO()
                pil_img.save(buffered_file, format="PNG")
                buffered_file.seek(0)
                image_buffers.append(buffered_file)  # 保持引用
                image_files.append(('image', (f'image_{img_idx}.png', buffered_file, 'image/png')))

                log(f"图片 {img_idx}: {width}x{height}, 已编码", "📸")

            # 构建请求
            draw_endpoint = provider.get_endpoint("draw")
            draw_url = f"{api_host}{draw_endpoint}"

            log(f"发送请求到: {draw_url}", "📤")
            log(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}", "✍️")

            # 构建 multipart/form-data 请求
            headers = {
                "Authorization": f"Bearer {api_key}"
            }

            # 构建 data 字段 - 官转组支持所有参数
            data = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "quality": quality
            }

            # 如果 response_format 不为空，添加到请求中
            if response_format:
                data["response_format"] = response_format

            log(f"发送 multipart 请求，文件数: {len(image_files)}, 数据字段: {list(data.keys())}", "📋", console_only=True)

            try:
                response = requests.post(
                    draw_url,
                    headers=headers,
                    files=image_files,
                    data=data,
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

            # 解析响应
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                error_msg = "API 返回的不是有效的 JSON 格式"
                log(error_msg, "❌")
                log(f"HTTP 状态码: {response.status_code}", "ℹ️")
                log(f"响应内容: {response.text[:500]}", "ℹ️")
                raise RuntimeError(f"{error_msg}\n响应内容: {response.text[:200]}")

            log(f"响应: {json.dumps(result, ensure_ascii=False)[:500]}...", "📥")

            # 解析响应获取图片 - 支持 url 和 b64_json 两种格式
            response_format_cfg = provider.response_format.get("draw", {})

            image_url = None
            b64_json = None

            if "image_url_path" in response_format_cfg:
                image_url = provider._get_nested_value(result, response_format_cfg["image_url_path"])

            if "b64_json_path" in response_format_cfg:
                b64_json = provider._get_nested_value(result, response_format_cfg["b64_json_path"])

            # 优先尝试从 URL 下载
            if image_url:
                log(f"获取到结果图片 URL", "🎨")
                log("正在下载图片...", "⬇️", console_only=True)

                try:
                    img_response = requests.get(image_url, timeout=timeout)
                    if img_response.status_code == 200:
                        result_img = Image.open(io.BytesIO(img_response.content))
                        result_img = result_img.convert("RGB")

                        img_width, img_height = result_img.size
                        log(f"图片尺寸: {img_width}x{img_height}", "📏")

                        # 保存图片
                        if save_to_output == "启用":
                            try:
                                output_dir = folder_paths.get_output_directory()
                                banana_dir = os.path.join(output_dir, "banana")
                                os.makedirs(banana_dir, exist_ok=True)

                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"gpt_image2_official_{timestamp}.png"
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
                        log_text = "\n".join(log_messages)

                        return comfy_io.NodeOutput(img_tensor, log_text)
                    else:
                        log(f"下载图片失败: {img_response.status_code}，尝试 base64 解码", "⚠️")
                except Exception as e:
                    log(f"下载图片失败: {str(e)}，尝试 base64 解码", "⚠️")

            # 如果 URL 下载失败或没有 URL，尝试 base64
            if b64_json:
                log("获取到 base64 编码的图片", "🎨")

                try:
                    img_data = base64.b64decode(b64_json)
                    result_img = Image.open(io.BytesIO(img_data))
                    result_img = result_img.convert("RGB")

                    img_width, img_height = result_img.size
                    log(f"图片尺寸: {img_width}x{img_height}", "📏")

                    # 保存图片
                    if save_to_output == "启用":
                        try:
                            output_dir = folder_paths.get_output_directory()
                            banana_dir = os.path.join(output_dir, "banana")
                            os.makedirs(banana_dir, exist_ok=True)

                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"gpt_image2_official_{timestamp}.png"
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
                    log_text = "\n".join(log_messages)

                    return comfy_io.NodeOutput(img_tensor, log_text)
                except Exception as decode_error:
                    error_msg = f"解码 base64 图片失败: {str(decode_error)}"
                    log(error_msg, "❌")
                    raise RuntimeError(error_msg)

            # 两种格式都没有找到
            error_msg = "响应中未找到图片数据（URL 或 base64）"
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
