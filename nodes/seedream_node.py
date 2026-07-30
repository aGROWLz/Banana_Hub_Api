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


class SeedreamImageGenerationNode(comfy_io.ComfyNode):
    """
    Seedream Image Generation API 节点
    支持 KK 中转站 doubao-seedream-5.0 图像生成
    不输入图片时为文生图，输入图片时自动切换为图生图
    """

    api_loader = None

    @classmethod
    def _init_api_loader(cls):
        if cls.api_loader is None:
            api_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api")
            cls.api_loader = APILoader(api_dir)

    @classmethod
    def _load_config(cls):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
        default_config = {"api_keys": {}}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载配置文件失败: {e}，使用默认配置")
        return default_config

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
    def _truncate_base64(cls, data_url):
        """截断 base64 data URL，只保留前缀和前 10 个字符"""
        prefix = "data:image/png;base64,"
        idx = data_url.find(prefix)
        if idx != -1:
            base64_part = data_url[idx + len(prefix):]
            return f"{prefix}{base64_part[:10]}..."
        return data_url[:20] + "..."

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        cls._init_api_loader()
        provider = cls.api_loader.get_provider("KK_seedream_api")
        if not provider:
            models = ["doubao-seedream-5-0-260128"]
            image_sizes = ["1024x1024", "1664x936", "936x1664", "1K", "2K", "4K"]
        else:
            models = provider.models
            image_sizes = provider.image_sizes
        
        image_sizes.append("自定义")

        return comfy_io.Schema(
            node_id="SeedreamImageGeneration",
            display_name="Seedream Image Generation API",
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
                    "model",
                    options=models,
                    default=models[0] if models else "doubao-seedream-5-0-260128"
                ),
                comfy_io.Combo.Input(
                    "image_size",
                    options=image_sizes,
                    default="1024x1024" if "1024x1024" in image_sizes else image_sizes[0]
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
                comfy_io.Int.Input(
                    "seed",
                    default=-1,
                    min=-1,
                    max=0xffffffffffffffff,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Int.Input(
                    "n",
                    default=1,
                    min=1,
                    max=15,
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
    def execute(cls, prompt, model, image_size, width, height, seed, n, timeout,
                image1=None, image2=None, image3=None, image4=None, image5=None) -> comfy_io.NodeOutput:
        
        cls._init_api_loader()
        log_messages = []

        def log(msg, icon="", console_only=False):
            full_msg = f"{icon} {msg}" if icon else msg
            if not console_only:
                log_messages.append(full_msg)
            print(f"[Seedream API] {full_msg}")

        try:
            config = cls._load_config()
            provider = cls.api_loader.get_provider("KK_seedream_api")

            if not provider:
                raise ValueError("未找到 Seedream API 提供商配置 (KK_seedream_api)")

            log(f"使用 API 提供商: {provider.name}", "🔌")

            api_host = provider.get_host("china")
            api_keys = config.get("api_keys", {})
            api_key = api_keys.get("KK_seedream_api", "")

            if not api_key:
                raise ValueError("错误: 未设置 API Key，请在配置文件的 api_keys.KK_seedream_api 中设置")

            api_host = api_host.rstrip('/')

            # 处理尺寸：自定义时用宽高拼接
            if image_size == "自定义":
                actual_size = f"{width}x{height}"
                log(f"使用自定义尺寸: {actual_size}", "📐")
            else:
                actual_size = image_size

            log(f"使用 API Host: {api_host}", "🌐")
            log(f"使用模型: {model}", "🎨")
            log(f"图片尺寸: {actual_size}", "📐")
            log(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}", "✍️")

            # 收集输入图片
            input_images = [(idx, img) for idx, img in
                           enumerate([image1, image2, image3, image4, image5], 1)
                           if img is not None]

            if input_images:
                log(f"检测到 {len(input_images)} 张输入图片，使用图生图模式", "🖼️")
            else:
                log("未检测到输入图片，使用文生图模式", "📝")

            draw_endpoint = provider.get_endpoint("draw")
            draw_url = f"{api_host}{draw_endpoint}"

            request_body = {
                "model": model,
                "prompt": prompt,
                "size": actual_size,
                "seed": seed,
                "response_format": "url",
                "output_format": "png",
                "watermark": False
            }

            # n > 1 时使用组图模式
            if n > 1:
                request_body["sequential_image_generation"] = "auto"
                request_body["sequential_image_generation_options"] = {"max_images": n}
                log(f"组图模式: 最多生成 {n} 张", "🖼️")

            if input_images:
                image_urls = [cls._tensor_to_data_url(img) for _, img in input_images]
                for idx in range(len(input_images)):
                    log(f"输入图片 {idx + 1} 已编码", "📎", console_only=True)
                request_body["image"] = image_urls if len(image_urls) > 1 else image_urls[0]

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            log(f"发送图像生成请求到: {draw_url}", "📤")
            log(f"生成数量: {n}", "📊")
            # 打印请求体前对 base64 图片数据做截断
            debug_body = {}
            for k, v in request_body.items():
                if k == "image":
                    if isinstance(v, list):
                        debug_body[k] = [cls._truncate_base64(img) for img in v]
                    else:
                        debug_body[k] = cls._truncate_base64(v)
                else:
                    debug_body[k] = v
            log(f"请求体: {json.dumps(debug_body, ensure_ascii=False)}", "📋", console_only=True)

            try:
                response = requests.post(draw_url, headers=headers, json=request_body, timeout=timeout)
            except requests.exceptions.Timeout:
                raise TimeoutError(f"请求超时 ({timeout}秒)，请检查网络连接或增加 timeout 参数")
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"请求失败: {e}")

            log(f"收到响应，状态码: {response.status_code}", "📨", console_only=True)

            if response.status_code != 200:
                raise RuntimeError(f"API 请求失败: {response.status_code} - {response.text}")

            try:
                result = response.json()
            except json.JSONDecodeError:
                raise RuntimeError(f"API 返回的不是有效的 JSON 格式\n响应内容: {response.text}")

            log(f"请求响应: {json.dumps(result, ensure_ascii=False)}", "📥")

            if "data" not in result or len(result["data"]) == 0:
                raise RuntimeError(f"响应中未找到图像数据，响应: {result}")

            # 处理所有返回的图片
            data_items = result["data"]
            log(f"API 返回了 {len(data_items)} 张图片", "🎨")

            output_dir = folder_paths.get_output_directory()
            seedream_dir = os.path.join(output_dir, "seedream")
            os.makedirs(seedream_dir, exist_ok=True)

            result_tensors = []

            for i, item in enumerate(data_items):
                img_url = item.get("url")
                img_b64 = item.get("b64_json")

                if img_url:
                    log(f"正在下载第 {i + 1} 张图片...", "⬇️", console_only=True)
                    img_response = requests.get(img_url, timeout=timeout)
                    if img_response.status_code == 200:
                        pil_img = Image.open(io.BytesIO(img_response.content)).convert("RGB")
                    else:
                        log(f"第 {i + 1} 张图片下载失败: {img_response.status_code}", "⚠️")
                        continue
                elif img_b64:
                    pil_img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
                else:
                    log(f"第 {i + 1} 张图片无 url 和 b64_json，跳过", "⚠️")
                    continue

                # 保存图片
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"seedream_{timestamp}_{i + 1}.png"
                    filepath = os.path.join(seedream_dir, filename)
                    pil_img.save(filepath, "PNG")
                    log(f"图片 {i + 1} 已保存: {filepath}", "💾")
                except Exception as save_error:
                    log(f"保存图片 {i + 1} 失败: {save_error}", "⚠️")

                img_array = np.array(pil_img).astype(np.float32) / 255.0
                result_tensors.append(torch.from_numpy(img_array))

            if not result_tensors:
                raise RuntimeError("未能成功处理任何图片")

            # 如果只有一张，保持 (H, W, C) 然后加 batch 维度
            # 如果多张且尺寸相同，堆叠为 batch
            if len(result_tensors) == 1:
                img_tensor = result_tensors[0][None,]
            else:
                # 检查尺寸是否一致
                first_shape = result_tensors[0].shape
                if all(t.shape == first_shape for t in result_tensors):
                    img_tensor = torch.stack(result_tensors, dim=0)
                else:
                    # 尺寸不一致时只输出第一张
                    log(f"多张图片尺寸不一致，仅输出第 1 张", "⚠️")
                    img_tensor = result_tensors[0][None,]

            log(f"处理完成，共输出 {img_tensor.shape[0]} 张图片", "✅")
            log_text = "\n".join(log_messages)
            return comfy_io.NodeOutput(img_tensor, log_text)

        except Exception as e:
            log(f"发生错误: {e}", "❌")
            raise

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        hashes = [kwargs.get("seed", 0)]
        for key in ("image1", "image2", "image3", "image4", "image5"):
            image = kwargs.get(key)
            if image is not None:
                hashes.append(hash(image.cpu().numpy().tobytes()))
        return hash(tuple(hashes))
