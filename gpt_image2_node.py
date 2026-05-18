import base64
import io
import json
import os
from datetime import datetime

import folder_paths
import numpy as np
import requests
import torch
from PIL import Image
from comfy_api.latest import io as comfy_io

from .api_loader import APILoader
from .gpt_image2_utils import build_request_payload


class _GPTImage2BaseNode(comfy_io.ComfyNode):
    api_loader = None
    provider_id = ""
    log_prefix = "GPT-Image-2"
    save_prefix = "gpt_image2"

    @classmethod
    def _init_api_loader(cls):
        if cls.api_loader is None:
            api_dir = os.path.join(os.path.dirname(__file__), "api")
            cls.api_loader = APILoader(api_dir)

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
    def _get_provider(cls):
        cls._init_api_loader()
        return cls.api_loader.get_provider(cls.provider_id)

    @classmethod
    def _collect_input_images(cls, *images):
        return [(idx, img) for idx, img in enumerate(images, 1) if img is not None]

    @classmethod
    def _build_headers(cls, request_format, api_key):
        headers = {}
        for key, value in request_format.get("headers", {}).items():
            if isinstance(value, str) and "{api_key}" in value:
                headers[key] = value.replace("{api_key}", api_key)
            else:
                headers[key] = value
        if "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @classmethod
    def _send_request(cls, *, url, headers, payload, timeout, images, log):
        content_type = payload["content_type"]
        if content_type == "multipart/form-data":
            files = cls._build_image_files(images, log)
            log(
                f"发送 multipart 请求，文件数: {len(files)}，字段: {list(payload['body'].keys())}",
                "i",
                console_only=True,
            )
            request_kwargs = {
                "headers": headers,
                "data": payload["body"],
                "timeout": timeout,
            }
            if files:
                request_kwargs["files"] = files
            return requests.post(url, **request_kwargs)

        log(
            f"发送 JSON 请求，字段: {list(payload['body'].keys())}",
            "i",
            console_only=True,
        )
        return requests.post(url, headers=headers, json=payload["body"], timeout=timeout)

    @classmethod
    def _build_image_files(cls, images, log):
        image_files = []
        for idx, img_tensor in images:
            if len(img_tensor.shape) == 4:
                img_tensor = img_tensor[0]

            height, width, _ = img_tensor.shape
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            buffered_file = io.BytesIO()
            pil_img.save(buffered_file, format="PNG")
            buffered_file.seek(0)
            image_files.append(("image[]", (f"image_{idx}.png", buffered_file, "image/png")))
            log(f"图片 {idx}: {width}x{height}, 已编码", "i")
        return image_files

    @classmethod
    def _extract_image(cls, result, provider, timeout, log):
        response_format = provider.response_format.get("draw", {})
        image_url = provider._get_nested_value(result, response_format.get("image_url_path", ""))
        b64_json = provider._get_nested_value(result, response_format.get("b64_json_path", ""))

        if image_url:
            log("获取到结果图片 URL", "i")
            img_response = requests.get(image_url, timeout=timeout)
            if img_response.status_code == 200:
                result_img = Image.open(io.BytesIO(img_response.content)).convert("RGB")
                log(f"图片尺寸: {result_img.size[0]}x{result_img.size[1]}", "i")
                return result_img
            log(f"下载图片失败: {img_response.status_code}，尝试 base64 解码", "!")

        if b64_json:
            log("获取到 base64 编码的图片", "i")
            result_img = Image.open(io.BytesIO(base64.b64decode(b64_json))).convert("RGB")
            log(f"图片尺寸: {result_img.size[0]}x{result_img.size[1]}", "i")
            return result_img

        raise RuntimeError("响应中未找到图片数据（URL 或 base64）")

    @classmethod
    def _save_image(cls, result_img, log):
        try:
            output_dir = folder_paths.get_output_directory()
            banana_dir = os.path.join(output_dir, "banana")
            os.makedirs(banana_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(banana_dir, f"{cls.save_prefix}_{timestamp}.png")
            result_img.save(filepath, "PNG")
            log(f"图片已保存: {filepath}", "i")
        except Exception as save_error:
            log(f"保存图片失败: {str(save_error)}", "!")

    @classmethod
    def _finalize_response(cls, response, provider, timeout, save_to_output, log):
        log(f"收到响应，状态码: {response.status_code}", "i", console_only=True)
        if response.status_code != 200:
            raise RuntimeError(f"API 请求失败: {response.status_code} - {response.text}")

        try:
            result = response.json()
        except json.JSONDecodeError:
            raise RuntimeError(f"API 返回的不是有效的 JSON 格式\n响应内容: {response.text[:200]}")

        log(f"响应: {json.dumps(result, ensure_ascii=False)[:500]}...", "i")
        result_img = cls._extract_image(result, provider, timeout, log)
        if save_to_output == "启用":
            cls._save_image(result_img, log)
        else:
            log("已跳过保存图片", "!")

        img_array = np.array(result_img.convert("RGB")).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        return img_tensor

    @classmethod
    def _execute_request(cls, *, request_name, payload, timeout, save_to_output, input_images, log, host_type):
        provider = cls._get_provider()
        if not provider:
            raise ValueError(f"未找到 {cls.provider_id} API 配置")

        config = cls._load_config()
        api_key = config.get("api_keys", {}).get("gpt_image2_api", "")
        if not api_key:
            raise ValueError("错误: 未设置 API Key，请在配置文件的 api_keys.gpt_image2_api 中设置")

        request_format = provider.request_format.get(request_name, {})
        payload["content_type"] = request_format.get(
            "content_type", payload.get("content_type", "application/json")
        )
        headers = cls._build_headers(request_format, api_key)
        endpoint = provider.get_endpoint(request_name).replace("{model}", str(payload["body"].get("model", "")))
        url = f"{provider.get_host(host_type).rstrip('/')}{endpoint}"

        log(f"使用 API: {provider.name}", "i")
        log(f"使用 API Host: {provider.get_host(host_type).rstrip('/')}", "i")
        log(f"发送请求到: {url}", "i")
        log(f"请求类型: {request_name}", "i")
        if input_images:
            log(f"输入图片数量: {len(input_images)}", "i")

        response = cls._send_request(
            url=url,
            headers=headers,
            payload=payload,
            timeout=timeout,
            images=input_images,
            log=log,
        )
        return cls._finalize_response(response, provider, timeout, save_to_output, log)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return kwargs.get("seed", 0)


class GPTImage2Node(_GPTImage2BaseNode):
    provider_id = "gpt_image2_api"
    log_prefix = "GPT-Image-2 Reverse"
    save_prefix = "gpt_image2_reverse"

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        provider = cls._get_provider()
        models = provider.models if provider and provider.models else ["gpt-image-2-all"]
        image_sizes = provider.image_sizes if provider and provider.image_sizes else [
            "1024x1024",
            "1792x1024",
            "1024x1792",
        ]

        return comfy_io.Schema(
            node_id="GPTImage2Reverse",
            display_name="GPT-Image-2 (逆向)",
            category="Banana",
            inputs=[
                comfy_io.Image.Input("image1", optional=True),
                comfy_io.Image.Input("image2", optional=True),
                comfy_io.Image.Input("image3", optional=True),
                comfy_io.Image.Input("image4", optional=True),
                comfy_io.Image.Input("image5", optional=True),
                comfy_io.String.Input("prompt", default="", multiline=True),
                comfy_io.Combo.Input("host_type", options=["china", "overseas", "custom"], default="china"),
                comfy_io.Combo.Input("model", options=models, default=models[0]),
                comfy_io.Combo.Input("size", options=image_sizes, default=image_sizes[0]),
                comfy_io.Int.Input("n", default=1, min=1, max=10, display_mode=comfy_io.NumberDisplay.number),
                comfy_io.Combo.Input(
                    "quality",
                    options=["", "standard", "hd", "auto", "low", "medium", "high"],
                    default="",
                ),
                comfy_io.Combo.Input(
                    "response_format",
                    options=["", "url", "b64_json"],
                    default="",
                ),
                comfy_io.Combo.Input("style", options=["", "vivid", "natural"], default=""),
                comfy_io.String.Input("user", default=""),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
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
                comfy_io.Combo.Input("save_to_output", options=["启用", "禁用"], default="启用"),
            ],
            outputs=[comfy_io.Image.Output("result_image"), comfy_io.String.Output("log")],
        )

    @classmethod
    def execute(
        cls,
        host_type,
        prompt,
        model,
        size,
        n,
        quality,
        response_format,
        style,
        user,
        seed,
        timeout,
        save_to_output,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
    ) -> comfy_io.NodeOutput:
        log_messages = []

        def log(msg, icon="", console_only=False):
            full_msg = f"{icon} {msg}" if icon else msg
            if not console_only:
                log_messages.append(full_msg)
            print(f"[{cls.log_prefix}] {full_msg}")

        input_images = cls._collect_input_images(image1, image2, image3, image4, image5)
        request_name = "edit" if input_images else "draw"
        provider = cls._get_provider()
        mapped_model = provider.map_model(model) if provider else model

        payload = build_request_payload(
            provider.config,
            prompt=prompt,
            model=mapped_model,
            size=size,
            n=n,
            quality=quality,
            response_format=response_format,
            style=style,
            user=user,
        )
        payload["body"]["model"] = mapped_model

        img_tensor = cls._execute_request(
            request_name=request_name,
            payload=payload,
            timeout=timeout,
            save_to_output=save_to_output,
            input_images=input_images,
            log=log,
            host_type=host_type,
        )
        log("处理完成", "OK")
        return comfy_io.NodeOutput(img_tensor, "\n".join(log_messages))


class GPTImage2FullNode(_GPTImage2BaseNode):
    provider_id = "gpt_image2_full_api"
    log_prefix = "GPT-Image-2 Full"
    save_prefix = "gpt_image2_full"

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        provider = cls._get_provider()
        models = provider.models if provider and provider.models else ["gpt-image-2"]
        image_sizes = provider.image_sizes if provider and provider.image_sizes else [
            "auto",
            "1024x1024",
            "1536x1024",
            "1024x1536",
        ]

        return comfy_io.Schema(
            node_id="GPTImage2Full",
            display_name="GPT-Image-2 (全参数)",
            category="Banana",
            inputs=[
                comfy_io.Image.Input("image1"),
                comfy_io.Image.Input("image2", optional=True),
                comfy_io.Image.Input("image3", optional=True),
                comfy_io.Image.Input("image4", optional=True),
                comfy_io.Image.Input("image5", optional=True),
                comfy_io.String.Input("prompt", default="", multiline=True),
                comfy_io.Combo.Input("host_type", options=["china", "overseas", "custom"], default="china"),
                comfy_io.Combo.Input("model", options=models, default=models[0]),
                comfy_io.Combo.Input("size", options=image_sizes, default=image_sizes[0]),
                comfy_io.Int.Input("n", default=1, min=1, max=10, display_mode=comfy_io.NumberDisplay.number),
                comfy_io.Combo.Input(
                    "quality",
                    options=["", "auto", "low", "medium", "high", "standard"],
                    default="",
                ),
                comfy_io.Combo.Input("moderation", options=["", "auto", "low"], default=""),
                comfy_io.String.Input("user", default=""),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
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
                comfy_io.Combo.Input("save_to_output", options=["启用", "禁用"], default="启用"),
            ],
            outputs=[comfy_io.Image.Output("result_image"), comfy_io.String.Output("log")],
        )

    @classmethod
    def execute(
        cls,
        host_type,
        prompt,
        model,
        size,
        n,
        quality,
        moderation,
        user,
        seed,
        timeout,
        save_to_output,
        image1,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
    ) -> comfy_io.NodeOutput:
        log_messages = []

        def log(msg, icon="", console_only=False):
            full_msg = f"{icon} {msg}" if icon else msg
            if not console_only:
                log_messages.append(full_msg)
            print(f"[{cls.log_prefix}] {full_msg}")

        input_images = cls._collect_input_images(image1, image2, image3, image4, image5)
        provider = cls._get_provider()
        mapped_model = provider.map_model(model) if provider else model

        payload = {
            "content_type": "multipart/form-data",
            "body": {
                "model": mapped_model,
                "prompt": prompt,
                "n": n,
                "size": size,
                "quality": quality,
                "moderation": moderation,
                "user": user,
            },
        }
        payload["body"] = {k: v for k, v in payload["body"].items() if v not in (None, "")}

        img_tensor = cls._execute_request(
            request_name="edit",
            payload=payload,
            timeout=timeout,
            save_to_output=save_to_output,
            input_images=input_images,
            log=log,
            host_type=host_type,
        )
        log("处理完成", "OK")
        return comfy_io.NodeOutput(img_tensor, "\n".join(log_messages))


GPTImage2ReverseNode = GPTImage2Node
