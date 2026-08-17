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

from ..utils import APILoader, validate_custom_dimensions


class _GPTImage2BaseNode(comfy_io.ComfyNode):
    api_loader = None
    log_prefix = "GPT-Image-2"
    save_prefix = "gpt_image2"
    # 本节点专属的 api 配置子文件夹
    API_FOLDER = "gpt_image2"

    @classmethod
    def _init_api_loader(cls):
        if cls.api_loader is None:
            api_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api", cls.API_FOLDER)
            cls.api_loader = APILoader(api_dir)

    @classmethod
    def _get_provider_ids(cls):
        cls._init_api_loader()
        ids = cls.api_loader.get_provider_ids()
        return ids if ids else ["vector_gpt2"]

    @classmethod
    def _get_provider(cls, api_provider):
        cls._init_api_loader()
        return cls.api_loader.get_provider(api_provider)

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
    def _send_request(cls, *, url, headers, payload, timeout, images, mask, log):
        content_type = payload["content_type"]
        if content_type == "multipart/form-data":
            files = cls._build_image_files(images, mask, log)
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
    def _build_image_files(cls, images, mask, log):
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

        # 可选遮罩：作为独立的 mask 字段上传，PNG 格式，尺寸需与 image 一致
        if mask is not None:
            if len(mask.shape) == 4:
                mask = mask[0]
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_np)
            mask_buffer = io.BytesIO()
            mask_img.save(mask_buffer, format="PNG")
            mask_buffer.seek(0)
            image_files.append(("mask", ("mask.png", mask_buffer, "image/png")))
            log(f"遮罩已编码: {mask_img.size[0]}x{mask_img.size[1]}", "i")

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

        # 截断响应中的 b64_json 数据用于日志显示
        debug_result = json.dumps(result, ensure_ascii=False)
        import re
        debug_result = re.sub(r'"b64_json"\s*:\s*"[^"]{10}[^"]*"', lambda m: m.group()[:m.group().index('"b64_json"') + len('"b64_json"') + 13] + '..."', debug_result)
        log(f"响应: {debug_result[:500]}...", "i")
        result_img = cls._extract_image(result, provider, timeout, log)
        if save_to_output == "启用":
            cls._save_image(result_img, log)
        else:
            log("已跳过保存图片", "!")

        img_array = np.array(result_img.convert("RGB")).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        return img_tensor

    @classmethod
    def _execute_request(cls, *, request_name, api_provider, payload, timeout, save_to_output, input_images, mask, log, host_type):
        provider = cls._get_provider(api_provider)
        if not provider:
            raise ValueError(f"未找到 {api_provider} API 配置")

        config = cls._load_config()
        api_key = config.get("api_keys", {}).get(api_provider, "")
        if not api_key:
            raise ValueError(f"错误: 未设置 API Key，请在配置文件的 api_keys.{api_provider} 中设置")

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
        if mask is not None:
            log("检测到遮罩输入，启用局部编辑", "i")

        response = cls._send_request(
            url=url,
            headers=headers,
            payload=payload,
            timeout=timeout,
            images=input_images,
            mask=mask,
            log=log,
        )
        return cls._finalize_response(response, provider, timeout, save_to_output, log)

    @classmethod
    def _extract_value(cls, param_with_providers):
        """从 'value (provider1, provider2)' 格式中提取实际值"""
        if "(" in param_with_providers:
            return param_with_providers.split("(")[0].strip()
        return param_with_providers

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return kwargs.get("seed", 0)


class GPTImage2FullNode(_GPTImage2BaseNode):
    log_prefix = "GPT-Image-2 Full"
    save_prefix = "gpt_image2_full"

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        provider_ids = cls._get_provider_ids()

        # 合并所有供应商的模型名（去重，直接以模型名为准，不标注供应商）
        models_set = set()
        # 合并所有供应商的尺寸预设
        sizes_set = set()
        for pid in provider_ids:
            provider = cls._get_provider(pid)
            if provider:
                models_set.update(provider.models or [])
                sizes_set.update(provider.image_sizes or [])

        models_options = sorted(models_set) if models_set else ["gpt-image-2"]
        default_model = models_options[0]

        # 尺寸预设 + “自定义”选项；默认“自定义”沿用宽高输入（1024x1024）
        CUSTOM_SIZE = "自定义"
        size_options = sorted(sizes_set) if sizes_set else ["1024x1024"]
        size_options.append(CUSTOM_SIZE)
        default_size = CUSTOM_SIZE

        return comfy_io.Schema(
            node_id="GPTImage2Full",
            display_name="GPT-Image-2",
            category="Banana",
            inputs=[
                comfy_io.Image.Input("image1", optional=True),
                comfy_io.Image.Input("image2", optional=True),
                comfy_io.Image.Input("image3", optional=True),
                comfy_io.Image.Input("image4", optional=True),
                comfy_io.Image.Input("image5", optional=True),
                comfy_io.Image.Input("mask", optional=True),
                comfy_io.String.Input("prompt", default="", multiline=True),
                comfy_io.Combo.Input(
                    "api_provider",
                    options=provider_ids,
                    default=provider_ids[0],
                ),
                comfy_io.Combo.Input("host_type", options=["china", "overseas", "custom"], default="china"),
                comfy_io.Combo.Input("model", options=models_options, default=default_model),
                comfy_io.Combo.Input(
                    "image_size",
                    options=size_options,
                    default=default_size,
                ),
                comfy_io.Int.Input(
                    "width",
                    default=1024,
                    min=0,
                    max=3840,
                    step=16,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Int.Input(
                    "height",
                    default=1024,
                    min=0,
                    max=3840,
                    step=16,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
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
        api_provider,
        host_type,
        prompt,
        model,
        image_size,
        width,
        height,
        n,
        quality,
        moderation,
        user,
        seed,
        timeout,
        save_to_output,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        image5=None,
        mask=None,
    ) -> comfy_io.NodeOutput:
        log_messages = []

        def log(msg, icon="", console_only=False):
            full_msg = f"{icon} {msg}" if icon else msg
            if not console_only:
                log_messages.append(full_msg)
            print(f"[{cls.log_prefix}] {full_msg}")

        model = cls._extract_value(model)
        provider = cls._get_provider(api_provider)
        if not provider:
            raise ValueError(f"未找到 {api_provider} API 配置")
        mapped_model = provider.map_model(model)

        # 尺寸：选择“自定义”时用宽高输入（≤3840px，16 的倍数），否则用下拉预设值
        if image_size == "自定义":
            size = validate_custom_dimensions(width, height)
        else:
            size = image_size

        log(f"使用 API: {provider.name}", "i")
        log(f"目标尺寸: {size}", "i")

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
        request_name = "edit" if cls._collect_input_images(image1, image2, image3, image4, image5) else "draw"
        input_images = cls._collect_input_images(image1, image2, image3, image4, image5)
        if request_name == "draw":
            payload["content_type"] = "application/json"

        img_tensor = cls._execute_request(
            request_name=request_name,
            api_provider=api_provider,
            payload=payload,
            timeout=timeout,
            save_to_output=save_to_output,
            input_images=input_images,
            mask=mask,
            log=log,
            host_type=host_type,
        )
        log("处理完成", "OK")
        return comfy_io.NodeOutput(img_tensor, "\n".join(log_messages))
