import base64
import io
import json
import os

import numpy as np
import requests
from PIL import Image
from comfy_api.latest import io as comfy_io

from ..utils import APILoader
from ..utils.product_detail_utils import parse_plan_json, validate_plan


class ProductDetailAgentNode(comfy_io.ComfyNode):
    """Plan product-detail images with Gemini Vision and return raw JSON plus the image batch."""

    FIXED_API_PROVIDER = "gemini_api"
    FIXED_MODEL = "gemini-3.1-flash-lite"
    API_FOLDER = "gemini"
    SKILL_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "agent_skills",
        "product-detail-planner",
        "SKILL.md",
    )
    PRODUCT_DETAIL_SKILL_PATH = SKILL_PATH

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="ProductDetailAgent",
            display_name="Product Detail Planner Agent",
            category="Banana/Product Detail",
            inputs=[
                comfy_io.Image.Input("images"),
                comfy_io.String.Input("prompt", default="", multiline=True),
                comfy_io.Combo.Input("host_type", options=["china", "overseas", "custom"], default="china"),
                comfy_io.Float.Input(
                    "temperature",
                    default=0.2,
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Int.Input(
                    "max_tokens",
                    default=12000,
                    min=100,
                    max=65536,
                    step=100,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Int.Input(
                    "timeout",
                    default=120,
                    min=10,
                    max=600,
                    step=10,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
            ],
            outputs=[
                comfy_io.String.Output("plan_json"),
                comfy_io.Image.Output("reference_images"),
                comfy_io.String.Output("log"),
            ],
        )

    @classmethod
    def _load_config(cls):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as file:
                    return json.load(file)
            except Exception as error:
                print(f"加载配置文件失败: {error}，使用默认配置")
        return {"api_keys": {}}

    @classmethod
    def _load_skill(cls):
        with open(cls.SKILL_PATH, "r", encoding="utf-8") as file:
            return file.read()

    @classmethod
    def _get_provider(cls):
        api_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api", cls.API_FOLDER)
        return APILoader(api_dir).get_provider(cls.FIXED_API_PROVIDER)

    @classmethod
    def _resolve_api_key(cls, api_keys, host_type):
        host_key = {"china": "gemini_china", "overseas": "gemini_overseas"}.get(host_type)
        if host_key and api_keys.get(host_key):
            return host_key, api_keys[host_key]
        if api_keys.get("gemini"):
            return "gemini", api_keys["gemini"]
        for key_name in ("gemini_api", "bltai_api"):
            if api_keys.get(key_name):
                return key_name, api_keys[key_name]
        return host_key or "gemini", ""

    @staticmethod
    def _split_image_batch(images):
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        return [(index, images[index - 1:index]) for index, _ in enumerate(images, start=1)]

    @staticmethod
    def _tensor_to_data_url(image):
        if len(image.shape) == 4:
            image = image[0]
        image_np = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        buffered = io.BytesIO()
        Image.fromarray(image_np).save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @classmethod
    def _build_messages(cls, skill, prompt, image_items):
        content = [{"type": "text", "text": prompt.strip()}]
        for _, image in image_items:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": cls._tensor_to_data_url(image)},
                }
            )
        return [
            {"role": "system", "content": skill},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _extract_response_text(result):
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            return "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
        return content if isinstance(content, str) else str(content)

    @classmethod
    def execute(
        cls,
        images,
        prompt,
        host_type,
        temperature,
        max_tokens,
        timeout,
        seed,
    ) -> comfy_io.NodeOutput:
        if images is None or len(images) == 0:
            raise ValueError("At least one reference image is required")
        if not prompt or not prompt.strip():
            raise ValueError("A product-detail planning prompt is required")

        logs = []

        def log(message, icon=""):
            line = f"{icon} {message}" if icon else message
            logs.append(line)
            print(f"[Product Detail Agent] {line}")

        config = cls._load_config()
        provider = cls._get_provider()
        if not provider:
            raise ValueError(f"未找到 API 提供商 {cls.FIXED_API_PROVIDER}")
        key_source, api_key = cls._resolve_api_key(config.get("api_keys", {}), host_type)
        if not api_key:
            raise ValueError(f"错误: 未设置 API Key，请在配置文件的 api_keys.{key_source} 中设置")

        image_items = cls._split_image_batch(images)
        messages = cls._build_messages(cls._load_skill(), prompt, image_items)
        request = provider.build_request(
            "image_understanding",
            api_key=api_key,
            model=cls.FIXED_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        host = provider.get_host(host_type).rstrip("/")
        url = f"{host}{provider.get_endpoint('image_understanding')}"
        log(f"Sending {len(image_items)} reference images to Gemini", "i")
        response = requests.post(url, headers=request["headers"], json=request["body"], timeout=timeout)
        if response.status_code != 200:
            raise RuntimeError(f"Gemini request failed: {response.status_code} - {response.text[:500]}")
        try:
            result = response.json()
        except json.JSONDecodeError as error:
            raise RuntimeError("Gemini response was not valid JSON") from error

        response_text = cls._extract_response_text(result).strip()
        plan = parse_plan_json(response_text)
        validate_plan(plan, len(image_items))
        plan_json = json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
        log(f"Validated {len(plan['tasks'])} generation tasks", "OK")
        return comfy_io.NodeOutput(plan_json, images, "\n".join(logs))

    @classmethod
    def IS_CHANGED(cls, images=None, prompt="", seed=0, **kwargs):
        image_hash = hash(images.cpu().numpy().tobytes()) if images is not None else 0
        return hash((image_hash, prompt, seed))
