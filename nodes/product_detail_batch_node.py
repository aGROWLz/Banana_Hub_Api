import json

import torch
from comfy_api.latest import io as comfy_io

from ..utils.product_detail_utils import (
    build_result_manifest,
    parse_plan_json,
    select_task_images,
    validate_plan,
)
from .gpt_image2_node import _GPTImage2BaseNode


class ProductDetailBatchNode(_GPTImage2BaseNode):
    """Execute a validated product-detail plan with one GPT Image request per task."""

    log_prefix = "Product Detail GPT Image"
    save_prefix = "product_detail"

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        provider_ids = cls._get_provider_ids()
        models = set()
        sizes = set()
        for provider_id in provider_ids:
            provider = cls._get_provider(provider_id)
            if provider:
                models.update(provider.models or [])
                sizes.update(provider.image_sizes or [])

        model_options = sorted(models) or ["gpt-image-2"]
        size_options = sorted(sizes) or ["1024x1024"]
        size_options.append("自定义")
        return comfy_io.Schema(
            node_id="ProductDetailBatch",
            display_name="Product Detail GPT Image Batch",
            category="Banana/Product Detail",
            inputs=[
                comfy_io.String.Input("plan_json", default="", multiline=True),
                comfy_io.Image.Input("reference_images"),
                comfy_io.Combo.Input("api_provider", options=provider_ids, default=provider_ids[0]),
                comfy_io.Combo.Input("host_type", options=["china", "overseas", "custom"], default="china"),
                comfy_io.Combo.Input("model", options=model_options, default=model_options[0]),
                comfy_io.Combo.Input("image_size", options=size_options, default=size_options[0]),
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
                comfy_io.Combo.Input(
                    "quality",
                    options=["", "auto", "low", "medium", "high", "standard"],
                    default="",
                ),
                comfy_io.Combo.Input("moderation", options=["", "auto", "low"], default=""),
                comfy_io.Int.Input("retry_count", default=0, min=0, max=10, display_mode=comfy_io.NumberDisplay.number),
                comfy_io.Int.Input(
                    "timeout",
                    default=300,
                    min=10,
                    max=600,
                    step=10,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Combo.Input("save_to_output", options=["启用", "禁用"], default="启用"),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
            ],
            outputs=[
                comfy_io.Image.Output("result_images"),
                comfy_io.String.Output("result_json"),
                comfy_io.String.Output("log"),
            ],
        )

    @classmethod
    def _resolve_size(cls, image_size, width, height):
        if image_size == "自定义":
            from ..utils import validate_custom_dimensions

            return validate_custom_dimensions(width, height)
        return image_size

    @classmethod
    def execute(
        cls,
        plan_json,
        reference_images,
        api_provider,
        host_type,
        model,
        image_size,
        width,
        height,
        quality,
        moderation,
        retry_count,
        timeout,
        save_to_output,
        seed,
    ) -> comfy_io.NodeOutput:
        if reference_images is None or len(reference_images) == 0:
            raise ValueError("At least one reference image is required")

        plan = parse_plan_json(plan_json)
        validate_plan(plan, len(reference_images))
        if plan["errors"]:
            raise ValueError(f"Cannot execute plan with errors: {plan['errors']}")
        tasks = sorted(plan["tasks"], key=lambda task: task["order"])
        if not tasks:
            raise ValueError("Plan contains no generation tasks")

        logs = []

        def log(message, icon="", console_only=False):
            line = f"{icon} {message}" if icon else message
            if not console_only:
                logs.append(line)
            print(f"[Product Detail GPT Image] {line}")

        model = cls._extract_value(model)
        size = cls._resolve_size(image_size, width, height)
        generated_images = []
        output_indices = []

        for task_number, task in enumerate(tasks, start=1):
            task_id = task["task_id"]
            try:
                input_images = select_task_images(reference_images, task["reference_indices"])
                request_name = "edit" if input_images else "draw"
                payload = {
                    "content_type": "multipart/form-data" if request_name == "edit" else "application/json",
                    "body": {
                        "model": model,
                        "prompt": task["prompt"],
                        "n": 1,
                        "size": size,
                        "quality": quality,
                        "moderation": moderation,
                    },
                }
                payload["body"] = {key: value for key, value in payload["body"].items() if value not in (None, "")}
                log(f"Task {task_number}/{len(tasks)}: {task_id} ({task['type']})", "i")
                result = cls._execute_request(
                    request_name=request_name,
                    api_provider=api_provider,
                    payload=payload,
                    timeout=timeout,
                    save_to_output=save_to_output,
                    input_images=input_images,
                    mask=None,
                    log=log,
                    host_type=host_type,
                    retry_count=retry_count,
                )
                generated_images.append(result)
                output_indices.append(len(generated_images))
            except Exception as error:
                raise RuntimeError(f"Task {task_id} failed: {error}") from error

        result_images = torch.cat(generated_images, dim=0)
        manifest = build_result_manifest(tasks, output_indices)
        log(f"Generated {len(generated_images)} images", "OK")
        return comfy_io.NodeOutput(
            result_images,
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
            "\n".join(logs),
        )

    @classmethod
    def IS_CHANGED(cls, plan_json="", reference_images=None, seed=0, **kwargs):
        image_hash = hash(reference_images.cpu().numpy().tobytes()) if reference_images is not None else 0
        return hash((plan_json, image_hash, seed))
