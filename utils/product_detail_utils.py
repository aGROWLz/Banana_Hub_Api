import json
from typing import Any

import torch


REQUIRED_PLAN_FIELDS = {"schema_version", "reference_images", "tasks", "errors"}


def is_white_placeholder_image(image: torch.Tensor, max_side: int = 128) -> bool:
    """Match XKU's small, nearly pure-white placeholder image filter."""
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3 or image.shape[0] > max_side or image.shape[1] > max_side:
        return False
    return bool(torch.min(image) >= (250.0 / 255.0))


def aggregate_image_batches(image_inputs: list[Any], max_slots: int = 10) -> torch.Tensor:
    """Concatenate optional ComfyUI image inputs and skip white placeholders."""
    if len(image_inputs) > max_slots:
        raise ValueError(f"At most {max_slots} image slots are supported")

    batches = []
    for image in image_inputs:
        if image is None:
            continue
        if not isinstance(image, torch.Tensor):
            raise ValueError("Each image input must be a torch.Tensor")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError("Each image input must have shape [B,H,W,C]")
        batches.extend(
            sample.unsqueeze(0)
            for sample in image
            if not is_white_placeholder_image(sample)
        )

    if not batches:
        raise ValueError("At least one image input is required")
    return torch.cat(batches, dim=0)


def parse_plan_json(text: str) -> dict:
    if not isinstance(text, str):
        raise ValueError("Agent response must be raw JSON")

    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Agent response must be raw JSON: {exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Agent response must be a JSON object")
    return parsed


def validate_plan(plan: dict, reference_count: int) -> dict:
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a JSON object")
    if set(plan) != REQUIRED_PLAN_FIELDS:
        missing = REQUIRED_PLAN_FIELDS - set(plan)
        extra = set(plan) - REQUIRED_PLAN_FIELDS
        raise ValueError(f"Invalid plan fields: missing={sorted(missing)}, extra={sorted(extra)}")
    if plan["schema_version"] != "1.0":
        raise ValueError("Unsupported plan schema_version")
    if not isinstance(reference_count, int) or reference_count < 1:
        raise ValueError("reference_count must be a positive integer")

    references = plan["reference_images"]
    tasks = plan["tasks"]
    errors = plan["errors"]
    if not isinstance(references, list) or not isinstance(tasks, list) or not isinstance(errors, list):
        raise ValueError("reference_images, tasks, and errors must be arrays")
    if errors and tasks:
        raise ValueError("A plan with errors must not contain tasks")

    expected_indices = set(range(1, reference_count + 1))
    reference_indices = []
    for reference in references:
        if not isinstance(reference, dict):
            raise ValueError("Each reference image must be an object")
        index = reference.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("Reference index must be an integer")
        if index in reference_indices:
            raise ValueError(f"Duplicate reference index: {index}")
        reference_indices.append(index)
        if not isinstance(reference.get("role"), str) or not reference["role"].strip():
            raise ValueError("Reference role must be a non-empty string")
        if not isinstance(reference.get("description"), str):
            raise ValueError("Reference description must be a string")
    if set(reference_indices) != expected_indices:
        raise ValueError("reference_images must contain every reference index")

    task_ids = []
    orders = []
    prompts = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Each task must be an object")
        task_id = task.get("task_id")
        order = task.get("order")
        prompt = task.get("prompt")
        selected_indices = task.get("reference_indices")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if task_id in task_ids:
            raise ValueError(f"Duplicate task_id: {task_id}")
        if not isinstance(order, int) or isinstance(order, bool):
            raise ValueError("Task order must be an integer")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Task prompt must be a non-empty string")
        if not isinstance(selected_indices, list) or not selected_indices:
            raise ValueError("Task reference_indices must be a non-empty array")
        if any(index not in expected_indices for index in selected_indices):
            raise ValueError("Task contains an unknown reference index")
        if any(not isinstance(index, int) or isinstance(index, bool) for index in selected_indices):
            raise ValueError("Task reference index must be an integer")
        task_ids.append(task_id)
        orders.append(order)
        prompts.append(prompt.strip())

    if len(set(prompts)) != len(prompts):
        raise ValueError("Task prompts must be distinct")
    if sorted(orders) != list(range(1, len(tasks) + 1)):
        raise ValueError("Task order must be continuous starting at 1")
    return plan


def select_task_images(reference_batch: Any, reference_indices: list[int]) -> list[tuple[int, Any]]:
    if not isinstance(reference_indices, list) or not reference_indices:
        raise ValueError("reference_indices must be a non-empty array")
    selected = []
    batch_size = len(reference_batch)
    for index in reference_indices:
        if not isinstance(index, int) or isinstance(index, bool) or index < 1 or index > batch_size:
            raise ValueError(f"Unknown reference index: {index}")
        selected.append((index, reference_batch[index - 1]))
    return selected


def build_result_manifest(tasks: list[dict], output_indices: list[int]) -> dict:
    if len(tasks) != len(output_indices):
        raise ValueError("Each task must have exactly one output image index")
    results = []
    for task, image_index in zip(tasks, output_indices):
        results.append(
            {
                "task_id": task["task_id"],
                "type": task["type"],
                "order": task["order"],
                "image_indices": [image_index],
            }
        )
    return {"schema_version": "1.0", "results": results}
