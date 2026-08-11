import base64
import io
import json
import os
import time
from datetime import datetime

import requests
import torch
import numpy as np
from PIL import Image
from comfy_api.latest import io as comfy_io

import folder_paths
from ..utils import APILoader


class KieAiQwen3ImageNode(comfy_io.ComfyNode):
    """Kie.ai Qwen3 Pro Image to Image 节点：本地图片自动上传，异步任务轮询，输出编辑后图片"""

    FIXED_API_PROVIDER = "kieai_qwen3_api"
    API_KEY_SOURCE = "kieai_qwen3_api"
    FIXED_MODEL = "qwen3/pro-image-to-image"
    RUNNING_STATES = ("waiting", "queuing", "generating")

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
    def _get_upload_host(cls, provider, host_type):
        return provider.config.get("upload_hosts", {}).get(host_type, "https://kieai.redpandaai.co").rstrip("/")

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        provider = cls._get_api_loader().get_provider(cls.FIXED_API_PROVIDER)
        if provider:
            image_sizes = provider.image_sizes
            resolutions = provider.config.get("resolutions", ["1K", "2K"])
            output_formats = provider.config.get("output_formats", ["png", "jpeg"])
        else:
            image_sizes = ["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9"]
            resolutions = ["1K", "2K"]
            output_formats = ["png", "jpeg"]

        return comfy_io.Schema(
            node_id="KieAiQwen3Image",
            display_name="Kie.ai Qwen3 Image to Image API",
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
                    "resolution",
                    options=resolutions,
                    default=resolutions[0] if resolutions else "1K",
                ),
                comfy_io.Combo.Input(
                    "image_size",
                    options=image_sizes,
                    default="1:1" if "1:1" in image_sizes else image_sizes[0],
                ),
                comfy_io.Combo.Input(
                    "output_format",
                    options=output_formats,
                    default="png" if "png" in output_formats else output_formats[0],
                ),
                comfy_io.Combo.Input(
                    "prompt_extend",
                    options=["启用", "禁用"],
                    default="启用",
                ),
                comfy_io.String.Input(
                    "negative_prompt",
                    default="",
                    multiline=True,
                    placeholder="可选，描述不希望出现的内容",
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=-1,
                    min=-1,
                    max=2147483647,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Combo.Input(
                    "nsfw_checker",
                    options=["启用", "禁用"],
                    default="禁用",
                ),
                comfy_io.Combo.Input(
                    "host_type",
                    options=["china", "overseas", "custom"],
                    default="china",
                ),
                comfy_io.Int.Input(
                    "timeout",
                    default=300,
                    min=10,
                    max=600,
                    step=10,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Int.Input(
                    "max_retries",
                    default=200,
                    min=1,
                    max=10000,
                    step=1,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Int.Input(
                    "poll_interval",
                    default=5,
                    min=1,
                    max=100,
                    step=1,
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
    def _tensor_to_base64(cls, image):
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
    def _upload_image(cls, upload_url, api_key, base64_data, timeout):
        request = {
            "headers": {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            "body": {"base64Data": base64_data, "uploadPath": "images"},
        }
        response = requests.post(upload_url, headers=request["headers"], json=request["body"], timeout=timeout)
        if response.status_code != 200:
            raise RuntimeError(f"图片上传失败: {response.status_code} - {response.text[:300]}")
        try:
            data = response.json()
        except json.JSONDecodeError:
            raise RuntimeError(f"上传接口返回无效 JSON: {response.text[:200]}")
        file_url = data.get("data", {}).get("fileUrl")
        if not file_url:
            raise RuntimeError(f"上传响应中未找到 fileUrl: {response.text[:300]}")
        return file_url

    @classmethod
    def _parse_result_url(cls, result_json):
        """解析 resultJson（JSON 字符串），返回 resultUrls[0]"""
        if not result_json:
            return None
        try:
            data = json.loads(result_json) if isinstance(result_json, str) else result_json
        except json.JSONDecodeError:
            return None
        urls = data.get("resultUrls") if isinstance(data, dict) else None
        if isinstance(urls, list) and urls:
            return urls[0]
        return None

    @classmethod
    def execute(
        cls,
        prompt,
        resolution,
        image_size,
        output_format,
        prompt_extend,
        negative_prompt,
        seed,
        nsfw_checker,
        host_type,
        timeout,
        max_retries,
        poll_interval,
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
            print(f"[Kie.ai Qwen3] {full_msg}")

        try:
            config = cls._load_config()
            provider = api_loader.get_provider(cls.FIXED_API_PROVIDER)
            if not provider:
                raise ValueError(f"未找到 API 提供商 {cls.FIXED_API_PROVIDER}")

            api_key = config.get("api_keys", {}).get(cls.API_KEY_SOURCE, "")
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
                raise ValueError("请至少提供 1 张输入图片")

            if not prompt or not prompt.strip():
                raise ValueError("请输入编辑指令 prompt")

            # 上传本地图片，获取 URL
            upload_host = cls._get_upload_host(provider, host_type)
            upload_url = f"{upload_host}{provider.get_endpoint('upload')}"
            image_urls = []
            for idx, img in images:
                base64_data = cls._tensor_to_base64(img)
                log(f"上传图片 {idx}...", "⬆️", console_only=True)
                file_url = cls._upload_image(upload_url, api_key, base64_data, timeout)
                image_urls.append(file_url)
                log_url = f"{file_url[:60]}..." if len(file_url) > 60 else file_url
                log(f"图片 {idx} 上传成功: {log_url}", "🖼️")

            # 创建任务
            api_host = provider.get_host(host_type).rstrip("/")
            draw_url = f"{api_host}{provider.get_endpoint('draw')}"
            draw_request = provider.build_request(
                "draw",
                api_key=api_key,
                model=cls.FIXED_MODEL,
                image_urls=image_urls,
                prompt=prompt.strip(),
                resolution=resolution,
                image_size=image_size,
                output_format=output_format,
                prompt_extend=prompt_extend == "启用",
                negative_prompt=negative_prompt.strip() if negative_prompt.strip() else None,
                seed=seed if seed >= 0 else None,
                nsfw_checker=nsfw_checker == "启用",
            )

            log(f"使用 API Host: {api_host}", "🌐")
            log(f"使用模型: {cls.FIXED_MODEL}", "🤖")
            log(f"分辨率: {resolution}, 比例: {image_size}, 格式: {output_format}", "📐")
            log(f"输入图片数量: {len(images)}", "🖼️")
            log(f"发送创建任务请求到: {draw_url}", "📤")

            draw_response = requests.post(
                draw_url, headers=draw_request["headers"], json=draw_request["body"], timeout=timeout
            )
            if draw_response.status_code != 200:
                raise RuntimeError(f"创建任务失败: {draw_response.status_code} - {draw_response.text[:500]}")
            try:
                draw_result = draw_response.json()
            except json.JSONDecodeError:
                raise RuntimeError(f"创建任务返回无效 JSON: {draw_response.text[:200]}")
            task_id = provider._get_nested_value(draw_result, provider.response_format["draw"]["task_id_path"])
            if not task_id:
                raise RuntimeError(f"响应中未找到 taskId: {draw_result}")
            log(f"任务 ID: {task_id}", "🆔")
            log(f"开始轮询结果，间隔 {poll_interval} 秒...", "🔄", console_only=True)

            # 轮询任务状态
            result_url = f"{api_host}{provider.get_endpoint('result')}"
            result_request = provider.build_request("result", api_key=api_key, task_id=task_id)
            result_format = provider.response_format["result"]
            success_status = result_format["success_status"]
            failed_status = result_format["failed_status"]
            start_time = time.time()

            for retry in range(max_retries):
                time.sleep(poll_interval)
                log(f"第 {retry + 1}/{max_retries} 次查询任务状态...", "🔍", console_only=True)
                try:
                    result_response = requests.get(
                        result_url,
                        headers=result_request["headers"],
                        params=result_request.get("query_params", {}),
                        timeout=timeout,
                    )
                    if result_response.status_code != 200:
                        log(f"查询失败: {result_response.status_code}", "⚠️", console_only=True)
                        continue
                    result_data = result_response.json()
                    if result_data.get("code") == 200:
                        data = result_data.get("data", {})
                    else:
                        data = result_data
                    status = provider._get_nested_value(data, result_format["status_path"])
                    log(f"任务状态: {status}", "📊")

                    if status == success_status:
                        result_json = provider._get_nested_value(data, result_format["result_json_path"])
                        image_url = cls._parse_result_url(result_json)
                        if not image_url:
                            raise RuntimeError(f"任务成功但未找到 resultUrls 图片: {result_json}")
                        elapsed = time.time() - start_time
                        log(f"轮询次数: {retry + 1} 次", "🔢")
                        log(f"生成耗时: {elapsed:.2f} 秒", "⏱️")
                        log(f"获取到结果图片 URL: {image_url[:80]}...", "🎨")
                        log("正在下载图片...", "⬇️", console_only=True)

                        img_response = requests.get(image_url, timeout=timeout)
                        if img_response.status_code != 200:
                            raise RuntimeError(f"下载图片失败: {img_response.status_code}")
                        result_img = Image.open(io.BytesIO(img_response.content)).convert("RGB")
                        img_width, img_height = result_img.size
                        log(f"图片尺寸: {img_width}x{img_height}", "📏")

                        if save_to_output == "启用":
                            try:
                                output_dir = folder_paths.get_output_directory()
                                banana_dir = os.path.join(output_dir, "banana")
                                os.makedirs(banana_dir, exist_ok=True)
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"kieai_qwen3_{timestamp}.png"
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
                    elif status == failed_status:
                        fail_msg = provider._get_nested_value(data, result_format["fail_msg_path"]) or ""
                        raise RuntimeError(f"任务失败 - 状态: {status}, 原因: {fail_msg}")
                    elif status in cls.RUNNING_STATES:
                        continue
                    else:
                        log(f"未知状态: {status}，继续轮询", "❓", console_only=True)
                except RuntimeError:
                    raise
                except Exception as e:
                    log(f"查询异常: {str(e)}", "⚠️", console_only=True)
                    continue

            error_msg = f"超过最大重试次数 ({max_retries})，任务可能仍在处理中"
            log(error_msg, "⏰")
            raise TimeoutError(error_msg)

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