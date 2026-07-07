import requests
import torch
import numpy as np
from PIL import Image
import io
import base64
import json
import time
from comfy_api.latest import io as comfy_io


class CTGTestNode(comfy_io.ComfyNode):
    """
    CTG API 测试节点
    用于测试 Chat Completions 格式的图片生成 API
    """

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="CTGTestNode",
            display_name="CTG Image Test API",
            category="Banana/Test",
            inputs=[
                comfy_io.Image.Input("image1", optional=True),
                comfy_io.Image.Input("image2", optional=True),
                comfy_io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                ),
                comfy_io.String.Input(
                    "api_key",
                    default="",
                ),
                comfy_io.String.Input(
                    "base_url",
                    default="https://tokenhub.ctclouds.com",
                ),
                comfy_io.String.Input(
                    "model",
                    default="ctg-gg-image-flash",
                ),
                # 请求参数开关
                comfy_io.Combo.Input(
                    "enable_aspect_ratio",
                    options=["关闭", "启用"],
                    default="关闭",
                ),
                comfy_io.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"],
                    default="auto",
                ),
                comfy_io.Combo.Input(
                    "enable_image_size",
                    options=["关闭", "启用"],
                    default="关闭",
                ),
                comfy_io.Combo.Input(
                    "image_size",
                    options=["1K", "2K", "4K"],
                    default="1K",
                ),
                comfy_io.Combo.Input(
                    "enable_n",
                    options=["关闭", "启用"],
                    default="关闭",
                ),
                comfy_io.Int.Input(
                    "n",
                    default=1,
                    min=1,
                    max=4,
                ),
                comfy_io.Combo.Input(
                    "enable_quality",
                    options=["关闭", "启用"],
                    default="关闭",
                ),
                comfy_io.Combo.Input(
                    "quality",
                    options=["standard", "hd"],
                    default="standard",
                ),
                comfy_io.Int.Input(
                    "timeout",
                    default=120,
                    min=10,
                    max=600,
                    step=10,
                ),
            ],
            outputs=[
                comfy_io.Image.Output(),
                comfy_io.String.Output(display_name="log"),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
        api_key: str,
        base_url: str,
        model: str,
        enable_aspect_ratio: str,
        aspect_ratio: str,
        enable_image_size: str,
        image_size: str,
        enable_n: str,
        n: int,
        enable_quality: str,
        quality: str,
        timeout: int,
        image1=None,
        image2=None,
    ):
        log_messages = []

        def log(msg, prefix=""):
            line = f"{prefix} {msg}".strip()
            print(f"[CTG Test] {line}")
            log_messages.append(line)

        # 验证必填参数
        if not api_key:
            raise ValueError("API Key 不能为空")
        if not prompt and image1 is None and image2 is None:
            raise ValueError("Prompt 和图片至少需要提供一个")

        # 构建 messages
        messages = []
        user_content = []

        # 添加图片
        input_images = []
        if image1 is not None:
            input_images.append(image1)
        if image2 is not None:
            input_images.append(image2)

        for img_tensor in input_images:
            if len(img_tensor.shape) == 4:
                img_tensor = img_tensor[0]

            # 转换为 base64
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })
            log(f"已编码图片: {pil_img.size[0]}x{pil_img.size[1]}", "📸")

        # 添加文本
        if prompt:
            user_content.append({
                "type": "text",
                "text": prompt
            })

        messages.append({
            "role": "user",
            "content": user_content if len(user_content) > 1 else user_content[0]["text"] if user_content else ""
        })

        # 构建请求体
        request_body = {
            "model": model,
            "stream": False,
            "messages": messages,
        }

        # 根据开关添加可选参数
        if enable_aspect_ratio == "启用":
            request_body["aspectRatio"] = aspect_ratio
            log(f"已添加 aspectRatio 参数: {aspect_ratio}", "⚙️")

        if enable_image_size == "启用":
            request_body["imageSize"] = image_size
            log(f"已添加 imageSize 参数: {image_size}", "⚙️")

        if enable_n == "启用":
            request_body["n"] = n
            log(f"已添加 n 参数: {n}", "⚙️")

        if enable_quality == "启用":
            request_body["quality"] = quality
            log(f"已添加 quality 参数: {quality}", "⚙️")

        # 构建 URL
        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        log(f"请求 URL: {url}", "📤")
        log(f"模型: {model}", "🤖")

        # 构建 headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        # 打印请求体（截断 base64）
        debug_body = json.dumps(request_body, ensure_ascii=False)
        if len(debug_body) > 500:
            debug_body = debug_body[:500] + "..."
        log(f"请求体: {debug_body}", "🔍")

        # 发送请求
        start_time = time.time()
        log("正在发送请求...", "⏳")

        try:
            response = requests.post(
                url,
                headers=headers,
                json=request_body,
                timeout=timeout,
            )

            elapsed = time.time() - start_time
            log(f"请求耗时: {elapsed:.2f} 秒", "⏱️")
            log(f"HTTP 状态码: {response.status_code}", "📥")

            if response.status_code != 200:
                error_msg = f"API 请求失败: {response.status_code}"
                log(error_msg, "❌")
                log(f"响应内容: {response.text[:500]}", "ℹ️")
                raise RuntimeError(f"{error_msg}\n{response.text[:200]}")

            # 检查响应是否为空
            if not response.text or response.text.strip() == "":
                error_msg = "API 返回空响应"
                log(error_msg, "❌")
                raise RuntimeError(error_msg)

            result = response.json()

            # 打印响应摘要
            log(f"响应 keys: {list(result.keys())}", "📥")

            # 尝试提取图片
            # Chat Completions 格式通常在 choices[0].message.content 中
            choices = result.get("choices", [])
            if not choices:
                error_msg = f"响应中没有 choices: {json.dumps(result, ensure_ascii=False)[:200]}"
                log(error_msg, "❌")
                raise RuntimeError(error_msg)

            message = choices[0].get("message", {})
            content = message.get("content", "")

            log(f"响应内容类型: {type(content)}", "🔍")

            # 如果 content 是字符串，尝试解析为 JSON（可能是 base64 图片）
            if isinstance(content, str):
                # 尝试解析为 JSON
                try:
                    content_data = json.loads(content)
                    if isinstance(content_data, dict) and "image" in content_data:
                        # 可能是 base64 图片
                        img_data = content_data["image"]
                        if img_data.startswith("data:image"):
                            # 去掉前缀
                            img_data = img_data.split(",")[1]
                        img_bytes = base64.b64decode(img_data)
                        result_img = Image.open(io.BytesIO(img_bytes))
                        result_img = result_img.convert("RGB")
                        log(f"成功解析图片: {result_img.size[0]}x{result_img.size[1]}", "🎨")
                    else:
                        log(f"响应内容: {str(content)[:200]}", "💬")
                        raise RuntimeError(f"无法识别的响应格式: {str(content)[:200]}")
                except json.JSONDecodeError:
                    # 不是 JSON，可能是纯文本错误信息
                    log(f"响应内容: {content[:200]}", "💬")
                    raise RuntimeError(f"API 返回非图片内容: {content[:200]}")
            elif isinstance(content, list):
                # 可能是多模态响应
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "image_url":
                            img_url = item.get("image_url", {}).get("url", "")
                            if img_url.startswith("data:image"):
                                img_data = img_url.split(",")[1]
                                img_bytes = base64.b64decode(img_data)
                                result_img = Image.open(io.BytesIO(img_bytes))
                                result_img = result_img.convert("RGB")
                                log(f"成功解析图片: {result_img.size[0]}x{result_img.size[1]}", "🎨")
                                break
                else:
                    raise RuntimeError(f"响应中未找到图片")
            else:
                raise RuntimeError(f"未知的响应格式: {type(content)}")

            # 转换为 tensor
            img_array = np.array(result_img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array)[None,]

            log("处理完成", "✅")
            log_text = "\n".join(log_messages)

            return comfy_io.NodeOutput(img_tensor, log_text)

        except requests.exceptions.Timeout:
            error_msg = f"请求超时 ({timeout}秒)"
            log(error_msg, "⏰")
            raise TimeoutError(error_msg)
        except requests.exceptions.ConnectionError as e:
            error_msg = f"连接失败: {str(e)}"
            log(error_msg, "❌")
            raise ConnectionError(error_msg)
        except RuntimeError:
            raise
        except Exception as e:
            error_msg = f"发生错误: {str(e)}"
            log(error_msg, "❌")
            raise
