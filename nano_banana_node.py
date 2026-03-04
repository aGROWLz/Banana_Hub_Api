import requests
import torch
import numpy as np
from PIL import Image
import io
import base64
import time
import json
import os
from datetime import datetime
from comfy_api.latest import io as comfy_io
import folder_paths


class NanoBananaNode(comfy_io.ComfyNode):
    """
    Grsai Nano Banana API 节点
    支持 5 个独立图片输入，调用 Nano Banana 绘画 API
    """

    @classmethod
    def _load_config(cls):
        """加载配置文件"""
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        default_config = {
            "api_host": "https://grsai.dakka.com.cn",
            "api_key": "",
            "host_overseas": "https://grsaiapi.com",
            "host_china": "https://grsai.dakka.com.cn"
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
        config = cls._load_config()
        
        return comfy_io.Schema(
            node_id="GrsaiNanoBanana",
            display_name="Grsai Nano Banana",
            category="Grsai",
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
                    options=[
                        "nano-banana-2",
                        "nano-banana-fast",
                        "nano-banana",
                        "nano-banana-pro",
                        "nano-banana-pro-vt",
                        "nano-banana-pro-cl",
                        "nano-banana-pro-vip",
                        "nano-banana-pro-4k-vip"
                    ],
                    default="nano-banana-fast"
                ),
                comfy_io.Combo.Input(
                    "aspect_ratio",
                    options=["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"],
                    default="auto"
                ),
                comfy_io.Combo.Input(
                    "image_size",
                    options=["1K", "2K", "4K"],
                    default="1K"
                ),
                comfy_io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xffffffffffffffff,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Combo.Input(
                    "api_host_preset",
                    options=["使用配置文件", "海外 Host", "国内 Host", "自定义"],
                    default="使用配置文件"
                ),
                comfy_io.String.Input(
                    "custom_host",
                    default="",
                    multiline=False,
                ),
                comfy_io.String.Input(
                    "api_key_override",
                    default="",
                    multiline=False,
                ),
                comfy_io.Int.Input(
                    "timeout",
                    default=180,
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
                    default="启用"
                ),
            ],
            outputs=[
                comfy_io.Image.Output("result_image"),
                comfy_io.String.Output("log"),
            ],
        )

    @classmethod
    def execute(cls, prompt, model, aspect_ratio, image_size, seed, api_host_preset, custom_host, api_key_override, timeout, max_retries, poll_interval, save_to_output, image1=None, image1_desc="", image2=None, image2_desc="", image3=None, image3_desc="", image4=None, image4_desc="", image5=None, image5_desc="") -> comfy_io.NodeOutput:
        log_messages = []  # 用于输出日志口
        
        def log(msg, icon="", console_only=False):
            """添加日志并立即打印到控制台
            
            Args:
                msg: 日志消息
                icon: emoji 图标
                console_only: 如果为 True，只打印到控制台，不添加到输出日志
            """
            full_msg = f"{icon} {msg}" if icon else msg
            if not console_only:
                log_messages.append(full_msg)
            print(f"[Grsai Nano Banana] {full_msg}")
        
        try:
            # 加载配置
            config = cls._load_config()
            
            # 确定使用的 API host
            if api_host_preset == "使用配置文件":
                api_host = config.get("api_host", "https://grsai.dakka.com.cn")
            elif api_host_preset == "海外 Host":
                api_host = config.get("host_overseas", "https://grsaiapi.com")
            elif api_host_preset == "国内 Host":
                api_host = config.get("host_china", "https://grsai.dakka.com.cn")
            elif api_host_preset == "自定义":
                api_host = custom_host if custom_host else config.get("api_host", "https://grsai.dakka.com.cn")
            else:
                api_host = config.get("api_host", "https://grsai.dakka.com.cn")
            
            # 确定使用的 API key
            api_key = api_key_override if api_key_override else config.get("api_key", "")
            
            if not api_key:
                error_msg = "错误: 未设置 API Key，请在配置文件或节点参数中设置"
                log(error_msg, "❌")
                raise ValueError(error_msg)
            
            # 移除 API host 末尾的斜杠
            api_host = api_host.rstrip('/')
            
            log(f"使用 API Host: {api_host}", "🌐")
            log(f"使用模型: {model}", "🤖")
            log(f"图片尺寸: {image_size}, 宽高比: {aspect_ratio}", "📐")
            log(f"随机种子: {seed}", "🎲")
            
            # 收集所有输入的图片
            input_images = []
            image_descs = [image1_desc, image2_desc, image3_desc, image4_desc, image5_desc]
            for idx, (img, desc) in enumerate(zip([image1, image2, image3, image4, image5], image_descs), 1):
                if img is not None:
                    input_images.append((idx, img, desc))
            
            # 转换图片为 base64（如果有输入图片）
            urls_list = []
            if input_images:
                log(f"输入图片数量: {len(input_images)}", "🖼️")
                for img_idx, img_tensor, desc in input_images:
                    # ComfyUI 图片格式是 (batch, height, width, channels)
                    # 如果是 batch，只取第一张
                    if len(img_tensor.shape) == 4:
                        img_tensor = img_tensor[0]  # 取第一张图片
                    
                    # 获取图片尺寸信息
                    height, width, channels = img_tensor.shape
                    
                    # 将 tensor 转换为 PIL Image
                    img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_np)
                    
                    # 转换为 base64
                    buffered = io.BytesIO()
                    pil_img.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    urls_list.append(f"data:image/png;base64,{img_base64}")
                    
                    # 输出详细信息
                    desc_info = f" ({desc})" if desc else ""
                    log(f"图片 {img_idx}{desc_info}: {width}x{height}, 已编码", "📸")
            else:
                log("未输入参考图片，仅使用 prompt 生成", "💭")
            
            # 准备请求数据
            draw_url = f"{api_host}/v1/draw/nano-banana"
            result_url = f"{api_host}/v1/draw/result"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # 构建请求体（按照 API 文档格式）
            request_data = {
                "model": model,
                "prompt": prompt,
                "aspectRatio": aspect_ratio,
                "imageSize": image_size,
                "webHook": "-1",  # 使用轮询方式
                "shutProgress": False
            }
            
            # 只有在有图片时才添加 urls 参数
            if urls_list:
                request_data["urls"] = urls_list
            
            log(f"发送绘画请求到: {draw_url}", "📤")
            log(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}", "✍️")
            
            # 发送绘画请求
            log("正在发送请求...", "⏳")
            response = requests.post(
                draw_url,
                headers=headers,
                json=request_data,
                timeout=timeout
            )
            
            if response.status_code != 200:
                error_msg = f"API 请求失败: {response.status_code} - {response.text}"
                log(error_msg, "❌")
                raise RuntimeError(error_msg)
            
            result = response.json()
            log(f"绘画请求响应: {json.dumps(result, ensure_ascii=False)}", "📥")
            
            # 获取任务 ID
            task_id = None
            if result.get("code") == 0:
                task_id = result.get("data", {}).get("id")
            else:
                # 尝试直接从响应中获取 id
                task_id = result.get("id")
            
            if not task_id:
                error_msg = f"未获取到任务 ID，响应: {result}"
                log(error_msg, "❌")
                raise RuntimeError(error_msg)
            
            log(f"任务 ID: {task_id}", "🆔")
            log(f"开始轮询结果，间隔 {poll_interval} 秒...", "🔄", console_only=True)
            
            # 记录开始时间
            start_time = time.time()
            poll_count = 0
            last_progress = -1  # 使用局部变量而不是类属性
            
            # 轮询获取结果
            for retry in range(max_retries):
                time.sleep(poll_interval)
                poll_count += 1
                
                # 打印轮询信息到控制台（不添加到输出日志）
                log(f"第 {poll_count}/{max_retries} 次查询结果...", "🔍", console_only=True)
                
                try:
                    result_response = requests.post(
                        result_url,
                        headers=headers,
                        json={"id": task_id},
                        timeout=timeout
                    )
                    
                    if result_response.status_code != 200:
                        log(f"查询失败: {result_response.status_code} - {result_response.text}", "⚠️", console_only=True)
                        continue
                    
                    result_data = result_response.json()
                    
                    # 检查响应格式
                    if result_data.get("code") == 0:
                        data = result_data.get("data", {})
                    elif result_data.get("code") == -22:
                        log("任务不存在", "❓", console_only=True)
                        continue
                    else:
                        data = result_data
                    
                    status = data.get("status")
                    progress = data.get("progress", 0)
                    
                    # 控制台显示所有状态变化
                    if last_progress != progress or status != "running":
                        if status == "running":
                            log(f"任务状态: {status}, 进度: {progress}%", "⏳", console_only=True)
                        else:
                            # 非 running 状态添加到输出日志
                            log(f"任务状态: {status}, 进度: {progress}%", "📊")
                        last_progress = progress
                    else:
                        # 相同进度也打印到控制台
                        log(f"任务状态: {status}, 进度: {progress}%", "⏳", console_only=True)
                    
                    if status == "succeeded":
                        # 计算总耗时
                        elapsed_time = time.time() - start_time
                        
                        # 获取结果
                        results = data.get("results", [])
                        
                        if results and len(results) > 0:
                            result_item = results[0]
                            image_url = result_item.get("url")
                            content = result_item.get("content", "")
                            
                            if content:
                                log(f"生成内容: {content}", "💬")
                            
                            if image_url:
                                log(f"获取到结果图片: {image_url}", "🎨")
                                log(f"轮询次数: {poll_count} 次", "🔢")
                                log(f"生成耗时: {elapsed_time:.2f} 秒", "⏱️")
                                log("正在下载图片...", "⬇️", console_only=True)
                                
                                # 下载图片
                                img_response = requests.get(image_url, timeout=timeout)
                                if img_response.status_code == 200:
                                    result_img = Image.open(io.BytesIO(img_response.content))
                                    result_img = result_img.convert("RGB")
                                    
                                    # 获取图片信息
                                    img_width, img_height = result_img.size
                                    log(f"图片尺寸: {img_width}x{img_height}", "📏")
                                    
                                    # 保存图片到 output/banana 目录
                                    if save_to_output == "启用":
                                        try:
                                            # 获取 ComfyUI 的 output 目录
                                            output_dir = folder_paths.get_output_directory()
                                            banana_dir = os.path.join(output_dir, "banana")
                                            
                                            # 创建 banana 目录（如果不存在）
                                            os.makedirs(banana_dir, exist_ok=True)
                                            
                                            # 生成文件名：banana_YYYYMMDD_HHMMSS_任务ID后8位.png
                                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            task_id_short = task_id.split("-")[-1][:8] if "-" in task_id else task_id[:8]
                                            filename = f"banana_{timestamp}_{task_id_short}.png"
                                            filepath = os.path.join(banana_dir, filename)
                                            
                                            # 保存图片
                                            result_img.save(filepath, "PNG")
                                            log(f"图片已保存: {filepath}", "💾")
                                            
                                        except Exception as save_error:
                                            log(f"保存图片失败: {str(save_error)}", "⚠️")
                                            # 保存失败不影响返回结果
                                    else:
                                        log("已跳过保存图片（保存功能已禁用）", "ℹ️")
                                    
                                    # 转换为 tensor
                                    img_array = np.array(result_img).astype(np.float32) / 255.0
                                    img_tensor = torch.from_numpy(img_array)[None,]
                                    
                                    log("处理完成", "✅")
                                    log_text = "\n".join(log_messages)
                                    
                                    return comfy_io.NodeOutput(img_tensor, log_text)
                                else:
                                    error_msg = f"下载图片失败: {img_response.status_code}"
                                    log(error_msg, "❌")
                                    raise RuntimeError(error_msg)
                            else:
                                error_msg = "结果中未找到图片 URL"
                                log(error_msg, "❌")
                                raise RuntimeError(error_msg)
                        else:
                            error_msg = "结果为空"
                            log(error_msg, "❌")
                            raise RuntimeError(error_msg)
                    
                    elif status == "failed":
                        failure_reason = data.get("failure_reason", "")
                        error_detail = data.get("error", "")
                        error_msg = f"任务失败 - 原因: {failure_reason}, 详情: {error_detail}"
                        log(error_msg, "❌")
                        
                        if failure_reason == "error":
                            log("提示: 遇到 'error' 时可尝试重新提交任务", "💡")
                        
                        raise RuntimeError(error_msg)
                    
                    elif status == "running":
                        # 继续等待
                        continue
                    else:
                        # 未知状态，记录并继续
                        log(f"未知状态: {status}", "❓")
                        continue
                    
                except RuntimeError:
                    # 重新抛出 RuntimeError（任务失败等）
                    raise
                except Exception as e:
                    log(f"查询异常: {str(e)}", "⚠️", console_only=True)
                    continue
                    log(f"查询异常: {str(e)}", "⚠️")
                    continue
            
            # 超过最大重试次数
            error_msg = f"超过最大重试次数 ({max_retries})，任务可能仍在处理中"
            log(error_msg, "⏰")
            raise TimeoutError(error_msg)
            
        except Exception as e:
            error_msg = f"发生错误: {str(e)}"
            log(error_msg, "❌")
            import traceback
            traceback_str = traceback.format_exc()
            log(traceback_str, "🔍")
            # 重新抛出异常，让 ComfyUI 显示错误
            raise
    
    @classmethod
    def _create_error_output(cls, log_messages, error_msg):
        """已废弃：现在直接抛出异常而不是返回错误图片"""
        pass
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """
        返回 seed 值，让 ComfyUI 知道输入变化了
        当 seed 改变时，强制重新执行节点，避免使用缓存
        """
        return kwargs.get("seed", 0)
