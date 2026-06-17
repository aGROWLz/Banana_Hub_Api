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
from ..utils import APILoader


class BananaImageGenerationNode(comfy_io.ComfyNode):
    """
    Banana Image Generation API 节点
    支持多个第三方 API 提供商
    """

    # 初始化 API 加载器
    api_loader = None

    @classmethod
    def _init_api_loader(cls):
        """初始化 API 加载器"""
        if cls.api_loader is None:
            api_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api")
            cls.api_loader = APILoader(api_dir)

    @classmethod
    def _load_config(cls):
        """加载用户配置文件"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
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
        config = cls._load_config()
        
        # 获取所有提供商
        provider_ids = cls.api_loader.get_provider_ids()
        if not provider_ids:
            provider_ids = ["grsai_api"]
        
        # 收集所有提供商的参数，并标注提供商名称
        models_map = {}  # {model_value: [provider_names]}
        image_sizes_map = {}
        aspect_ratios_map = {}
        
        for provider_id in provider_ids:
            provider = cls.api_loader.get_provider(provider_id)
            if provider:
                provider_name = provider.name
                
                for model in provider.models:
                    if model not in models_map:
                        models_map[model] = []
                    models_map[model].append(provider_name)
                
                for size in provider.image_sizes:
                    if size not in image_sizes_map:
                        image_sizes_map[size] = []
                    image_sizes_map[size].append(provider_name)
                
                for ratio in provider.aspect_ratios:
                    if ratio not in aspect_ratios_map:
                        aspect_ratios_map[ratio] = []
                    aspect_ratios_map[ratio].append(provider_name)
        
        # 构建显示选项（带提供商标注）
        def build_options(value_map, show_providers=True, exclude_from_annotation=None, custom_sort=None):
            """构建选项列表，格式：value (provider1, provider2)"""
            if exclude_from_annotation is None:
                exclude_from_annotation = []
            
            options = []
            # 使用自定义排序或默认排序
            if custom_sort:
                sorted_items = custom_sort(value_map.items())
            else:
                sorted_items = sorted(value_map.items())
            
            for value, providers in sorted_items:
                if show_providers and value not in exclude_from_annotation:
                    providers_str = ", ".join(providers)
                    display_text = f"{value} ({providers_str})"
                else:
                    display_text = value
                options.append(display_text)
            return options
        
        # aspect_ratio 自定义排序：auto 放最前面，其他按字母排序
        def sort_aspect_ratios(items):
            items_list = list(items)
            auto_items = [item for item in items_list if item[0] == "auto"]
            other_items = sorted([item for item in items_list if item[0] != "auto"])
            return auto_items + other_items
        
        # nano-banana-2 和 nano-banana-pro 不显示提供商标注，其他模型显示
        exclude_models = ["nano-banana-2", "nano-banana-pro"]
        models_options = build_options(models_map, show_providers=True, exclude_from_annotation=exclude_models) if models_map else ["nano-banana-2"]
        image_sizes_options = build_options(image_sizes_map, show_providers=False) if image_sizes_map else ["1K"]
        aspect_ratios_options = build_options(aspect_ratios_map, show_providers=False, custom_sort=sort_aspect_ratios) if aspect_ratios_map else ["auto"]
        
        return comfy_io.Schema(
            node_id="BananaImageGeneration",
            display_name="Banana Image Generation API",
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
                    "api_provider",
                    options=provider_ids,
                    default=provider_ids[0]
                ),
                comfy_io.Combo.Input(
                    "host_type",
                    options=["china", "overseas", "custom"],
                    default="china"
                ),
                comfy_io.Combo.Input(
                    "model",
                    options=models_options,
                    default="nano-banana-2" if "nano-banana-2" in [m.split(" (")[0] for m in models_options] else (models_options[0] if models_options else "nano-banana-2")
                ),
                comfy_io.String.Input(
                    "aspect_ratio",
                    default="auto" if "auto" in aspect_ratios_options else (aspect_ratios_options[0] if aspect_ratios_options else "auto")
                ),
                comfy_io.String.Input(
                    "image_size",
                    default=image_sizes_options[0] if image_sizes_options else "1K"
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
    def execute(cls, api_provider, host_type, prompt, model, aspect_ratio, image_size, seed, timeout, max_retries, poll_interval, save_to_output, 
                image1=None, image2=None, image3=None, image4=None, image5=None) -> comfy_io.NodeOutput:
        
        cls._init_api_loader()
        log_messages = []
        
        def log(msg, icon="", console_only=False):
            """添加日志并立即打印到控制台"""
            full_msg = f"{icon} {msg}" if icon else msg
            if not console_only:
                log_messages.append(full_msg)
            print(f"[Banana API] {full_msg}")
        
        def extract_value(param_with_providers):
            """从 'value (provider1, provider2)' 格式中提取实际值"""
            if '(' in param_with_providers:
                return param_with_providers.split('(')[0].strip()
            return param_with_providers
        
        # 提取实际参数值
        model = extract_value(model)
        aspect_ratio = extract_value(aspect_ratio)
        image_size = extract_value(image_size)
        
        # 当 aspect_ratio 为 "auto" 时，设为 None，这样请求中就不会包含该字段
        if aspect_ratio == "auto":
            aspect_ratio = None
        
        try:
            # 加载配置和 API 提供商
            config = cls._load_config()
            provider = cls.api_loader.get_provider(api_provider)
            
            if not provider:
                error_msg = f"未找到 API 提供商: {api_provider}"
                log(error_msg, "❌")
                raise ValueError(error_msg)
            
            log(f"使用 API 提供商: {provider.name}", "🔌")
            
            # 确定使用的 API host
            api_host = provider.get_host(host_type)
            
            # 从配置文件获取 API key
            api_keys = config.get("api_keys", {})
            api_key = api_keys.get(api_provider, "")
            
            if not api_key:
                error_msg = f"错误: 未设置 API Key，请在配置文件的 api_keys.{api_provider} 中设置"
                log(error_msg, "❌")
                raise ValueError(error_msg)
            
            api_host = api_host.rstrip('/')
            
            log(f"使用 API Host: {api_host}", "🌐")
            log(f"使用模型: {model}", "🤖")
            log(f"图片尺寸: {image_size}, 宽高比: {aspect_ratio}", "📐")
            log(f"随机种子: {seed}", "🎲")
            
            # 收集所有输入的图片
            input_images = []
            for idx, img in enumerate([image1, image2, image3, image4, image5], 1):
                if img is not None:
                    input_images.append((idx, img))
            
            # 转换图片为 base64 或文件对象
            urls_list = []
            image_files = []  # 用于 multipart/form-data
            image_buffers = []  # 保持 BytesIO 对象的引用
            
            if input_images:
                log(f"输入图片数量: {len(input_images)}", "🖼️")
                for img_idx, img_tensor in input_images:
                    if len(img_tensor.shape) == 4:
                        img_tensor = img_tensor[0]
                    
                    height, width, channels = img_tensor.shape
                    img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_np)
                    
                    # 为 base64 创建一个 BytesIO
                    buffered_b64 = io.BytesIO()
                    pil_img.save(buffered_b64, format="PNG")
                    img_base64 = base64.b64encode(buffered_b64.getvalue()).decode('utf-8')
                    urls_list.append(f"data:image/png;base64,{img_base64}")
                    
                    # 为 multipart 创建另一个 BytesIO
                    buffered_file = io.BytesIO()
                    pil_img.save(buffered_file, format="PNG")
                    buffered_file.seek(0)
                    image_buffers.append(buffered_file)  # 保持引用
                    image_files.append(('image', (f'image_{img_idx}.png', buffered_file, 'image/png')))
                    
                    log(f"图片 {img_idx}: {width}x{height}, 已编码", "📸")
            else:
                log("未输入参考图片，仅使用 prompt 生成", "💭")
            
            # 构建请求
            draw_endpoint = provider.get_endpoint("draw")
            result_endpoint = provider.get_endpoint("result")
            
            # 映射模型名称（如果提供商有模型映射）
            mapped_model = provider.map_model(model)
            if mapped_model != model:
                log(f"模型映射: {model} -> {mapped_model}", "🔄", console_only=True)
            
            # 替换 URL 中的模型占位符（Gemini 原生格式）
            draw_url = f"{api_host}{draw_endpoint}".replace("{model}", mapped_model)
            result_url = f"{api_host}{result_endpoint}" if result_endpoint else ""
            
            log(f"发送绘画请求到: {draw_url}", "📤")
            log(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}", "✍️")
            
            # 检查 API 格式类型
            is_chat_format = "/chat/completions" in draw_endpoint
            is_gemini_format = ":generateContent" in draw_endpoint
            
            if is_gemini_format:
                # 构建 Gemini 原生格式的 contents
                contents = [
                    {
                        "role": "user",
                        "parts": []
                    }
                ]
                
                # 添加图片（使用 inlineData 格式，纯 base64 不带前缀）
                for idx, url in enumerate(urls_list):
                    # 移除 data:image/png;base64, 前缀
                    if url.startswith("data:image/png;base64,"):
                        base64_data = url.replace("data:image/png;base64,", "")
                    else:
                        base64_data = url
                    
                    contents[0]["parts"].append({
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": base64_data
                        }
                    })
                
                # 添加文本内容
                contents[0]["parts"].append({
                    "text": prompt
                })
                
                log(f"构建 Gemini 原生格式，{len(urls_list)} 张图片 + 文本", "🔷", console_only=True)
                
                # 使用 API 提供商构建请求
                draw_request = provider.build_request(
                    "draw",
                    api_key=api_key,
                    model=mapped_model,
                    contents=contents,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size
                )
            elif is_chat_format:
                # 构建 Chat Completions 格式的 messages
                messages = [
                    {
                        "role": "user",
                        "content": []
                    }
                ]
                
                # 添加文本内容
                messages[0]["content"].append({
                    "type": "text",
                    "text": prompt
                })
                
                # 添加图片（使用 base64 格式）
                for url in urls_list:
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": url
                        }
                    })
                
                log(f"构建 Chat Completions 消息，文本 + {len(urls_list)} 张图片", "💬", console_only=True)
                
                # 使用 API 提供商构建请求
                draw_request = provider.build_request(
                    "draw",
                    api_key=api_key,
                    model=mapped_model,
                    messages=messages,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size
                )
            else:
                # 使用 API 提供商构建请求（普通格式）
                draw_request = provider.build_request(
                    "draw",
                    api_key=api_key,
                    model=mapped_model,
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    urls=urls_list,
                    image_files=image_files
                )
            
            # 检查请求类型
            content_type = draw_request.get("content_type", "application/json")
            
            # 调试：打印请求信息
            if content_type == "multipart/form-data":
                debug_body = {k: v for k, v in draw_request["body"].items() if k != "image"}
                if "image" in draw_request["body"]:
                    debug_body["image"] = f"[{len(image_files)} files]"
                log(f"请求体 (multipart): {json.dumps(debug_body, ensure_ascii=False)}", "🔍", console_only=True)
            else:
                debug_body = draw_request["body"].copy()
                if "urls" in debug_body and debug_body["urls"]:
                    debug_body["urls"] = f"[{len(debug_body['urls'])} images]"
                if "images" in debug_body and debug_body["images"]:
                    debug_body["images"] = f"[{len(debug_body['images'])} images]"
                # 截断 contents 中的 base64 数据
                if "contents" in debug_body and isinstance(debug_body["contents"], list):
                    debug_contents = []
                    for content in debug_body["contents"]:
                        if isinstance(content, dict) and "parts" in content:
                            debug_parts = []
                            for part in content["parts"]:
                                if isinstance(part, dict) and "inlineData" in part:
                                    data_str = part["inlineData"].get("data", "")
                                    debug_parts.append({"inlineData": {"data": f"{data_str[:10]}...", "mimeType": part["inlineData"].get("mimeType", "")}})
                                else:
                                    debug_parts.append(part)
                            debug_contents.append({"parts": debug_parts})
                        else:
                            debug_contents.append(content)
                    debug_body["contents"] = debug_contents
                log(f"请求体 (json): {json.dumps(debug_body, ensure_ascii=False)}", "🔍", console_only=True)
            
            log("正在发送请求...", "⏳", console_only=True)
            
            # 根据 content_type 发送不同格式的请求
            try:
                if content_type == "multipart/form-data":
                    # multipart/form-data 请求
                    files = []
                    data = {}
                    
                    for key, value in draw_request["body"].items():
                        if key == "image" and image_files:
                            # 添加图片文件
                            files.extend(image_files)
                        elif value:  # 只添加非空值
                            data[key] = value
                    
                    log(f"发送 multipart 请求，文件数: {len(files)}, 数据字段: {list(data.keys())}", "📋", console_only=True)
                    
                    response = requests.post(
                        draw_url,
                        headers=draw_request["headers"],
                        files=files,
                        data=data,
                        timeout=timeout
                    )
                else:
                    # JSON 请求
                    response = requests.post(
                        draw_url,
                        headers=draw_request["headers"],
                        json=draw_request["body"],
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
            
            # 尝试解析 JSON，如果失败则显示原始响应
            try:
                if not response.text or response.text.strip() == "":
                    error_msg = f"API 返回空响应（可能是欠费或服务异常）"
                    log(error_msg, "❌")
                    log(f"HTTP 状态码: {response.status_code}", "ℹ️")
                    log(f"响应头: {dict(response.headers)}", "ℹ️")
                    raise RuntimeError(error_msg)
                result = response.json()
            except json.JSONDecodeError as e:
                error_msg = f"API 返回的不是有效的 JSON 格式"
                log(error_msg, "❌")
                log(f"HTTP 状态码: {response.status_code}", "ℹ️")
                log(f"响应内容: {response.text[:500]}", "ℹ️")
                raise RuntimeError(f"{error_msg}\n响应内容: {response.text[:200]}")
            
            # 截断响应中的 b64_json 数据用于日志显示
            debug_result = json.dumps(result, ensure_ascii=False)
            import re
            debug_result = re.sub(r'"b64_json"\s*:\s*"[^"]{10}[^"]*"', lambda m: m.group()[:m.group().index('"b64_json"') + len('"b64_json"') + 13] + '..."', debug_result)
            log(f"绘画请求响应: {debug_result[:500]}...", "📥")
            
            # 检查是否是同步返回结果（如 bltai, 147ai）
            draw_response_format = provider.response_format.get("draw", {})
            
            # Gemini 原生格式（返回 inline data）
            if "inline_data_path" in draw_response_format:
                log("检测到 Gemini 原生格式，直接处理结果", "🔷")
                
                parts_result = provider._get_nested_value(result, draw_response_format["inline_data_path"])
                
                if parts_result:
                    image_data = None
                    text_response = ""
                    
                    for part in parts_result:
                        if "text" in part:
                            text_response += part["text"]
                        elif "inlineData" in part:
                            image_data = part["inlineData"]
                    
                    if text_response:
                        log(f"生成文本: {text_response}", "💬")
                    
                    if image_data:
                        log("获取到生成的图片", "🎨")
                        
                        try:
                            img_bytes = base64.b64decode(image_data["data"])
                            result_img = Image.open(io.BytesIO(img_bytes))
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
                                    filename = f"banana_{timestamp}_{provider.name.replace(' ', '_')}.png"
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
                            error_msg = f"解码图片失败: {str(decode_error)}"
                            log(error_msg, "❌")
                            raise RuntimeError(error_msg)
                    else:
                        error_msg = "响应中未找到图片数据"
                        log(error_msg, "❌")
                        raise RuntimeError(error_msg)
                else:
                    error_msg = "Gemini 响应中未找到 parts 数据"
                    log(error_msg, "❌")
                    raise RuntimeError(error_msg)
            
            # Gemini 原生格式（147AI）- 返回 base64 图片数据
            elif "image_data_path" in draw_response_format:
                log("检测到 Gemini 原生格式（147AI），直接处理结果", "🔷")
                
                image_data = provider._get_nested_value(result, draw_response_format["image_data_path"])
                
                if image_data:
                    log("获取到 base64 编码的图片数据", "🎨")
                    
                    try:
                        # 解码 base64 图片
                        img_bytes = base64.b64decode(image_data)
                        result_img = Image.open(io.BytesIO(img_bytes))
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
                                filename = f"banana_{timestamp}_{provider.name.replace(' ', '_')}.png"
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
                else:
                    error_msg = "Gemini 响应中未找到图片数据"
                    log(error_msg, "❌")
                    raise RuntimeError(error_msg)
            
            # Chat Completions 格式（旧版 147AI）
            elif "content_path" in draw_response_format:
                log("检测到 Chat Completions 格式，直接处理结果", "⚡")
                
                content = provider._get_nested_value(result, draw_response_format["content_path"])
                
                if content:
                    # content 可能是图片 URL 或包含图片 URL 的文本
                    image_url = content.strip()
                    
                    # 截断 base64 URL 用于日志显示
                    log_url = f"{image_url[:10]}..." if image_url.startswith("data:") and len(image_url) > 10 else image_url
                    log(f"获取到结果图片 URL: {log_url}", "🎨")
                    log("正在下载图片...", "⬇️", console_only=True)

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
                                filename = f"banana_{timestamp}_{provider.name.replace(' ', '_')}.png"
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
                        error_msg = f"下载图片失败: {img_response.status_code}"
                        log(error_msg, "❌")
                        raise RuntimeError(error_msg)
                else:
                    error_msg = "Chat Completions 响应中未找到内容"
                    log(error_msg, "❌")
                    raise RuntimeError(error_msg)

            # 其他同步 API（bltai）
            elif "image_url_path" in draw_response_format or "b64_json_path" in draw_response_format:
                # 同步 API，直接从响应中获取图片
                log("检测到同步 API，直接处理结果", "⚡")

                image_url = None
                b64_json = None

                if "image_url_path" in draw_response_format:
                    image_url = provider._get_nested_value(result, draw_response_format["image_url_path"])

                if "b64_json_path" in draw_response_format:
                    b64_json = provider._get_nested_value(result, draw_response_format["b64_json_path"])

                if image_url:
                    # 截断 base64 URL 用于日志显示
                    log_url = f"{image_url[:10]}..." if image_url.startswith("data:") and len(image_url) > 10 else image_url
                    log(f"获取到结果图片 URL: {log_url}", "🎨")
                    log("正在下载图片...", "⬇️", console_only=True)
                    
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
                                filename = f"banana_{timestamp}_{provider.name.replace(' ', '_')}.png"
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
                        error_msg = f"下载图片失败: {img_response.status_code}"
                        log(error_msg, "❌")
                        raise RuntimeError(error_msg)
                
                elif b64_json:
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
                                filename = f"banana_{timestamp}_{provider.name.replace(' ', '_')}.png"
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
                else:
                    error_msg = "同步 API 响应中未找到图片数据"
                    log(error_msg, "❌")
                    raise RuntimeError(error_msg)
            
            # 异步 API（需要轮询），解析响应获取任务 ID
            if result.get("code") == 0:
                task_id = result.get("data", {}).get("id")
            else:
                task_id = result.get("id")
            
            if not task_id:
                error_msg = f"未获取到任务 ID，响应: {result}"
                log(error_msg, "❌")
                raise RuntimeError(error_msg)
            
            log(f"任务 ID: {task_id}", "🆔")
            log(f"开始轮询结果，间隔 {poll_interval} 秒...", "🔄", console_only=True)
            
            # 轮询获取结果
            start_time = time.time()
            poll_count = 0
            last_progress = -1
            
            result_request = provider.build_request("result", api_key=api_key, task_id=task_id)
            result_method = result_request.get("method", "POST")
            
            for retry in range(max_retries):
                time.sleep(poll_interval)
                poll_count += 1
                
                log(f"第 {poll_count}/{max_retries} 次查询结果...", "🔍", console_only=True)
                
                try:
                    # 根据请求方法选择 GET 或 POST
                    if result_method == "GET":
                        result_response = requests.get(
                            result_url,
                            headers=result_request["headers"],
                            params=result_request.get("query_params", {}),
                            timeout=timeout
                        )
                    else:
                        result_response = requests.post(
                            result_url,
                            headers=result_request["headers"],
                            json=result_request["body"],
                            timeout=timeout
                        )
                    
                    if result_response.status_code != 200:
                        log(f"查询失败: {result_response.status_code}", "⚠️", console_only=True)
                        continue
                    
                    result_data = result_response.json()
                    
                    # 从响应中提取实际值
                    if result_data.get("code") == 0:
                        data = result_data.get("data", {})
                    else:
                        data = result_data
                    
                    # 获取状态和进度
                    status = provider._get_nested_value(data, provider.response_format["result"]["status_path"])
                    progress = provider._get_nested_value(data, provider.response_format["result"]["progress_path"]) or 0
                    
                    # 获取配置中的状态常量
                    success_status = provider.response_format["result"]["success_status"]
                    failed_status = provider.response_format["result"]["failed_status"]
                    running_status = provider.response_format["result"]["running_status"]
                    violation_status = provider.response_format["result"].get("violation_status")
                    
                    # 控制台显示所有状态变化
                    if last_progress != progress or status != running_status:
                        if status == running_status:
                            log(f"任务状态: {status}, 进度: {progress}%", "⏳", console_only=True)
                        else:
                            log(f"任务状态: {status}, 进度: {progress}%", "📊")
                        last_progress = progress
                    else:
                        log(f"任务状态: {status}, 进度: {progress}%", "⏳", console_only=True)
                    
                    if status == success_status:
                        elapsed_time = time.time() - start_time
                        
                        # 提取图片 URL 和内容
                        image_url = provider._get_nested_value(data, provider.response_format["result"]["image_url_path"])
                        content = provider._get_nested_value(data, provider.response_format["result"].get("content_path", "")) or ""
                        
                        if content:
                            log(f"生成内容: {content}", "💬")
                        
                        if image_url:
                            # 截断 base64 URL 用于日志显示
                            log_url = f"{image_url[:10]}..." if image_url.startswith("data:") and len(image_url) > 10 else image_url
                            log(f"获取到结果图片: {log_url}", "🎨")
                            log(f"轮询次数: {poll_count} 次", "🔢")
                            log(f"生成耗时: {elapsed_time:.2f} 秒", "⏱️")
                            log("正在下载图片...", "⬇️", console_only=True)
                            
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
                                        task_id_short = task_id.split("-")[-1][:8] if "-" in task_id else task_id[:8]
                                        filename = f"banana_{timestamp}_{task_id_short}.png"
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
                                error_msg = f"下载图片失败: {img_response.status_code}"
                                log(error_msg, "❌")
                                raise RuntimeError(error_msg)
                        else:
                            error_msg = "结果中未找到图片 URL"
                            log(error_msg, "❌")
                            raise RuntimeError(error_msg)
                    
                    elif status == failed_status or (violation_status and status == violation_status):
                        failure_reason = provider._get_nested_value(data, provider.response_format["result"].get("failure_reason_path", "")) or ""
                        error_detail = provider._get_nested_value(data, provider.response_format["result"].get("error_path", "")) or ""
                        error_msg = f"任务失败 - 状态: {status}, 原因: {failure_reason}, 详情: {error_detail}"
                        log(error_msg, "❌")
                        raise RuntimeError(error_msg)
                    
                    elif status == running_status:
                        continue
                    else:
                        log(f"未知状态: {status}", "❓", console_only=True)
                        continue
                
                except RuntimeError:
                    raise
                except Exception as e:
                    log(f"查询异常: {str(e)}", "⚠️", console_only=True)
                    continue
            
            error_msg = f"超过最大重试次数 ({max_retries})，任务可能仍在处理中"
            log(error_msg, "⏰")
            raise TimeoutError(error_msg)
            
        except Exception as e:
            error_msg = f"发生错误: {str(e)}"
            log(error_msg, "❌")
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
