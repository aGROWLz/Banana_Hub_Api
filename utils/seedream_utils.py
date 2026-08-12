"""Seedream 图像尺寸工具"""

# size 参数"方式 1"支持的分辨率档位（官方文档：5.0 Pro 为 1K/1.5K/2K，5.0 Lite 为 2K/3K/4K，
# 节点全部提供，不支持的组合由 API 报错）
SEEDREAM_SIZE_PRESETS = ["1K", "1.5K", "2K", "3K", "4K"]

# size 参数"方式 2"（宽高像素值）的官方约束
SEEDREAM_MIN_PIXELS = 3686400  # 2560x1440
SEEDREAM_MAX_PIXELS = 16777216  # 4096x4096
SEEDREAM_MIN_RATIO = 1 / 16
SEEDREAM_MAX_RATIO = 16


def validate_custom_size(width: int, height: int) -> str:
    """校验自定义宽高是否符合官方范围，返回 "WxH" 字符串。

    不满足时抛出 ValueError，由节点向上抛给 ComfyUI 展示。
    """
    if width <= 0 or height <= 0:
        raise ValueError("自定义尺寸的宽高必须为正整数")
    total_pixels = width * height
    if not (SEEDREAM_MIN_PIXELS <= total_pixels <= SEEDREAM_MAX_PIXELS):
        raise ValueError(
            f"总像素 {width}x{height}={total_pixels} 超出范围，"
            f"需在 {SEEDREAM_MIN_PIXELS}~{SEEDREAM_MAX_PIXELS} 之间"
        )
    ratio = width / height
    if not (SEEDREAM_MIN_RATIO <= ratio <= SEEDREAM_MAX_RATIO):
        raise ValueError(
            f"宽高比 {width}:{height} 超出范围，需在 1:16 ~ 16:1 之间"
        )
    return f"{width}x{height}"
