import math
from math import gcd

MULTIPLE = 16
MIN_EDGE = 16
MAX_EDGE = 3840
MIN_PIXELS = 655360
MAX_PIXELS = 8294400
AUTO_SIZE = "auto"

RATIO_PRESETS = {
    "16:9": (16, 9),
    "9:16": (9, 16),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "1:1": (1, 1),
}

SIZE_BUCKETS = {
    "1K": 1024,
    "2K": 2048,
    "2.5K": 2560,
    "3K": 3072,
    "3.5K": 3584,
    "4K": 3840,
}


def round_to_multiple(value, multiple=MULTIPLE):
    return max(multiple, int(round(value / multiple)) * multiple)


def floor_to_multiple(value, multiple=MULTIPLE):
    return max(multiple, int(math.floor(value / multiple)) * multiple)


def simplify_ratio(width, height):
    if width <= 0 or height <= 0:
        raise ValueError("width 和 height 必须大于 0")
    factor = gcd(width, height)
    return width // factor, height // factor


def ratio_to_string(width, height):
    numerator, denominator = simplify_ratio(width, height)
    return f"{numerator}:{denominator}"


def validate_ratio(width, height):
    long_edge = max(width, height)
    short_edge = min(width, height)
    if short_edge <= 0:
        raise ValueError("width 和 height 必须大于 0")
    if long_edge / short_edge > 3:
        raise ValueError("长边 / 短边 比值不能超过 3:1")


def validate_size_dimensions(width, height):
    if width == 0 and height == 0:
        return AUTO_SIZE
    if width <= 0 or height <= 0:
        raise ValueError("width 和 height 必须同时为正整数，或同时为 0 表示 auto")
    if width > MAX_EDGE or height > MAX_EDGE:
        raise ValueError(f"图片最大边长不能超过 {MAX_EDGE}px")
    if width % MULTIPLE != 0 or height % MULTIPLE != 0:
        raise ValueError(f"宽高必须都是 {MULTIPLE}px 的倍数")

    validate_ratio(width, height)

    total_pixels = width * height
    if total_pixels < MIN_PIXELS or total_pixels > MAX_PIXELS:
        raise ValueError(f"总像素必须在 {MIN_PIXELS} 到 {MAX_PIXELS} 之间")
    return f"{width}x{height}"


def validate_custom_dimensions(width, height):
    """自定义尺寸校验：仅限制长宽 ≤3840px 且为 16 的倍数。"""
    if width <= 0 or height <= 0:
        raise ValueError("width 和 height 必须为正整数")
    if width > MAX_EDGE or height > MAX_EDGE:
        raise ValueError(f"图片最大边长不能超过 {MAX_EDGE}px")
    if width % MULTIPLE != 0 or height % MULTIPLE != 0:
        raise ValueError(f"宽高必须都是 {MULTIPLE}px 的倍数")
    return f"{width}x{height}"


def resolve_ratio(label, source_width, source_height):
    if label == AUTO_SIZE:
        validate_ratio(source_width, source_height)
        return source_width / source_height
    if label in RATIO_PRESETS:
        width, height = RATIO_PRESETS[label]
        return width / height
    # 支持任意 "a:b" 比例（如 "7:2"、"5:3"），供非标准比例场景使用
    if isinstance(label, str) and ':' in label:
        try:
            w_str, h_str = label.split(':', 1)
            w, h = float(w_str), float(h_str)
        except (TypeError, ValueError):
            raise ValueError(f"不支持的宽高比: {label}")
        if w <= 0 or h <= 0:
            raise ValueError(f"宽高比数值必须大于 0: {label}")
        return w / h
    raise ValueError(f"不支持的宽高比: {label}")


def _build_dimensions_from_long_edge(long_edge, ratio):
    long_edge = round_to_multiple(long_edge)
    if ratio >= 1:
        width = long_edge
        height = round_to_multiple(width / ratio)
    else:
        height = long_edge
        width = round_to_multiple(height * ratio)
    return width, height


def calculate_bucket_dimensions(source_width, source_height, ratio_label, size_bucket):
    if source_width <= 0 or source_height <= 0:
        raise ValueError("输入图片宽高必须大于 0")
    if size_bucket not in SIZE_BUCKETS:
        raise ValueError(f"不支持的尺寸档位: {size_bucket}")

    ratio = resolve_ratio(ratio_label, source_width, source_height)
    target_long_edge = SIZE_BUCKETS[size_bucket]
    width, height = _build_dimensions_from_long_edge(target_long_edge, ratio)

    while (
        width * height < MIN_PIXELS
        and max(width, height) < MAX_EDGE
    ):
        target_long_edge += MULTIPLE
        width, height = _build_dimensions_from_long_edge(target_long_edge, ratio)

    while (
        (width > MAX_EDGE or height > MAX_EDGE or width * height > MAX_PIXELS)
        and target_long_edge > MIN_EDGE
    ):
        target_long_edge -= MULTIPLE
        width, height = _build_dimensions_from_long_edge(target_long_edge, ratio)

    validate_size_dimensions(width, height)
    return width, height


# 档位 → 总像素上限（按名义分辨率 K² 计算，长边可突破档位名义值）
PIXEL_BUDGETS = {
    "1K": 1024 * 1024,
    "2K": 2048 * 2048,
    "2.5K": 2560 * 2560,
    "3K": 3072 * 3072,
    "3.5K": 3584 * 3584,
    "4K": 4096 * 4096,
}

# 宽高比允许的极端范围（长边 / 短边 不超过 8）
MAX_LONG_SHORT_RATIO = 8


def calculate_dimensions_by_pixel_budget(source_width, source_height, ratio_label, size_bucket):
    """按总像素上限计算宽高：宽×高 ≤ 档位 K²，长边可突破档位名义值。

    比例由 ratio_label 决定（auto 取输入图比例 / 预设比例 / 任意 "a:b"），
    支持 1:8 ~ 8:1 的极端比例。
    """
    if source_width <= 0 or source_height <= 0:
        raise ValueError("输入图片宽高必须大于 0")
    if size_bucket not in PIXEL_BUDGETS:
        raise ValueError(f"不支持的尺寸档位: {size_bucket}")

    ratio = resolve_ratio(ratio_label, source_width, source_height)
    if not (1 / MAX_LONG_SHORT_RATIO <= ratio <= MAX_LONG_SHORT_RATIO):
        raise ValueError(
            f"宽高比 {ratio_label} 超出范围，需在 1:{MAX_LONG_SHORT_RATIO} ~ {MAX_LONG_SHORT_RATIO}:1 之间"
        )

    budget = PIXEL_BUDGETS[size_bucket]
    # 理想宽高：宽×高 = budget 且 宽/高 = ratio
    width = math.sqrt(budget * ratio)
    height = math.sqrt(budget / ratio)

    width = floor_to_multiple(width)
    height = floor_to_multiple(height)

    # 向下取整后极端情况下仍可能超预算，递减较长边
    while width * height > budget and max(width, height) > MULTIPLE:
        if width >= height:
            width -= MULTIPLE
        else:
            height -= MULTIPLE

    return width, height
