import math

from comfy_api.latest import io as comfy_io

from ..utils.image_size_utils import AUTO_SIZE, MULTIPLE, floor_to_multiple, ratio_to_string, resolve_ratio, round_to_multiple

# ============================================================
# GPT-Image-2.5 尺寸约束
# ============================================================
# 参考 gpt-image-2.5-sunburst / gpt-image-2.5-flare
# - 总像素：655,360 <= width * height <= 8,294,400
# - 宽高都必须为 16 的倍数
# - 宽高比限制在 1:3 ~ 3:1
# ============================================================
GPT_MIN_PIXELS = 655360
GPT_MAX_PIXELS = 8294400
GPT_MAX_LONG_SHORT_RATIO = 3.0

# 档位 -> 总像素预算（名义 K 档，长边可突破档位名义值，上限为 GPT 最大总像素）
GPT_PIXEL_BUDGETS = {
    "1K": 1_048_576,      # 1024^2
    "2K": 3_686_400,      # 1920^2，用户参考的 2K 档（1:1 时 1920x1920）
    "2.5K": 5_000_000,
    "3K": 6_000_000,
    "3.5K": 7_500_000,
    "4K": 8_294_400,      # 2880^2，GPT-Image-2.5 最大总像素
}


def calculate_gpt_dimensions(source_width, source_height, ratio_label, size_bucket):
    """按 GPT-Image-2.5 约束计算宽高：宽×高 ⩽ 档位预算 且宽高均为 16 的倍数。

    比例由 ratio_label 决定（auto 取输入图比例 / 预设比例 / 任意 "a:b"），
    支持 1:3 ~ 3:1。
    """
    if source_width <= 0 or source_height <= 0:
        raise ValueError("输入图片宽高必须大于 0")
    if size_bucket not in GPT_PIXEL_BUDGETS:
        raise ValueError(f"不支持的尺寸档位: {size_bucket}")

    ratio = resolve_ratio(ratio_label, source_width, source_height)
    if not (1 / GPT_MAX_LONG_SHORT_RATIO <= ratio <= GPT_MAX_LONG_SHORT_RATIO):
        raise ValueError(
            f"宽高比 {ratio_label} 超出范围，需在 1:3 ~ 3:1 之间"
        )

    budget = GPT_PIXEL_BUDGETS[size_bucket]
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

    # 向下取整可能让比例轻微越界（如 1:3 变为 1:3.02），钳制较长边将其回正
    min_ratio = 1 / GPT_MAX_LONG_SHORT_RATIO
    max_ratio = GPT_MAX_LONG_SHORT_RATIO
    if width / height < min_ratio:
        # 太竖向：以宽为基准收紧高，使 width / height = 1/3
        height = floor_to_multiple(width * max_ratio)
    elif width / height > max_ratio:
        # 太横向：以高为基准收紧宽，使 width / height = 3
        width = floor_to_multiple(height * max_ratio)

    # 不放大到超出最大总像素，同时确保满足最小总像素（各预算档位天然满足）
    if width * height > GPT_MAX_PIXELS:
        raise ValueError(f"宽高 {width}x{height} 总像素超出 GPT 上限 {GPT_MAX_PIXELS:,}")
    if width * height < GPT_MIN_PIXELS:
        raise ValueError(f"宽高 {width}x{height} 总像素低于 GPT 下限 {GPT_MIN_PIXELS:,}")

    return width, height


class GPTImageSizeAdapterNode(comfy_io.ComfyNode):
    """根据输入图比例与尺寸档位，计算符合 GPT-Image-2.5 约束的目标尺寸。"""

    RATIO_OPTIONS = ["auto", "16:9", "9:16", "4:3", "3:4", "1:1"]
    SIZE_OPTIONS = list(GPT_PIXEL_BUDGETS.keys())

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="GPTImageSizeAdapter",
            display_name="GPT Image Size Adapter",
            category="GPT",
            inputs=[
                comfy_io.Image.Input("image"),
                comfy_io.Combo.Input("aspect_ratio", options=cls.RATIO_OPTIONS, default="auto"),
                comfy_io.Combo.Input("image_size", options=cls.SIZE_OPTIONS, default="2K"),
            ],
            outputs=[
                comfy_io.Int.Output("width"),
                comfy_io.Int.Output("height"),
                comfy_io.String.Output("size"),
                comfy_io.String.Output("source_ratio"),
            ],
        )

    @classmethod
    def execute(cls, image, aspect_ratio, image_size) -> comfy_io.NodeOutput:
        image_tensor = image[0] if len(image.shape) == 4 else image
        source_height, source_width, _ = image_tensor.shape
        width, height = calculate_gpt_dimensions(
            source_width,
            source_height,
            aspect_ratio,
            image_size,
        )
        return comfy_io.NodeOutput(
            width,
            height,
            f"{width}x{height}",
            ratio_to_string(source_width, source_height),
        )


class GPTAspectRatioNode(comfy_io.ComfyNode):
    """与 Banana 的 Aspect Ratio V2 同语义：宽高填 16x16 并连接图片时，
    按所选档位把原图缩放到符合 GPT-Image-2.5 约束的分辨率。"""

    STANDARD_RATIOS = (
        ("16:9", 16 / 9),
        ("9:16", 9 / 16),
        ("4:3", 4 / 3),
        ("3:4", 3 / 4),
        ("1:1", 1.0),
    )
    SIZE_OPTIONS = ["原始尺寸", *list(GPT_PIXEL_BUDGETS.keys())]

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="GPTAspectRatio",
            display_name="GPT Aspect Ratio",
            category="GPT",
            inputs=[
                comfy_io.Image.Input("image", optional=True),
                comfy_io.Int.Input(
                    "width",
                    default=1024,
                    min=1,
                    max=65535,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Int.Input(
                    "height",
                    default=1024,
                    min=1,
                    max=65535,
                    display_mode=comfy_io.NumberDisplay.number,
                ),
                comfy_io.Combo.Input(
                    "image_size",
                    options=cls.SIZE_OPTIONS,
                    default="原始尺寸",
                ),
            ],
            outputs=[
                comfy_io.String.Output("aspect_ratio"),
                comfy_io.Int.Output("width"),
                comfy_io.Int.Output("height"),
            ],
        )

    @classmethod
    def execute(cls, width, height, image_size="原始尺寸", image=None) -> comfy_io.NodeOutput:
        # 连了图片且选了尺寸档位：一律按档位从原图缩放，保证 16 对齐且符合 GPT 约束
        if image_size != "原始尺寸":
            if image is None:
                raise ValueError("选择尺寸档位时需连接图片输入，以根据图片实际尺寸计算"
                                 f"（档位: {image_size}）或改用宽高填 16x16 的方式")
            image_tensor = image[0] if len(image.shape) == 4 else image
            img_height, img_width = image_tensor.shape[:2]
            width, height = calculate_gpt_dimensions(img_width, img_height, AUTO_SIZE, image_size)
            return comfy_io.NodeOutput(AUTO_SIZE, width, height)

        # 未选档位，但宽高填 16x16：用图片原始尺寸（补 16 对齐）
        if width == 16 and height == 16:
            if image is None:
                raise ValueError("宽高为(16,16)时需连接图片输入，以根据图片实际尺寸计算尺寸")
            image_tensor = image[0] if len(image.shape) == 4 else image
            img_height, img_width = image_tensor.shape[:2]
            width, height = round_to_multiple(img_width), round_to_multiple(img_height)
            return comfy_io.NodeOutput(AUTO_SIZE, width, height)

        if width <= 0 or height <= 0:
            raise ValueError("width 和 height 必须大于 0")

        # 其余路径：按输入宽高分类比例，并强制 16 对齐
        width, height = round_to_multiple(width), round_to_multiple(height)
        ratio = width / height
        best_label = min(
            cls.STANDARD_RATIOS,
            key=lambda item: abs(ratio - item[1]),
        )[0]
        return comfy_io.NodeOutput(best_label, width, height)