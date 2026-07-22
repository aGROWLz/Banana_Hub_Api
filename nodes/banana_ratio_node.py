from comfy_api.latest import io as comfy_io

from ..utils import AUTO_SIZE, calculate_bucket_dimensions, ratio_to_string


class BananaAspectRatioNode(comfy_io.ComfyNode):
    STANDARD_RATIOS = (
        ("16:9", 16 / 9),
        ("9:16", 9 / 16),
        ("4:3", 4 / 3),
        ("3:4", 3 / 4),
        ("5:4", 5 / 4),
        ("4:5", 4 / 5),
        ("3:2", 3 / 2),
        ("2:3", 2 / 3),
        ("1:1", 1.0),
    )

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="BananaAspectRatio",
            display_name="Banana Aspect Ratio",
            category="Banana",
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
                    options=["原始尺寸", "1K", "2K", "4K"],
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
        # 当宽高都是16时，使用图片实际尺寸来计算比例
        if width == 16 and height == 16:
            if image is None:
                raise ValueError("宽高为(16,16)时需连接图片输入，以根据图片实际尺寸计算比例")
            image_tensor = image[0] if len(image.shape) == 4 else image
            img_height, img_width = image_tensor.shape[:2]

            # 根据 image_size 选项缩放尺寸
            if image_size != "原始尺寸":
                width, height = calculate_bucket_dimensions(img_width, img_height, "auto", image_size)
            else:
                width, height = img_width, img_height

        if width <= 0 or height <= 0:
            raise ValueError("width 和 height 必须大于 0")

        ratio = width / height
        best_label = min(
            cls.STANDARD_RATIOS,
            key=lambda item: abs(ratio - item[1]),
        )[0]
        return comfy_io.NodeOutput(best_label, width, height)


class BananaImageSizeAdapterNode(comfy_io.ComfyNode):
    RATIO_OPTIONS = [AUTO_SIZE, "16:9", "9:16", "4:3", "3:4", "5:4", "4:5", "3:2", "2:3", "1:1"]
    SIZE_OPTIONS = ["1K", "2K", "3K", "4K"]

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="BananaImageSizeAdapter",
            display_name="Banana Image Size Adapter",
            category="Banana",
            inputs=[
                comfy_io.Image.Input("image"),
                comfy_io.Combo.Input("aspect_ratio", options=cls.RATIO_OPTIONS, default=AUTO_SIZE),
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
        width, height = calculate_bucket_dimensions(
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
