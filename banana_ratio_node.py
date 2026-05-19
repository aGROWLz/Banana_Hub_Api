from comfy_api.latest import io as comfy_io

from .image_size_utils import AUTO_SIZE, calculate_bucket_dimensions, ratio_to_string


class BananaAspectRatioNode(comfy_io.ComfyNode):
    STANDARD_RATIOS = (
        ("16:9", 16 / 9),
        ("9:16", 9 / 16),
        ("4:3", 4 / 3),
        ("3:4", 3 / 4),
        ("1:1", 1.0),
    )

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="BananaAspectRatio",
            display_name="Banana Aspect Ratio",
            category="Banana",
            inputs=[
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
            ],
            outputs=[comfy_io.String.Output("aspect_ratio")],
        )

    @classmethod
    def execute(cls, width, height) -> comfy_io.NodeOutput:
        if width <= 0 or height <= 0:
            raise ValueError("width 和 height 必须大于 0")

        ratio = width / height
        best_label = min(
            cls.STANDARD_RATIOS,
            key=lambda item: abs(ratio - item[1]),
        )[0]
        return comfy_io.NodeOutput(best_label)


class BananaImageSizeAdapterNode(comfy_io.ComfyNode):
    RATIO_OPTIONS = [AUTO_SIZE, "16:9", "9:16", "4:3", "3:4", "1:1"]
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
