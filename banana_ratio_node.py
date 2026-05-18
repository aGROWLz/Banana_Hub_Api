from comfy_api.latest import io as comfy_io


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
