from comfy_api.latest import io as comfy_io

from ..utils.product_detail_utils import aggregate_image_batches


class ProductDetailImageAggregatorNode(comfy_io.ComfyNode):
    """Aggregate up to ten optional image inputs into one IMAGE batch."""

    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="ProductDetailImageAggregator",
            display_name="Product Detail Image Aggregator",
            category="Banana/Product Detail",
            inputs=[
                comfy_io.Image.Input(f"image{index}", optional=True)
                for index in range(1, 11)
            ],
            outputs=[comfy_io.Image.Output("images")],
        )

    @classmethod
    def execute(cls, **kwargs) -> comfy_io.NodeOutput:
        images = aggregate_image_batches([kwargs.get(f"image{index}") for index in range(1, 11)])
        return comfy_io.NodeOutput(images)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        image_hashes = []
        for index in range(1, 11):
            image = kwargs.get(f"image{index}")
            image_hashes.append(hash(image.cpu().numpy().tobytes()) if image is not None else 0)
        return hash(tuple(image_hashes))
