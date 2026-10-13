from .nodes import (
    BananaAspectRatioNode,
    BananaAspectRatioNodeV2,
    BananaImageGenerationNode,
    BananaCustomUrlGenerationNode,
    BananaImageSizeAdapterNode,
    BananaImageSizeAdapterNodeV2,
    GeminiVisionNode,
    GeminiVisionNodeV2,
    GPTImage2FullNode,
    GPTAspectRatioNode,
    GPTImageSizeAdapterNode,
    GrokImageNode,
    CTGTestNode,
    WanImageGenerationNode,
    SeedreamImageGenerationNode,
    SeedreamImageGenerationNodeV2,
    QwenImageEditNode,
    KieAiQwen3ImageNode,
    ProductDetailAgentNode,
    ProductDetailBatchNode,
    ProductDetailImageAggregatorNode,
)
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class GrsaiApiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            BananaImageGenerationNode,
            BananaCustomUrlGenerationNode,
            BananaAspectRatioNode,
            BananaAspectRatioNodeV2,
            BananaImageSizeAdapterNode,
            BananaImageSizeAdapterNodeV2,
            GeminiVisionNode,
            GeminiVisionNodeV2,
            GPTImage2FullNode,
            GPTAspectRatioNode,
            GPTImageSizeAdapterNode,
            GrokImageNode,
            CTGTestNode,
            WanImageGenerationNode,
            SeedreamImageGenerationNode,
            SeedreamImageGenerationNodeV2,
            QwenImageEditNode,
            KieAiQwen3ImageNode,
            ProductDetailAgentNode,
            ProductDetailBatchNode,
            ProductDetailImageAggregatorNode,
        ]


async def comfy_entrypoint() -> GrsaiApiExtension:
    return GrsaiApiExtension()
