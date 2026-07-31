from .nodes import (
    BananaAspectRatioNode,
    BananaAspectRatioNodeV2,
    BananaImageGenerationNode,
    BananaImageSizeAdapterNode,
    BananaImageSizeAdapterNodeV2,
    GeminiVisionNode,
    GPTImage2FullNode,
    GPTImage2Node,
    GrokImageNode,
    CTGTestNode,
    WanImageGenerationNode,
    SeedreamImageGenerationNode,
)
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class GrsaiApiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            BananaImageGenerationNode,
            BananaAspectRatioNode,
            BananaAspectRatioNodeV2,
            BananaImageSizeAdapterNode,
            BananaImageSizeAdapterNodeV2,
            GeminiVisionNode,
            GPTImage2Node,
            GPTImage2FullNode,
            GrokImageNode,
            CTGTestNode,
            WanImageGenerationNode,
            SeedreamImageGenerationNode,
        ]


async def comfy_entrypoint() -> GrsaiApiExtension:
    return GrsaiApiExtension()
