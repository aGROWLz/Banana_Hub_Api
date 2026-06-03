from .nodes import (
    BananaAspectRatioNode,
    BananaImageGenerationNode,
    BananaImageSizeAdapterNode,
    GeminiVisionNode,
    GPTImage2FullNode,
    GPTImage2Node,
    GrokImageNode,
)
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class GrsaiApiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            BananaImageGenerationNode,
            BananaAspectRatioNode,
            BananaImageSizeAdapterNode,
            GeminiVisionNode,
            GPTImage2Node,
            GPTImage2FullNode,
            GrokImageNode,
        ]


async def comfy_entrypoint() -> GrsaiApiExtension:
    return GrsaiApiExtension()
