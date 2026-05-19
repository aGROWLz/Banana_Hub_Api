from .banana_node import BananaImageGenerationNode
from .banana_ratio_node import BananaAspectRatioNode, BananaImageSizeAdapterNode
from .gemini_node import GeminiVisionNode
from .gpt_image2_node import GPTImage2FullNode, GPTImage2Node
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
        ]


async def comfy_entrypoint() -> GrsaiApiExtension:
    return GrsaiApiExtension()
