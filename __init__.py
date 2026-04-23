from .banana_node import BananaImageGenerationNode
from .gemini_node import GeminiVisionNode
from .gpt_image2_node import GPTImage2ReverseNode, GPTImage2OfficialNode
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class GrsaiApiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            BananaImageGenerationNode,
            GeminiVisionNode,
            GPTImage2ReverseNode,
            GPTImage2OfficialNode,
        ]


async def comfy_entrypoint() -> GrsaiApiExtension:
    return GrsaiApiExtension()
