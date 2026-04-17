from .banana_node import BananaImageGenerationNode
from .gemini_node import GeminiVisionNode
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class GrsaiApiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            BananaImageGenerationNode,
            GeminiVisionNode,
        ]


async def comfy_entrypoint() -> GrsaiApiExtension:
    return GrsaiApiExtension()
