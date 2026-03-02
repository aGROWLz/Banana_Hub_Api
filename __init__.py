from .nano_banana_node import NanoBananaNode
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class GrsaiApiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            NanoBananaNode,
        ]


async def comfy_entrypoint() -> GrsaiApiExtension:
    return GrsaiApiExtension()
