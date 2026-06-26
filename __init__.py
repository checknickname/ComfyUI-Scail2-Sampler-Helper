from .nodes import SCAILExtension

WEB_DIRECTORY = "./js"


async def comfy_entrypoint():
    return SCAILExtension()
