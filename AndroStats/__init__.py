import pkg_resources


def get_resource(path: str) -> str:
    return pkg_resources.resource_filename(__name__, path)
