__version__ = "0.2.3"

import os

if 'TDS_RESOURCE_PATH' in os.environ:
    _resource_path = os.environ['TDS_RESOURCE_PATH'].split(':')
else:
    _resource_path = []
_resource_path += [os.path.join(x, 'resources/maps') for x in __path__]


def assert_pytorch3d_available():
    from torchdrivesim.rendering.pytorch3d import is_available as pytorch3d_available
    if not pytorch3d_available:
        from torchdrivesim.rendering.pytorch3d import Pytorch3DNotFound
        raise Pytorch3DNotFound()
