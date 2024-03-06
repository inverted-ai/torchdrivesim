__version__ = "0.1.0"


def assert_pytorch3d_available():
    from torchdrivesim.rendering.pytorch3d import is_available as pytorch3d_available
    if not pytorch3d_available:
        from torchdrivesim.rendering.pytorch3d import Pytorch3DNotFound
        raise Pytorch3DNotFound()
