
from vae.controller import VAEController

def load_vae(path=None, z_size=None):
    """
    :param path: (str)
    :param z_size: (int)
    :return: (VAEController)
    """
    # z_size will be recovered from saved model
    if z_size is None:
        assert path is not None

    vae = VAEController(z_size=z_size)
    if path is not None:
        vae.load(path)
    print("Dim VAE = {}".format(vae.z_size))
    return vae