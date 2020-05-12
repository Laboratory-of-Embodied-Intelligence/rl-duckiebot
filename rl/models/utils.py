import tensorflow as tf
from stable_baselines.ddpg.policies import FeedForwardPolicy as DDPGPolicy
from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
from stable_baselines.common.policies import register_policy

class CustomDDPGPolicy(DDPGPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDDPGPolicy, self).__init__(*args, **kwargs,
                                               layers=[32, 8],
                                               feature_extraction="mlp",
                                               layer_norm=True)

class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                              layers=[32, 16],
                                              act_fun=tf.nn.elu,
                                              feature_extraction="mlp")

register_policy('CustomDDPGPolicy', CustomDDPGPolicy)
register_policy('CustomSACPolicy', CustomSACPolicy)

def make_image(numpy_img):
    from PIL import Image
    height, width, channel = numpy_img.shape
    image = Image.fromarray(numpy_img)
    # import io
    # output = io.BytesIO()
    # image.save(output, format='PNG')
    # image_string = output.getvalue()
    # output.close()
    return image
