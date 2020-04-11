from stable_baselines.ddpg.policies import FeedForwardPolicy as DDPGPolicy
from stable_baselines.common.policies import register_policy

class CustomDDPGPolicy(DDPGPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDDPGPolicy, self).__init__(*args, **kwargs,
                                               layers=[32, 8],
                                               feature_extraction="mlp",
                                               layer_norm=True)

register_policy('CustomDDPGPolicy', CustomDDPGPolicy)

