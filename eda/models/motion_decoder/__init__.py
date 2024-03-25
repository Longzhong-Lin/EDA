"""
Based on https://github.com/sshaoshuai/MTR/blob/master/mtr/models/motion_decoder/__init__.py
"""


from MTR.mtr.models.motion_decoder.mtr_decoder import MTRDecoder
from .eda_decoder import EDADecoder


__all__ = {
    'MTRDecoder': MTRDecoder,
    'EDADecoder': EDADecoder
}


def build_motion_decoder(in_channels, config):
    model = __all__[config.NAME](
        in_channels=in_channels,
        config=config
    )

    return model
