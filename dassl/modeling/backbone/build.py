from dassl.utils import Registry, check_availability

BACKBONE_REGISTRY = Registry("BACKBONE")


def build_backbone(name, cfg, verbose=True, **kwargs):
    avai_backbones = BACKBONE_REGISTRY.registered_names()
    check_availability(name, avai_backbones)
    if verbose:
        print("Backbone: {}".format(name))
    kwargs["cfg"] = cfg
    return BACKBONE_REGISTRY.get(name)(**kwargs)
