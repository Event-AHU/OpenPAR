from models.registry import BACKBONE
from models.registry import CLASSIFIER
from models.registry import LOSSES


def build_backbone(key, multi_scale=False):

    model_dict = {
        'resnet34': 512,
        'resnet18': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'tresnet': 2432,
        'swin_s': 768,
        'swin_b': 1024,
        'vit_s': 768,
        'vit_b': 768,
        'vit_ce_b': 768,
        'bninception': 1024,
        'tresnetM': 2048,
        'tresnetL': 2048,

    }

    model = BACKBONE[key]()
    output_d = model_dict[key]

    return model, output_d


def build_classifier(key):

    return CLASSIFIER[key]


def build_loss(key):

    return LOSSES[key]

