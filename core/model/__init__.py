from .ships_detector import ShipsDetector

_MODEL_META_ARCHITECTURES = {
    "ShipsDetector": ShipsDetector,
}


def build_model(type, num_classes):
    meta_arch = _MODEL_META_ARCHITECTURES[type]
    return meta_arch(num_classes)