import timm


def create_backbone(name: str, pretrained: bool = True):
    return timm.create_model(name, pretrained=pretrained, num_classes=0)

