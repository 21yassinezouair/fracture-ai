"""
Construit un modele a partir de son nom (via timm), pre-entraine sur ImageNet.
Sortie : 1 neurone (classification binaire fracture / non-fracture).
"""
import timm

def build_model(name="resnet50", pretrained=True):
    model = timm.create_model(name, pretrained=pretrained, num_classes=1)
    return model
