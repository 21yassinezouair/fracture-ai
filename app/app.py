"""
Demonstration - Detection de fractures osseuses par deep learning.
Interface Gradio : upload d'une radiographie -> probabilite + carte Grad-CAM.

OUTIL DE RECHERCHE UNIQUEMENT - ne remplace pas un avis medical.
"""
from pathlib import Path
import sys
import numpy as np
import torch
import gradio as gr
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.data.transforms import get_val_transforms
from src.models.build import build_model

MODELE = "resnet50"
SEUIL = 0.5
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

# --- Charger le modele une seule fois au demarrage ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(MODELE, pretrained=False).to(device)
poids = ROOT / "checkpoints" / f"best_{MODELE}.pt"
model.load_state_dict(torch.load(poids, map_location=device))
model.eval()
transforms = get_val_transforms()
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

def analyser(image):
    """Recoit une image numpy (RGB), renvoie (libelle, dict probas, heatmap)."""
    if image is None:
        return "Aucune image", {}, None

    # Pretraitement identique a l'entrainement
    tensor = transforms(image=image)["image"].unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        proba = torch.sigmoid(model(tensor)).item()

    # Grad-CAM
    grayscale = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(0)])[0]
    rgb = tensor[0].cpu().permute(1, 2, 0).numpy()
    rgb = (rgb * STD + MEAN).clip(0, 1).astype(np.float32)
    heatmap = show_cam_on_image(rgb, grayscale, use_rgb=True)

    verdict = "FRACTURE DETECTEE" if proba >= SEUIL else "Pas de fracture detectee"
    probas = {"Fracture": proba, "Normal": 1 - proba}
    return verdict, probas, heatmap

AVERTISSEMENT = """
# Detection automatisee de fractures osseuses

**Outil de recherche uniquement - ne remplace en aucun cas un avis medical.**

Modele : ResNet-50 entraine sur le jeu de donnees public FracAtlas (4 083 radiographies).
Performance : AUC = 0,907 +/- 0,011 (validation croisee 5-fold).

Travail de these - Universite Mohammed VI des Sciences de la Sante (UM6SS), Laboratoire L3S.
Les zones colorees (Grad-CAM) indiquent les regions ayant le plus influence la decision.
"""

demo = gr.Interface(
    fn=analyser,
    inputs=gr.Image(type="numpy", label="Radiographie osseuse"),
    outputs=[
        gr.Textbox(label="Resultat"),
        gr.Label(label="Probabilites"),
        gr.Image(label="Grad-CAM - zones determinantes"),
    ],
    title="Fracture AI - Demonstration",
    description=AVERTISSEMENT,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(share=True)
