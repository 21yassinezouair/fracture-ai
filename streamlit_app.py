"""
Fracture AI - Demonstration Streamlit.
Upload d'une radiographie -> probabilite de fracture + carte Grad-CAM.

OUTIL DE RECHERCHE UNIQUEMENT - ne remplace pas un avis medical.
"""
from pathlib import Path
import sys
import numpy as np
import streamlit as st
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from src.data.transforms import get_val_transforms
from src.models.build import build_model

MODELE = "resnet50"
SEUIL = 0.5
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

st.set_page_config(page_title="Fracture AI", page_icon="🦴", layout="centered")

@st.cache_resource
def charger_modele():
    """Charge le modele une seule fois (mis en cache par Streamlit)."""
    model = build_model(MODELE, pretrained=False)
    poids = ROOT / "checkpoints" / f"best_{MODELE}.pt"
    model.load_state_dict(torch.load(poids, map_location="cpu"))
    model.eval()
    return model

# --- En-tete ---
st.title("🦴 Detection automatisee de fractures osseuses")
st.warning(
    "**Outil de recherche uniquement — ne remplace en aucun cas un avis medical.**"
)
st.markdown(
    """
    Modele **ResNet-50** entraine sur le jeu de donnees public **FracAtlas**
    (4 083 radiographies).
    Performance : **AUC = 0,907 ± 0,011** (validation croisee 5-fold).

    *Travail de these — Universite Mohammed VI des Sciences de la Sante (UM6SS),
    Laboratoire L3S.*
    """
)
st.divider()

# --- Upload ---
fichier = st.file_uploader(
    "Choisissez une radiographie osseuse",
    type=["jpg", "jpeg", "png"],
)

if fichier is not None:
    image_pil = Image.open(fichier).convert("RGB")
    image = np.array(image_pil)

    with st.spinner("Analyse en cours..."):
        model = charger_modele()
        transforms = get_val_transforms()
        tensor = transforms(image=image)["image"].unsqueeze(0)

        # Prediction
        with torch.no_grad():
            proba = torch.sigmoid(model(tensor)).item()

        # Grad-CAM
        cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
        grayscale = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(0)])[0]
        rgb = tensor[0].permute(1, 2, 0).numpy()
        rgb = (rgb * STD + MEAN).clip(0, 1).astype(np.float32)
        heatmap = show_cam_on_image(rgb, grayscale, use_rgb=True)

    # --- Resultat ---
    if proba >= SEUIL:
        st.error(f"### FRACTURE DETECTEE — probabilite {proba:.1%}")
    else:
        st.success(f"### Pas de fracture detectee — probabilite {proba:.1%}")

    st.progress(proba)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image originale")
        st.image(image_pil, use_container_width=True)
    with col2:
        st.subheader("Grad-CAM")
        st.image(heatmap, use_container_width=True)

    st.caption(
        "Les zones chaudes (rouge/jaune) indiquent les regions ayant le plus "
        "influence la decision du modele."
    )
else:
    st.info("Chargez une radiographie pour lancer l'analyse.")
