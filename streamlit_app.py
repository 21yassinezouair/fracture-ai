"""
Fracture AI - Demonstration Streamlit.
Version autonome : aucune dependance a OpenCV ni albumentations
(Grad-CAM reimplemente en PyTorch pur pour un deploiement fiable).
"""
from pathlib import Path
import sys
import numpy as np
import streamlit as st
import torch
import matplotlib.cm as cm
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from src.models.build import build_model

MODELE = "resnet50"
SEUIL = 0.5
TAILLE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

st.set_page_config(page_title="Fracture AI", page_icon="🦴", layout="centered")


@st.cache_resource
def charger_modele():
    model = build_model(MODELE, pretrained=False)
    poids = ROOT / "checkpoints" / f"best_{MODELE}.pt"
    model.load_state_dict(torch.load(poids, map_location="cpu"))
    model.eval()
    return model


def pretraiter(image_pil):
    """Redimensionne + normalise (equivalent de get_val_transforms, sans cv2)."""
    img = image_pil.convert("RGB").resize((TAILLE, TAILLE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr_norm = (arr - MEAN) / STD
    tensor = torch.from_numpy(arr_norm).permute(2, 0, 1).unsqueeze(0)
    return tensor, arr  # tensor pour le modele, arr pour l'affichage


def gradcam(model, tensor):
    """Grad-CAM implemente a la main : hooks sur la derniere couche conv."""
    activations, gradients = {}, {}

    def hook_avant(module, entree, sortie):
        activations["v"] = sortie.detach()

    def hook_arriere(module, grad_entree, grad_sortie):
        gradients["v"] = grad_sortie[0].detach()

    couche = model.layer4[-1]
    h1 = couche.register_forward_hook(hook_avant)
    h2 = couche.register_full_backward_hook(hook_arriere)

    sortie = model(tensor)
    model.zero_grad()
    sortie[0, 0].backward()

    h1.remove()
    h2.remove()

    # Ponderation des cartes d'activation par les gradients moyens
    poids = gradients["v"].mean(dim=(2, 3), keepdim=True)
    carte = torch.relu((poids * activations["v"]).sum(dim=1)).squeeze(0)
    carte = carte / (carte.max() + 1e-8)

    # Redimensionner la carte (7x7) vers 224x224 avec PIL
    carte_img = Image.fromarray((carte.numpy() * 255).astype(np.uint8))
    carte_img = carte_img.resize((TAILLE, TAILLE), Image.BILINEAR)
    return np.asarray(carte_img, dtype=np.float32) / 255.0, torch.sigmoid(sortie).item()


def superposer(image_arr, carte):
    """Superpose la carte de chaleur (colormap jet) sur l'image."""
    couleurs = cm.jet(carte)[:, :, :3]
    melange = 0.5 * couleurs + 0.5 * image_arr
    return (melange.clip(0, 1) * 255).astype(np.uint8)


# ---------------- Interface ----------------
st.title("🦴 Detection automatisee de fractures osseuses")
st.warning("**Outil de recherche uniquement — ne remplace en aucun cas un avis medical.**")
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

fichier = st.file_uploader(
    "Choisissez une radiographie osseuse", type=["jpg", "jpeg", "png"]
)

if fichier is not None:
    image_pil = Image.open(fichier)
    with st.spinner("Analyse en cours..."):
        model = charger_modele()
        tensor, image_arr = pretraiter(image_pil)
        carte, proba = gradcam(model, tensor)
        heatmap = superposer(image_arr, carte)

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
