# Fracture AI — Detection automatisee de fractures osseuses par apprentissage profond

**Demonstration en ligne :** https://fracture-ai-cl4nv82uamhmp3ledstamm.streamlit.app/

Travail de these de doctorat — Universite Mohammed VI des Sciences de la Sante (UM6SS),
Laboratoire L3S, Casablanca, Maroc.

- **Doctorant :** Yassine Zouair
- **Directrice de these :** Pr. Nawal Bouknani
- **Reference protocole ethique :** CE-UM6SS-2026-FRACT-IA-001

---

## Objectif

Developper et valider des modeles d'apprentissage profond pour la detection binaire
de fractures osseuses sur radiographies standard.

Ce depot contient le pipeline complet developpe et valide sur donnees publiques
(**FracAtlas**, 4 083 radiographies), en amont de la phase clinique de la these.

---

## Resultats

### Comparaison des architectures (jeu de test, n = 613)

| Modele | AUC (IC 95%) | Sensibilite | Specificite | VPP | VPN | F1 |
|---|---|---|---|---|---|---|
| ConvNeXt-Tiny | **0,907** (0,869–0,941) | 0,685 | 0,964 | 0,804 | 0,935 | 0,740 |
| ResNet-50 | 0,881 (0,839–0,918) | 0,778 | 0,842 | 0,512 | 0,947 | 0,618 |
| ViT-B/16 | 0,859 (0,818–0,896) | 0,769 | 0,786 | 0,435 | 0,941 | 0,555 |
| EfficientNet-B4 | 0,846 (0,803–0,887) | 0,722 | 0,768 | 0,400 | 0,928 | 0,515 |

### Comparaison statistique (test de DeLong)

| Comparaison | p | Conclusion |
|---|---|---|
| ConvNeXt vs ResNet-50 | 0,076 | Non significatif — performances equivalentes |
| ConvNeXt vs ViT-B/16 | 0,0004 | Significatif |
| ConvNeXt vs EfficientNet-B4 | 0,0005 | Significatif |

### Validation croisee 5-fold (ResNet-50)

**AUC = 0,907 ± 0,011** (min 0,893 — max 0,925)

### Interpretabilite

Analyse Grad-CAM : les activations se concentrent sur des regions osseuses et non sur
des artefacts d'image ou des marqueurs, suggerant que le modele a appris des
caracteristiques anatomiquement pertinentes.

---

## Limites

- Donnees publiques uniquement (FracAtlas ne fournit pas d'identifiant patient :
  le decoupage train/validation/test est donc realise au niveau de l'image).
- Analyse par localisation anatomique non concluante : effectifs insuffisants pour
  la hanche (5 fractures) et l'epaule (10 fractures) dans le jeu de test.
- Classification binaire uniquement (pas de localisation par boite englobante).

---

## Structure du depot
## Reproduction

```bash
conda create -n fracture-ai python=3.11 -y
conda activate fracture-ai
pip install -r requirements.txt

python scripts/prepare_data.py      # index + splits (seed = 42)
python scripts/train.py --model resnet50 --epochs 15
python scripts/evaluate.py --model resnet50
```

Toutes les experiences utilisent un **seed fixe (42)** pour garantir la reproductibilite.

---

## Avertissement

Ce systeme est un **outil de recherche**. Il ne constitue pas un dispositif medical
et ne doit en aucun cas se substituer a un avis medical qualifie.

## Licence

MIT
