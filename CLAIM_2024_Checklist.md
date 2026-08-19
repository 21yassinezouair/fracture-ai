# CLAIM 2024 Checklist — Completed for This Study

**Study:** A Rigorous and Reproducible Benchmark of CNN and Transformer Architectures for Bone Fracture Detection on FracAtlas

**Reporting guideline:** Checklist for Artificial Intelligence in Medical Imaging (CLAIM), 2024 Update (Tejani, Klontzas, Gatti, Mongan, Moy, Park, Kahn; *Radiology: Artificial Intelligence* 2024;6(4):e240300).

> This checklist documents where each applicable CLAIM item is addressed in the manuscript. Section names refer to the submitted manuscript. Items that do not apply to a retrospective, publicly available dataset benchmark are marked **N/A** with a brief justification.

---

## TITLE AND ABSTRACT

| # | Item | Addressed? | Location / Note |
|---|------|-----------|-----------------|
| 1 | Identification as a study of AI methodology, specifying the category of technology used (e.g., deep learning) | Yes | Title; Abstract (deep learning benchmark) |
| 2 | Structured summary of study design, methods, results, and conclusions | Yes | Abstract (Background / Methods / Results / Conclusions) |

## INTRODUCTION

| # | Item | Addressed? | Location / Note |
|---|------|-----------|-----------------|
| 3 | Scientific and clinical background, including the intended use and clinical role of the AI approach | Yes | Introduction (triage / decision support for fracture detection) |
| 4 | Study objectives and hypotheses | Yes | Introduction (five stated contributions; benchmark objective) |

## METHODS — Study Design

| # | Item | Addressed? | Location / Note |
|---|------|-----------|-----------------|
| 5 | Prospective or retrospective study | Yes | Retrospective, using public datasets (Methods — Dataset) |
| 6 | Study goal (e.g., model creation, evaluation) | Yes | Model evaluation / benchmarking (Introduction; Methods) |

## METHODS — Data

| # | Item | Addressed? | Location / Note |
|---|------|-----------|-----------------|
| 7 | Data sources | Yes | FracAtlas and GRAZPEDWRI-DX, both public and cited (Methods — Dataset; External validation) |
| 8 | Inclusion and exclusion criteria | Yes | All FracAtlas images used; exact duplicates removed (Methods — Duplicate removal) |
| 9 | Data pre-processing steps | Yes | Methods — Preprocessing (resize 224×224, ImageNet normalization, augmentation) |
| 10 | Selection of data subsets, if applicable | Yes | Stratified 70/15/15 split, fixed seed (Methods — Data splitting) |
| 11 | De-identification methods | N/A | Both datasets are publicly released in fully de-identified form by their original providers; no identifiable data were handled by the authors |
| 12 | How missing data were handled | N/A | No missing image-level labels in the binary classification task; all included radiographs carry a fracture/non-fracture label |
| 13 | Definition of ground-truth reference standard | Yes | FracAtlas expert annotations; GRAZPEDWRI-DX `fracture_visible` field (Methods — Dataset; External validation) |
| 14 | Rationale for choosing the reference standard | Yes | Original expert-annotated labels from the source datasets |
| 15 | Source of ground-truth annotations; qualifications of annotators | Yes | As provided by the original dataset publications (cited) |
| 16 | Annotation tools | N/A | Labels taken as provided by source datasets; no new annotation performed |
| 17 | Measurement of inter- and intrarater variability | N/A | No new annotation performed by the authors |

## METHODS — Data Partitions

| # | Item | Addressed? | Location / Note |
|---|------|-----------|-----------------|
| 18 | Intended sample size and how it was determined | Yes | Full FracAtlas dataset (n = 4,083; 4,068 after duplicate removal); no subsampling |
| 19 | How data were assigned to partitions | Yes | Stratified random split, fixed seed 42 (Methods — Data splitting). Image-level partitioning; limitation discussed |
| 20 | Level at which partitions were disjoint (e.g., image, patient) | Yes | Image level, owing to absence of patient identifiers; explicitly stated and discussed in Limitations |

## METHODS — Model

| # | Item | Addressed? | Location / Note |
|---|------|-----------|-----------------|
| 21 | Detailed description of model | Yes | Methods — Architectures (ResNet-50, EfficientNet-B4, ViT-B/16, ConvNeXt-Tiny via timm) |
| 22 | Software libraries, frameworks, and packages | Yes | PyTorch / timm (Methods); full environment in GitHub repository |
| 23 | Initialization of model parameters | Yes | ImageNet-pretrained weights (Methods — Architectures; Training) |

## METHODS — Training

| # | Item | Addressed? | Location / Note |
|---|------|-----------|-----------------|
| 24 | Details of training approach | Yes | Methods — Training (class-weighted BCE, AdamW, cosine annealing, AMP, gradient clipping, early stopping, seed 42, hardware) |
| 25 | Method of selecting the final model | Yes | Best validation-AUC checkpoint retained (Methods — Training) |
| 26 | Details of ensembling, if used | N/A | No ensembling; single model per architecture |

## METHODS — Evaluation

| # | Item | Addressed? | Location / Note |
|---|------|-----------|-----------------|
| 27 | Metrics of model performance | Yes | AUC, sensitivity, specificity, PPV, NPV, F1, ECE (Methods — Evaluation, Calibration) |
| 28 | Statistical measures of significance and uncertainty (e.g., confidence intervals) | Yes | 95% percentile bootstrap CIs (1,000 iterations); pairwise DeLong tests with Holm-Bonferroni correction (Methods — Evaluation) |
| 29 | Robustness or sensitivity analysis | Partial | Stratified 5-fold cross-validation; external validation on GRAZPEDWRI-DX. A near-duplicate leave-out sensitivity analysis is noted as future work (Limitations) |
| 30 | Methods for explainability or interpretability | Yes | Grad-CAM, with stated limitations (Methods — Interpretability; Results) |
| 31 | Validation or testing on external data | Yes | Zero-shot external validation on GRAZPEDWRI-DX (Methods — External validation; Results) |

## RESULTS

| # | Item | Addressed? | Location / Note |
|---|------|-----------|-----------------|
| 32 | Flow of participants/cases, using a diagram to indicate inclusion/exclusion | Partial | Case counts reported in text (4,083 → 4,068 after duplicate removal; 2,858/612/613 splits); no separate flow diagram |
| 33 | Demographic and clinical characteristics of cases in each partition | Partial | Fracture prevalence per split reported; detailed demographics limited by source dataset metadata |
| 34 | Performance metrics for each data partition | Yes | Test-set table (Table 1); cross-validation table (Table 3); external results (Results) |
| 35 | Estimates of diagnostic accuracy and their precision | Yes | AUC with 95% CIs (Table 1); confusion matrices (Figure) |
| 36 | Failure analysis of incorrectly classified cases | Yes | Grad-CAM failure-mode analysis (false positives / false negatives) (Results — Interpretability) |

## DISCUSSION

| # | Item | Addressed? | Location / Note |
|---|------|-----------|-----------------|
| 37 | Study limitations, including potential bias, statistical uncertainty, and generalizability | Yes | Limitations (dataset size, image-level split, residual near-duplicates, single seed, single external cohort; ViT training-recipe caveat in Discussion) |
| 38 | Implications for practice, including the intended clinical role | Yes | Discussion (triage vs confirmatory vs autonomous use; concrete false-positive rate; decision-support framing) |

## OTHER INFORMATION

| # | Item | Addressed? | Location / Note |
|---|------|-----------|-----------------|
| 39 | Registration number and registry name, if applicable | N/A | Not a registered clinical trial (retrospective public-dataset benchmark) |
| 40 | Where the full study protocol can be accessed | Yes | Code and configuration: GitHub repository (github.com/21yassinezouair/fracture-ai); model checkpoint archived on Zenodo (DOI 10.5281/zenodo.21500630) |
| 41 | Sources of funding and role of funders | Yes | Funding declaration (doctoral scholarship; no dedicated project funding) |

---

## Notes on Completion

- This checklist reflects the study as reported in the submitted manuscript.
- Items marked **N/A** are justified above and stem from the use of publicly available, pre-de-identified, pre-annotated datasets rather than newly collected clinical data.
- "Partial" indicates the item is addressed in text but not with the full formalism (e.g., a flow diagram) that a primary clinical study might include; this is appropriate for a retrospective benchmark on public data.
- The corresponding manuscript section is indicated for each "Yes" item to facilitate reviewer verification.

*Item numbering follows the structure of the CLAIM 2024 Update. Authors should verify item wording and numbering against the official CLAIM 2024 publication (Radiology: AI 2024;6(4):e240300) before submission, as the checklist may be updated.*
