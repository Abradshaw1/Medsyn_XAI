# Medsyn_XAI

This repository provides tools for generating synthetic CT volumes from text prompts and for visualizing how specific medical terms (identified via NER) influence the generated imagery. It is intended for researchers exploring patient-clinician interaction, explainability, and saliency attribution in generative radiology models.

1. `generate_explanations/` — handles synthetic CT generation and saliency visualization this is all stored on the jetstream but I made kept a cleaner copy here
2. `NER/` — focuses on Named Entity Recognition (NER)

---

## generate_explanations/

These are a cleaned up version of the jetstream except for the stage1.py

### Files:
- **`RunModel.ipynb`** – Runs inference using a text-to-CT model and optionally overlays saliency heatmaps.
- **`extract_text.py`** – Uses a pretrained BERT-style model to extract token embeddings from a radiology report.
- **`dataloader.py`** – Prepares and transforms imaging/text data using MONAI and PyTorch tools.
- **`text.py`** – Tokenizes and embeds free-text inputs for conditional inference.
- **`token_avg_vis.ipynb`** – Loads saved attention/saliency maps and visualizes token-level averages.
- **`temp.ipynb`** – Prototype/debugging notebook for quick testing and experiments.
- **`stage1.py`** – main scritpt responsible for sampling and diffusion process

---

## Named Entity Recognition (NER) Module

The NER module is designed to identify clinical entities (e.g. `FINDING`, `BODY_PART`, `MODIFIER`) from text prompts to be used for word-voumle explanation extraction and pairing downsteam. 

### Files:

#### `NER_Averages.py`

This script is the core logic for the NER pipeline.

1. **Named Entity Extraction**:
   - Loads a pretrained transformer model (`RadBERT`) fine-tuned for clinical NER.
   - Extracts labeled tokens (e.g. `FINDING`, `BODY_PART`) from the input text.
   - Converts labeled tokens into single words and n-grams.

2. **Heatmap Matching**:
   - Waits until saliency heatmaps have been generated for the given `study_id`.
   - Uses regex-based matching to associate NER tokens with per-token `.npy` heatmap files generated during sampling (e.g., `_token_14_pneumonia_heatmaps.npy`).

3. **Resizing and Aggregating**:
   - Resizes 3D saliency maps to a uniform `(64, 64, 64)` voxel grid.
   - Averages heatmaps if multiple maps correspond to the same entity.

4. **Saving NER-Specific Heatmaps**:
   - Stores final heatmaps under `results/NER_subsets_heatmaps/`, labeled by entity name (e.g., `pneumonia_heatmap.npy`).
   - Optionally computes average heatmaps for multi-word phrases (e.g., `"lung consolidation"`).

> This file is intended to be run after saliency maps have already been generated for a given study. After generating heatmps for all tokens we then use NER on the original prompts in order to select the main pathologies, or it can be adpated to hook into stage one executing during the generation poreocess.

#### `NER_stage1`
- basically same as the original stage 1 but you can  hook the NER script just handle the imag volumes post-hoc.
- Raw output from the first stage of NER preprocessing.
- Token-to-label mappins before final filtering or aggregation.
- Used to decouple preprocessing from sampling—for instance, NER labels can be cached early and later reused without recomputation.


