import os
import re
import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from scipy.ndimage import zoom
import time
import logging

SAL_MAPS_FOLDER = "/media/volume/gen-ai-volume/MedSyn/results/saliency_maps/"
NER_HEATMAPS_FOLDER = "/media/volume/gen-ai-volume/MedSyn/results/NER_subsets_heatmaps/"
LOG_FILE = "/home/exouser/MedsynBackend/src/log.txt"  # Ensure it matches the model log

logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Load RadBERT for NER
NER_MODEL_NAME = "StanfordAIMI/RadBERT"
tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def extract_medical_keywords(text):
    """Extract key medical terms from the input text using RadBERT."""
    logging.debug(f"Original Input Text: {text}")

    # Tokenize with offsets to map back to words
    inputs = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True).to(device)
    tokenized_words = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    logging.debug(f"Tokenized Words: {tokenized_words}")

    with torch.no_grad():
        outputs = model(**inputs).logits
        logging.debug(f"Model Output Shape: {outputs.shape}") 

    predictions = torch.argmax(outputs, dim=-1).cpu().numpy()[0]

    # Map token indices to NER labels
    id2label = model.config.id2label
    predicted_labels = [id2label[idx] for idx in predictions]
    logging.debug("NER Predictions per Token:")
    for i, (token, label) in enumerate(zip(tokenized_words, predicted_labels)):
        logging.debug(f"Token: {token}, Label: {label}")
    extracted_keywords = [
        tokenized_words[i] for i, label in enumerate(predicted_labels) 
        if label not in ["O", "[PAD]", "[CLS]", "[SEP]"]
    ]
    logging.debug(f"Extracted NER Tokens: {extracted_keywords}")
    extracted_ngrams = extracted_keywords[:]
    for i in range(len(extracted_keywords) - 1):
        extracted_ngrams.append(f"{extracted_keywords[i]} {extracted_keywords[i+1]}")
    for i in range(len(extracted_keywords) - 2):
        extracted_ngrams.append(f"{extracted_keywords[i]} {extracted_keywords[i+1]} {extracted_keywords[i+2]}")

    extracted_ngrams = list(set(extracted_ngrams))  # Remove duplicates this is hacky, using more powerful BERT varaint or jsut transfomrer might be better
    logging.debug(f"Extracted NER Keywords (with n-grams): {', '.join(extracted_ngrams)}")
    return extracted_ngrams

def wait_for_heatmaps(study_id, max_wait=300):
    """Wait until heatmaps are available for the study ID."""
    heatmap_folder = os.path.join(SAL_MAPS_FOLDER, study_id)
    waited = 0

    while waited < max_wait:
        heatmap_files = [f for f in os.listdir(heatmap_folder) if f.endswith(".npy")]
        if heatmap_files:
            return heatmap_files
        time.sleep(5)
        waited += 5

    raise TimeoutError(f"No heatmaps found for {study_id} after {max_wait} seconds.")

def resize_heatmap(heatmap, target_shape=(64, 64, 64)):
    """Resize heatmap to match CT scan dimensions."""
    zoom_factors = (target_shape[0] / heatmap.shape[0], target_shape[1] / heatmap.shape[1], target_shape[2] / heatmap.shape[2])
    return zoom(heatmap, zoom_factors, order=1)

def match_ner_terms_to_tokens(ner_terms, heatmap_files):
    """
    Match extracted NER terms to the tokenized words in the heatmap filenames.
    Uses regex to match the keyword to the token in filenames.
    """
    matched_files = {}

    for term in ner_terms:
        term_cleaned = re.escape(term.replace(" ", "_"))  # Convert spaces to underscores for matching, terrible nnaming convention ion backend
        pattern = re.compile(rf"_token_\d+_({term_cleaned})_heatmaps\.npy", re.IGNORECASE)

        for filename in heatmap_files:
            if pattern.search(filename):
                matched_files.setdefault(term, []).append(filename)

    return matched_files

def load_and_save_heatmaps(study_id, ner_keywords):
    """Load heatmaps for matched NER keywords and save them individually."""
    heatmap_folder = os.path.join(SAL_MAPS_FOLDER, study_id)
    output_folder = os.path.join(NER_HEATMAPS_FOLDER, study_id)
    os.makedirs(output_folder, exist_ok=True)

    heatmap_files = [f for f in os.listdir(heatmap_folder) if f.endswith(".npy")]
    matched_files = match_ner_terms_to_tokens(ner_keywords, heatmap_files)

    saved_heatmaps = {}

    for keyword, files in matched_files.items():
        temp_stack = []
        for file in files:
            heatmap_path = os.path.join(heatmap_folder, file)
            heatmap_data = np.load(heatmap_path)
            resized_heatmap = resize_heatmap(heatmap_data, target_shape=(64, 64, 64))
            temp_stack.append(resized_heatmap)

        if temp_stack:
            if len(temp_stack) == 1:
                final_heatmap = temp_stack[0]
            else:
                final_heatmap = np.mean(np.stack(temp_stack, axis=0), axis=0)  # Average if multiple

            save_path = os.path.join(output_folder, f"{keyword}_heatmap.npy")
            np.save(save_path, final_heatmap)
            saved_heatmaps[keyword] = save_path

            with open(LOG_FILE, "a") as log:
                log.write(f"Saved heatmap for '{keyword}' at {save_path}\n")

    return saved_heatmaps

def generate_phrase_heatmaps(saved_heatmaps):
    """Generate averaged heatmaps for multi-word phrases."""
    for phrase, heatmap_path in saved_heatmaps.items():
        words = phrase.split()
        if len(words) > 1:
            individual_paths = [saved_heatmaps[word] for word in words if word in saved_heatmaps]
            if len(individual_paths) > 1:
                heatmaps = [np.load(p) for p in individual_paths]
                avg_heatmap = np.mean(np.stack(heatmaps, axis=0), axis=0)

                phrase_heatmap_path = os.path.join(os.path.dirname(heatmap_path), f"{phrase.replace(' ', '_')}_heatmap.npy")
                np.save(phrase_heatmap_path, avg_heatmap)

                with open(LOG_FILE, "a") as log:
                    log.write(f"Saved averaged phrase heatmap for '{phrase}' at {phrase_heatmap_path}\n")

def process_ner_heatmaps(prompt, study_id):
    """Main function to extract NER terms, match tokens, and generate heatmaps."""
    ner_keywords = extract_medical_keywords(prompt)
    print(f"Extracted NER Keywords: {ner_keywords}")

    wait_for_heatmaps(study_id)
    saved_heatmaps = load_and_save_heatmaps(study_id, ner_keywords)
    generate_phrase_heatmaps(saved_heatmaps)
