# External datasets for intent training (optional)

Put downloaded files in this folder (`training_data/`). Run:

```powershell
python 04_intent_dataset.py --movies databse.csv --import-dir training_data
python train_intent_classifier.py
```

## Required format (easiest)

Any **CSV** or **JSONL** with two columns:

| text | label |
|------|--------|
| who has the lead role in harry potter movies | factual_cast |
| movies by christopher nolan | recommend |
| what happens in inception | factual_plot |

**Valid labels:** `factual_cast`, `factual_director`, `factual_plot`, `factual_year`, `factual_other`, `recommend`, `discussion`

Column aliases also work: `query`/`utterance` + `intent`/`category`.

---

## Recommended downloads (free)

### 1. CLINC150 (English — broad phrasing, ~22k train)

Helps the classifier handle varied wording. Map only loosely to movie intents; synthetic movie data is still the main signal.

- **GitHub:** https://github.com/jeroboyle/Clinc150  
- **Direct JSON (train):** https://raw.githubusercontent.com/jeroboyle/Clinc150/master/data/data_full.json  
- Save as: `training_data/clinc150/data_full.json`  
- Or clone the repo into `training_data/clinc150/`

### 2. SNIPS (smaller, good for short commands)

- **GitHub:** https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines  
- Export to CSV with `text,label` yourself, or use our synthetic set only.

### 3. Bitext movie conversations (optional — chat style, not labels)

For LoRA **tone**, not intent routing:

- **Kaggle:** https://www.kaggle.com/datasets/arsalanaftab001/bitext-movie-conversations-dataset  
- Convert to `instruction`/`output` JSONL if you want; not required for intent classifier.

---

## What we do *not* need

- Image datasets  
- Full IMDB review dumps (unless you convert to intent-labeled CSV yourself)  
- Multi-GB video metadata unless you extract `text,label` pairs  

---

## Suggested workflow

```powershell
cd C:\Users\JOEY\Documents\slm
.\.venv\Scripts\Activate.ps1

# 1) Build movie + intent examples (no download required)
python 04_intent_dataset.py --movies databse.csv --chat-addon

# 2) Train intent classifier (~2–5 min CPU)
pip install scikit-learn joblib
python train_intent_classifier.py

# 3) Optional: merge intent examples into LoRA dataset and re-train chat model
python 00_synthetic_dataset.py --input databse.csv --output dataset.jsonl
python 06_merge_datasets.py --base dataset.jsonl --addon data/intent_chat_addon.jsonl --output dataset_full.jsonl
python 02_train_qlora.py --dataset dataset_full.jsonl --output .\movie-nerd-lora --merge
```

After step 2, restart `web_chat.py` — routing uses **rules + trained intent model**.
