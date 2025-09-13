# MTL Midsem Project ( Neeharika's work)

## Aim
Multi-task learning for **emotion classification, stress prediction, and affective state (valence-arousal) regression** using facial features extracted from videos.

## Project Pipeline (so far)
Video → OpenFace → Feature Extraction → Aligned Images & CSV/Parquet → Preprocessing → Dataset Splits → Fine-tuning / Testing

**Details:**
1. **Video Input:** Videos of subjects expressing different emotions (Anger, Sad, Happy, Neutral).  
2. **OpenFace Processing:**  
   - Extract **facial landmarks, head pose, gaze, and Action Unit (AU) features**  
   - Generate **aligned face images** for each video frame  
   - Produce **CSV and Parquet file** with features per emotion (e.g., `Anger.csv`)  
3. **Feature Processing (Preprocessing):**  
   - 70% of rows from each CSV used for **encoder pretraining** (unlabelled)  
   - 30% of rows from each CSV used for **multi-task classifier** (labelled with emotion, stress, valence, arousal)  
4. **Labelled Dataset Split (70–20–10):**  
   - **70% SSL Dataset** → Pretraining encoder (unlabelled)  
   - **20% Classifier Dataset** → General supervised classifier training  
   - **10% Fine-tuning Dataset** → Personalized / domain-specific fine-tuning  
   - Stratified split to maintain per-emotion balance  

## Folder Structure
```

mtl-midsem/
│
├─ code/
│ ├─ prepare_datasets.py # Preprocess OpenFace CSVs, create pretrain & labelled datasets
│ └─ split_labelled.py # Split labelled dataset into SSL, Classifier, Fine-tuning sets
│
├─ results/
│ ├─ openface_csv/ # CSVs extracted from OpenFace
│ │ ├─ Anger/
│ │ │ ├─ Anger.csv
│ │ │ ├─ Anger.avi
│ │ │ ├─ Anger.hog
│ │ │ ├─ Anger_aligned/ # Folder with aligned face images
│ │ │ └─ Anger_of_details.txt
│ │ ├─ Sad/
│ │ │ ├─ Sad.csv
│ │ │ ├─ Sad.avi
│ │ │ ├─ Sad.hog
│ │ │ ├─ Sad_aligned/
│ │ │ └─ Sad_of_details.txt
│ │ ├─ Happy/
│ │ │ ├─ Happy.csv
│ │ │ ├─ Happy.avi
│ │ │ ├─ Happy.hog
│ │ │ ├─ Happy_aligned/
│ │ │ └─ Happy_of_details.txt
│ │ └─ Neutral/
│ │ ├─ Neutral.csv
│ │ ├─ Neutral.avi
│ │ ├─ Neutral.hog
│ │ ├─ Neutral_aligned/
│ │ └─ Neutral_of_details.txt
│ │
│ ├─ parquet/
│ │ └─ affectnet_openface.parquet
│ │
│ └─ processed_csv/
│ ├─ pretrain_dataset.csv
│ ├─ labelled_dataset.csv
│ ├─ ssl_dataset.csv
│ ├─ classifier_dataset.csv
│ └─ finetune_dataset.csv
|
├─ report/mtl_midsem_report.pdf
└─ slides/mtl_midsem_slides.pptx

````

## Instructions to Run

1. **Clone repository**
```bash
git clone <repo-url>
cd mtl-midsem
````

2. **Setup Python environment**

```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

**Dependencies:** pandas, scikit-learn, OpenFace (installed separately)

3. **Run OpenFace on video data**

```bash
# Navigate to OpenFace installation folder, then run:
FeatureExtraction -f /path/to/video.avi -out_dir /path/to/output/folder -aligned
```

**Example commands per emotion folder:**

```bash
FeatureExtraction -f results/videos/Anger/Anger1.avi -out_dir results/openface_csv/Anger/ -aligned
FeatureExtraction -f results/videos/Sad/Sad1.avi -out_dir results/openface_csv/Sad/ -aligned
FeatureExtraction -f results/videos/Happy/Happy1.avi -out_dir results/openface_csv/Happy/ -aligned
FeatureExtraction -f results/videos/Neutral/Neutral1.avi -out_dir results/openface_csv/Neutral/ -aligned
```

* Produces CSV files in `results/openface_csv/<Emotion>/`
* Produces optional **`affectnet_openface.parquet`** 
* Produces aligned images in `results/aligned_images/<Emotion>/`

4. **Prepare pretrain and labelled datasets**

```bash
python code/prepare_datasets.py
```

* Generates `pretrain_dataset.csv` → 70% unlabelled data for encoder pretraining
* Generates `labelled_dataset.csv` → 30% labelled data for multi-task classifier

5. **Split labelled dataset into SSL / Classifier / Fine-tuning sets**

```bash
python code/split_labelled.py
```

* Generates:

  * `ssl_dataset.csv` → 121 rows
  * `classifier_dataset.csv` → 34 rows
  * `finetune_dataset.csv` → 18 rows
* Stratified by emotion to maintain balance

### Class Distribution per Split

| Dataset           | Angry | Sad | Neutral | Happy | Total |
| ----------------- | ----- | --- | ------- | ----- | ----- |
| SSL (70%)         | 42    | 42  | 27      | 10    | 121   |
| Classifier (20%)  | 12    | 12  | 8       | 2     | 34    |
| Fine-tuning (10%) | 6     | 6   | 4       | 2     | 18    |

---

## Notes on Labels

* **Emotion:** Comes from folder name (`Anger`, `Sad`, `Happy`, `Neutral`)
* **Stress:** Derived from emotion

  * Angry & Sad → High (1)
  * Happy & Neutral → Low (0)
* **Valence-Arousal (Affective State):**

| Emotion | Valence | Arousal |
| ------- | ------- | ------- |
| Happy   | 2.0     | 2.0     |
| Sad     | 1.0     | 1.0     |
| Angry   | 1.0     | 2.0     |
| Neutral | 1.5     | 1.5     |

---

## Notes on Fine-tuning / Last Step

* **SSL dataset (`ssl_dataset.csv`)** → 70% of labelled data used for semi-supervised pretraining of the encoder.  
* **Classifier dataset (`classifier_dataset.csv`)** → 20% of labelled data used to train a general multi-task classifier (emotion, stress, valence-arousal).  
* **Fine-tuning dataset (`finetune_dataset.csv`)** → 10% of labelled data used for personalized or domain-specific fine-tuning of the classifier.  
* All datasets are **stratified by emotion** to maintain label balance.  
* Testing should be performed on **held-out videos** or a separate dataset, as no separate test CSV is generated by the current code.  
* After this step, the pipeline is ready for **multi-task learning experiments**.


### ⚠️ Class Imbalance Note

* The **`happy`** emotion has fewer samples across all splits:

  * Total labelled samples: Angry (60), Sad (60), Neutral (39), Happy (14)
* This may affect model performance on the `happy` class.

**Suggested Remedies:**

1. Data augmentation (SMOTE, mixup, perturbations)
2. Class weighting in loss computation
3. Resampling (oversample `happy` / undersample others)
4. Use **per-class F1 / macro metrics** for evaluation
