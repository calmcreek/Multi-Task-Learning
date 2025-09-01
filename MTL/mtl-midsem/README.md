# MTL Midsem Project

## Aim
Multi-task learning for **emotion classification, stress prediction, and affective state (valence-arousal) regression** using facial features extracted from videos.

## Project Pipeline (so far)
Video → OpenFace → Feature Extraction → Aligned Images & CSV/Parquet → Preprocessing → CSV datasets → Fine-tuning / Testing split

**Details:**
1. **Video Input:** Videos of subjects expressing different emotions (Anger, Sad, Happy, Neutral).  
2. **OpenFace Processing:**  
   - Extract **facial landmarks, head pose, gaze, and Action Unit (AU) features**  
   - Generate **aligned face images** for each video frame  
   - Produce **CSV and Parquet files** with features per emotion (e.g., `Anger.csv` or `Anger.parquet`)  
3. **Feature Processing (Preprocessing):**  
   - 70% of rows from each CSV used for **encoder pretraining** (unlabelled)  
   - 30% of rows from each CSV used for **multi-task classifier** (labelled with emotion, stress, valence, arousal)  
4. **Labelled Dataset Split:**  
   - Labelled dataset is split into **Fine-tuning (80%)** and **Testing (20%)** sets  
   - Stratified split to maintain per-emotion balance  

## Folder Structure
```

mtl-midsem/
│
├─ code/
│   ├─ prepare_datasets.py   # Preprocess OpenFace CSVs, create pretrain & labelled datasets
│   └─ split_labelled.py     # Split labelled dataset into fine-tune and test sets
│
├─ results/
│   ├─ openface_csv/
│   │   ├─ Anger/Anger.csv
│   │   ├─ Sad/Sad.csv
│   │   ├─ Happy/Happy.csv
│   │   └─ Neutral/Neutral.csv
│   ├─ parquet/
│   │   └─ affectnet_openface.parquet
│   ├─ aligned_images/
│   │   ├─ Anger/
│   │   ├─ Sad/
│   │   ├─ Happy/
│   │   └─ Neutral/
│   └─ processed_csv/
│       ├─ pretrain_dataset.csv
│       ├─ labelled_dataset.csv
│       ├─ fine_tune_dataset.csv
│       └─ test_dataset.csv
│
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
* Produces optional parquet file in `results/parquet/`
* Produces aligned images in `results/aligned_images/<Emotion>/`

4. **Prepare pretrain and labelled datasets**

```bash
python code/prepare_datasets.py
```

* Generates `pretrain_dataset.csv` → 70% unlabelled data for encoder pretraining
* Generates `labelled_dataset.csv` → 30% labelled data for multi-task classifier

5. **Split labelled dataset into Fine-tuning and Testing sets**

```bash
python code/split_labelled.py
```

* Generates `fine_tune_dataset.csv` → 138 rows
* Generates `test_dataset.csv` → 35 rows
* Stratified by emotion to maintain balance

### Results of Split

| Dataset     | Angry | Sad | Neutral | Happy | Total |
| ----------- | ----- | --- | ------- | ----- | ----- |
| Fine-tuning | 48    | 48  | 31      | 11    | 138   |
| Testing     | 12    | 12  | 8       | 3     | 35    |

## Notes on Labels

* **Emotion:** Comes from folder name (`Anger`, `Sad`, `Happy`, `Neutral`)
* **Stress:** Derived from emotion

  * Angry & Sad → High (1)
  * Happy & Neutral → Low (0)
* **Valence-Arousal (Affective State):**
  \| Emotion | Valence | Arousal |
  \|---------|---------|---------|
  \| Happy   | 2.0     | 2.0     |
  \| Sad     | 1.0     | 1.0     |
  \| Angry   | 1.0     | 2.0     |
  \| Neutral | 1.5     | 1.5     |

## Notes on Fine-tuning / Last Step

* **Fine-tuning dataset (`fine_tune_dataset.csv`)** is used to train the multi-task classifier using the pre-trained encoder.
* **Testing dataset (`test_dataset.csv`)** is used to evaluate performance on unseen data.
* Both datasets are **stratified by emotion** to maintain label balance.
* After this step, the pipeline is ready for **multi-task learning experiments**.
