# Multi-Task-Learning
### Stiti's Work till 07/09/2025

1. **Pretrained the datasets**
   - Dropped columns that were entirely NaN.  
   - Filled remaining NaNs.  
   - Prepared clean datasets for SSL.  

2. **Implemented two encoders**
   - Physiological signals encoder.  
   - Image encoder.  
   - Trained both encoders using SSL (contrastive learning).  

3. **Implemented the classifier with three multi-task heads**
   - Emotion classification.  
   - Stress detection.  
   - Valence–Arousal regression.  

4. **Fine-tuning**
   - Performed loss calculation and backpropagation using labelled datasets.  
