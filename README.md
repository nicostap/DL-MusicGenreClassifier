# Music Genre Classification Using Deep Learning

## 1. Problem
* The rapid growth of digital music streaming platforms necessitates accurate and efficient music recommendation systems.
* To build these systems, automatic music genre classification is required as a labeling mechanism. 
* Manual labeling is highly ineffective and resource-intensive for large-scale music databases, creating a strong need for an artificial intelligence approach.
* While the GTZAN dataset is heavily used as a standard benchmark, it lacks representation of highly popular local Indonesian music genres, specifically "Dangdut". 

## 2. Method
* **Dataset:** The project utilizes the standard GTZAN dataset (containing 10 genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock) and adds 100 manually curated Dangdut music clips extracted from YouTube.
* **Audio Pre-processing:** The dataset was tested in three configurations: full 30-second clips, segmented into 3-second clips with 50% overlap, and segmented into 3-second clips without overlap. 
* **Feature Extraction:** The researchers tested modern spectral features including Mel-Spectrogram (with FFT window sizes of 1024 and 2048) and Mel-Frequency Cepstral Coefficients (MFCC), alongside classic timbral, rhythmic, and pitch features.
* **Model Architectures:** Several deep learning models were designed and compared, including:
    * CNN + LSTM
    * CNN + GRU
    * CNN + Bi-LSTM
    * CNN + Bi-GRU
    * MLP and pure LSTM
    * **Multiscale CNN Transformer:** A custom multi-branch architecture designed to capture local, time-axis, and frequency-axis features simultaneously.
* **Evaluation:** Models were evaluated at both the segment-level and song-level, and SHAP (SHapley Additive exPlanations) analysis was used to interpret the frequency-time impact for each genre.

## 3. Result
* **Best Performing Model:** The proposed Multiscale CNN Transformer, utilizing Mel-Spectrogram (FFT window size 2048) and song-level evaluation, achieved the highest accuracy.
* **Accuracy Benchmark:** Before adding the Dangdut genre, the Multiscale CNN Transformer achieved a 90.91% accuracy.
* **Impact of Local Genre:** After adding Dangdut, the model's accuracy actually increased to 93.64%. The introduction of this local genre did not degrade the model's performance. 
* **Classification Efficacy:** Confusion matrix analysis showed that the model recognized the Dangdut genre consistently and accurately, proving the system can be both inclusive and highly performant.
* **Feature Insights:** SHAP analysis successfully revealed how specific frequency ranges influence classification; for example, Dangdut is positively influenced by frequencies below 3000 Hz and above 8000 Hz, while remaining neutral in between.