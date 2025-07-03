
## Speech Emotion Recognition (SER) using CNN + LSTM

This project implements a Speech Emotion Recognition system using deep learning (CNN + LSTM) to classify emotions from speech audio. The main dataset used is **RAVDESS**, and the model is trained on **MFCC features** extracted from the audio signals.

### Dataset

- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- Includes 8 emotions: `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`

You can download the dataset from: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

### Model Architecture

- **Preprocessing:**
  - Audio is resampled and normalized.
  - Extracted MFCC (Mel-frequency cepstral coefficients) with 40 features.
  - Each audio is converted to a fixed shape of `(216, 40)` MFCC matrix.

- **Model:**
  - CNN layers to extract spatial features.
  - LSTM to capture temporal dynamics.
  - Fully connected layers to classify into 8 emotions.

### How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download and place dataset:

Download RAVDESS and extract into the `audio/` folder. Adjust path if needed in the notebook.

3. Run the notebook:

Open `SER.ipynb` and run all cells. It includes:
- Data loading
- Feature extraction (MFCC)
- Model training (CNN + LSTM)
- Evaluation (accuracy and confusion matrix)

### Results

- Model achieves high accuracy on training and validation sets.
- The notebook provides plots for training history and confusion matrix.

### Future Work

- Add support for other datasets: CREMA-D, SAVEE, Toronto Emotion.
- Apply noise augmentation for robustness.
- Convert to real-time inference app with Streamlit or Flask.
- Integrate attention mechanism into LSTM.

### Requirements

- Python 3.8+
- TensorFlow / Keras
- librosa
- numpy, pandas, matplotlib, sklearn

### Acknowledgments

- RAVDESS Dataset: https://zenodo.org/record/1188976
- Emotion recognition research community
