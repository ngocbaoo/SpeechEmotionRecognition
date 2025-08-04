
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

### Requirements

- Python 3.8+
- TensorFlow / Keras
- librosa
- numpy, pandas, matplotlib, sklearn

### Acknowledgments

- RAVDESS Dataset: https://zenodo.org/record/1188976
