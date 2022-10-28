# ProSS
ProSS - Prosodic Speech Synthesis 


Dataset: https://keithito.com/LJ-Speech-Dataset/
- Extract the dataset into 'src/ProSS/LJSpeech-1.1'

## Automatic Prosody Annotation
Method based upon this paper: https://arxiv.org/abs/2206.07956
- Audio Encoder: "PPG extractor: Motivated by the great success of Phonetic posteriorgram (PPG)"
- Text Encoder: Pre-Trained English BERT: https://huggingface.co/bert-base-uncased?text=The+goal+of+life+is+%5BMASK%5D.

