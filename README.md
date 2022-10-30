# ProSS
ProSS - Prosodic Speech Synthesis 


Dataset: https://keithito.com/LJ-Speech-Dataset/
- Extract the dataset into 'src/ProSS/LJSpeech-1.1'

Other Dataset: http://www.festvox.org/cmu_arctic/
- Extract the dataset into 'src/ProSS/cmu_arctic/
    - This can be done with !wget on colab

## Automatic Prosody Annotation
Method based upon this paper: https://arxiv.org/abs/2206.07956
- Audio Encoder: "PPG extractor: Phonetic posteriorgram (PPG)"
    - wav -> spectrogram -> mfccs -> phoneme dist.
    - incredibly helpful read: https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd
- Text Encoder: Pre-Trained English BERT: https://huggingface.co/bert-base-uncased?text=The+goal+of+life+is+%5BMASK%5D.


