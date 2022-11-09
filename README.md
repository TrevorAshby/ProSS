# ProSS
ProSS - Prosodic Speech Synthesis 

![Entire Pipeline](ProSS.png "Prosodic Speech Synthesis")

## Technologies
- WaveNet: 
- BERT
- Automatic Prosody Annotation: https://arxiv.org/abs/2206.07956
- Tacotron2: https://arxiv.org/pdf/1712.05884.pdf
- Waveglow: https://arxiv.org/pdf/1811.00002.pdf

## Datasets
Dataset: https://keithito.com/LJ-Speech-Dataset/
- Extract the dataset into 'src/ProSS/LJSpeech-1.1'

Other Dataset: http://www.festvox.org/cmu_arctic/
- Extract the dataset into 'src/ProSS/cmu_arctic/
    - This can be done with !wget -e robots=off -r -np http://www.festvox.org/cmu_arctic/cmu_arctic/cmu_us_bdl_arctic/
    - Currently only using 'cmu_us_bdl_arctic'

## Automatic Prosody Annotation
Method based upon this paper: https://arxiv.org/abs/2206.07956
- Audio Encoder: "PPG extractor: Phonetic posteriorgram (PPG)"
    - wav -> spectrogram -> mfccs -> phoneme dist.
        - alternatively I am developing a model capable of wav -> PPG conversion.
    - incredibly helpful read: https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd
- Text Encoder: Pre-Trained English BERT: https://huggingface.co/bert-base-uncased?text=The+goal+of+life+is+%5BMASK%5D.
    ## Annotations
    - based upon ToBI annotations: https://linguistics.ucla.edu/people/jun/papers%20in%20pdf/J54-ToBI%20Ch04%20ToBI%20and%20commentary-MIT%20Press2022.pdf


