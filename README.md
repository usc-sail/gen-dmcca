# Multiview Shared Subspace Learning for Speakers and Speech Commands  

## Generalized Deep Multiset Canonical Correlation Analysis for Multiview Learning of Speech Representations

### Package Requirements  

All development was done in Python 2.7
```
scipy==1.0.0
numpy==1.11.0
Keras==2.2.4
pandas==0.22.0
Theano==1.0.4
matplotlib==2.2.2
resampy==0.2.0
tqdm==4.11.2
scikit_learn==0.20.3
utils==0.9.0
```

### Proposed Approach:  

We propose a novel direction of multiview learning to obtain speech representations in presence of multiple known sources of variability. We constrain one mode of variability as multiple views and learn features that are discriminative in the other mode.  A schematic to the approach is shown below:

![drawing](https://docs.google.com/drawings/d/16QHfxoro1UIbR4yB6SA_TDhdlPCLeRqwscBCufErlPM/export/png)


### Implementation Steps:  

1.  Download [**Speech Commands Dataset**](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) as follows: 
```
cd ./data && wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz && tar -xvzf speech_commands_v0.01.tar.gz
```

2. Extract log-mel features (using defaults params in `./audioset_scripts/vggish_params.py`; edit these to replicate experiments in the paper)
```
python extract_logmel_features.py

```

3. Learn *speaker-invariant representations of speech commands* for Command-ID task  
```
python train_speech_commands.py
```

4. Learn *command-invariant representations of speakers* for Speaker-ID task  
```
python train_speaker_embeddings_open_set.py
```

5. Do inference after the model is trained; See example inference code:
```
python test_speaker_embeddings_open_set.py
```

This last script should generate SVM predictions, as well as t-SNE embeddings for the shared representations learnt
