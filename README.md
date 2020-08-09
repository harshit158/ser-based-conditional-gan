# Speech Emotion based face generation using Condition GANs
Generating human faces through conditional GANs which are conditioned on emotions identified from a human speech using SER (Speech Emotion Recognition)

An image showing the overall pipeline
![alt text](images/pipeline.png 'ser')

Below is a short demo of the web app showing generation of human faces based on emotion identified from human speech.
![alt text](images/demo.gif)

## Results

Training samples
![alt text](images/sample_images.png)

Generated samples
![alt text](images/generated_images.png)
 
<br/>

## Getting Started

### Prerequisites
```bash
pandas==1.0.4
Keras==2.3.1
librosa==0.7.2
streamlit==0.61.0
tensorflow==2.0.0
numpy==1.18.1
tqdm==4.42.0
scipy==1.4.1
tensorflow_hub==0.8.0
matplotlib==3.1.3
Flask==1.1.2
ipython==7.17.0
Pillow==7.2.0
pyaudio==0.2.11
scikit_learn==0.23.2
```

<br/>

### Directory Structure
> 

    Project
    
    ├── speech_emotion_recognition
    │   ├── code
    │   │   ├── ser_training.ipynb
    │   │   ├── ser_prediction.ipynb
    │   ├── data
    │   │   ├── Audio_Speech_Actors_01-24
    │   │   │   ├── Actor_01
    │   │   │   │   ├── 03-01-01-01-01-01-01.wav
    │   │   │   │   ├── 03-01-01-01-01-02-01.wav
    │   │   │   │   ...
    │   │   │   ├── Actor_02
    │   │   │   ...
    │   │   │   ├── Actor_24
    │   ├── weights
    
    ├── conditional_gan
    │   ├── code
    │   │   ├── cgan_training.ipynb
    │   │   ├── cgan_prediction.ipynb
    │   ├── data
    │   │   ├── fer2013.csv
    │   ├── weights
    
    ├── streamlit_webapp

### Data
#### For SER : 
The dataset can be downloaded at:<br/>
https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio <br/>
and should be put it in the location<br/>
```bash
./speech_emotion_recognition/data/
```

It consists of speech audios in the voice of 24 actors. 5 sample audio file by the first actor has been put in the above location as an example. <br/>

#### For GANs : 
The dataset can be downloaded at:<br/>
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data <br/>
and should be put it in the location<br/>
```bash
./conditional_gan/data/
```
We are interested in the "fer2013.csv" file from the data bundle. A sample file containing data for only 5 faces has been put as an example.

<br/>

### Model Trainining 

[Note: Please host and run these files on <b>Google Colab</b>] <br/>
For each of SER and cGAN, there are two separate Jupyter Notebook files, one for training and one for prediction. <br/>

#### For SER:

##### Training :</br>
```bash
./speech_emotion_recognition/code/ser_training.ipynb
```
The weights obtained are stored in ./speech_emotion_recognition/weights <br/>
The pretrained weights corresponding to the best model are already put at this location.

##### Prediction :<br/>
```bash
./speech_emotion_recognition/code/ser_prediction.ipynb
```

#### For cGAN:

##### Training :</br>
```bash
./conditional_gan/code/cgan_training.ipynb
```

##### Prediction :<br/>
```bash
./conditional_gan/code/cgan_prediction.ipynb
```
<br>

## References
1. Mirza, M. and Osindero, S., 2014. Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784.
2. Livingstone, S.R. and Russo, F.A., 2018. The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PloS one, 13(5), p.e0196391. 
3. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. and Bengio, Y., 2014. Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
4. Francois Chollet. 2017. Deep Learning with Python (1st. ed.). Manning Publications Co., USA.
5. https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
6. https://medium.com/@ma.bagheri/a-tutorial-on-conditional-generative-adversarial-nets-keras-implementation-694dcafa6282 
7. https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/

<br/>

