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

<br/>

### Directory Structure
> 

    Project
    
    ├── speech_emotion_recognition
    │   ├── code
    │   │   ├── train_ser.py
    │   │   ├── test_ser.py
    │   ├── data
    │   ├── pretrained_weights
    
    ├── conditional_gan
    │   ├── code
    │   │   ├── train_cgan.py
    │   │   ├── test_cgan.py
    │   ├── data
    │   ├── pretrained_weights
    
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

To train both the models separately, run commands below. 

For SER:
```bash
$ python train_ser.py
```

For cGAN:
```bash
$ python train_cgan.py
```
<br>

### Prediction

<br/>

## References

<br/>

