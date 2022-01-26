# Physiotherapy Pose Estimator

## Overview
Incorrect posture and counting during physiotherapy sessions has always engaged a human(generally a physiotherapist) which is a waste of human resources in this evolving era of real time computer vision applications. The physiotherapist aka human can always get into more fruitful and creative work while the computer system can monitor the patient's inaccuracy during the exercise.

This project aims at doing the same for knee physiotherapy where the patient has to bend its leg from knee, hold it for 8 seconds; relax by straightening the leg and then repeat the process. 

- Holding timer of 8 seconds with a feedback system which will be activated if person fails to hold for minimum of 8 seconds.
- Leg closer to the camera will be considered for the exercise.
- Leg should be bent to start the timer. 

## Installation and Demo
* Installing the dependencies:
```
pip install requirements.txt
```

* To run the demo:
#### On webcam:
```
python demo.py
```
#### On a video file:
``` 
TO BE DEFINED
```

Press 'q' to quit the OpenCV demonstration.


## Project Description:
* Based on Mediapipe's Pose Estimation Model [Blazepose](https://arxiv.org/abs/2006.10204) at the backend which is responsible for calculating the 3D coordinates of the human body keypoints.
* We extract the three: ankle, knee and hip keypoints and form two straight lines joning ankle and knee(A) / knee and hip(B). Calculating the angle between the two straight lines A and B gives us the angle formed at knee which is then used for counter increments and calculating holding timer. 


## Future Improvements:
* Inclsion of more forms of physiotherapy exercises which is currently limited to knee exercise.
* Retraining the model for better accuracy as the predictions are bad if the person is close within a threshold to the camera.


## References and Credits
1. [Guide to Human Pose Estimation with Deep Learning(Nanonets)](https://nanonets.com/blog/human-pose-estimation-2d-guide/)
2. [Mediapipe Pose Classification(Google's Github)](https://google.github.io/mediapipe/solutions/pose_classification.html)
3. [Real-time Human Pose Estimation in the Browser(TF Blog)](https://blog.tensorflow.org/2018/05/real-time-human-pose-estimation-in.html)
4. [MediaPipePoseEstimation(Nicknochnack's Github)](https://github.com/nicknochnack/MediaPipePoseEstimation)
