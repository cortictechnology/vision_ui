# Manipulation of 3D Models Using Hand Gesture

This program allows user to manipulation 3D models (.obj format) with their hands.  The project support both the OAK-D and OAK-D-LITE.




## Install dependencies

On an Intel-based macOS machine, run the following command in the terminal:

```
git clone https://github.com/cortictechnology/vision_ui.git
cd vision_ui
python3 -m pip install -r requirements.txt
```

## To run

1. Make sure the OAK-D/OAK-D-Lite device is plug into the computer.
2. In the terminal, run

```
python3 main.py
```

## AI Model description

The ai_models folder includes two Intel Myriad X optimized models:

1. palm_detection_sh4.blob: This is the palm detection model
2. hand_landmark_sh4.blob: This is the model to detect the hand landmarks using the palm detection model

## Credits
* [Google Mediapipe](https://github.com/google/mediapipe)
* [depthai_hand_tracker from geaxgx](https://github.com/geaxgx/depthai_hand_tracker)