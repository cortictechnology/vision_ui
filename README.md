# Manipulation of 3D Models Using Hand Gesture

This program allows user to manipulation 3D models (.obj format) with their hands.  The project support both the OAK-D and OAK-D-LITE.

![3d-manipulation](./resources/3D_Manipulation.gif)


## Install dependencies

On an Intel-based macOS or Linux machine, run the following command in the terminal:

```
git clone https://github.com/cortictechnology/vision_ui.git
cd vision_ui
python3 -m pip install -r requirements.txt
```

For Linux only, make sure your OAK-D device is not plugged in and then run the following:

```
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
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