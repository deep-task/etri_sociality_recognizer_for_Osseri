# etri_sociality_recognizer_for_Osseri

This Repository is a part of code for social action recognition among the whole technology


## Installation

### Hardware requirements

* NVIDIA Jetson Xavier NX board

### Software requirements

#### Xavier OS

Installation instructions are availale at: [Getting Started With Jetson Xavier NX Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit#intro)

#### Pytorch & torchvision

Installation instructions are availale at: [Pytorch for JetPack 4.4](https://forums.developer.nvidia.com/t/pytorch-for-jetson-nano-version-1-6-0-now-available/72048)
```
### pytorch 
$ wget https://nvidia.box.com/shared/static/ncgzus5o23uck9i5oth2n8n06k340l6k.whl -O torch-1.4.0-cp36-cp36m-linux_aarch64.whl
$ sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
$ pip3 install Cython
$ pip3 install numpy torch-1.4.0-cp36-cp36m-linux_aarch64.whl

### torch vision
$ sudo apt-get install libjpeg-dev zlib1g-dev
$ git clone --branch <version> https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
$ cd torchvision
$ export BUILD_VERSION=0.x.0  # where 0.x.0 is the torchvision version  
$ sudo python setup.py install     # use python3 if installing for Python 3.6
$ cd ../  # attempting to load torchvision from build dir will result in import error
$ pip install 'pillow<7' # always needed for Python 2.7, not needed torchvision v0.5.0+ with Python 3.6
```

#### Build opencv

There is no opencv whl for jetson, so it must be built from the source code.
```
### install dependency
$ sudo apt install build-essential cmake pkg-config -y 
$ sudo apt install libjpeg-dev libtiff5-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libx265-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran python3-dev
$ sudo apt install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-pulseaudio libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

### build opencv
$ wget -O opencv.zip https://github.com/Itseez/opencv/archive/4.1.0.zip
$ unzip opencv.zip
$ cd opencv
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_TBB=OFF \
    -D WITH_IPP=OFF \
    -D WITH_1394=OFF \
    -D BUILD_WITH_DEBUG_INFO=OFF \
    -D BUILD_DOCS=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D WITH_QT=OFF \
    -D WITH_GTK=ON \
    -D WITH_OPENGL=OFF \
    -D WITH_V4L=ON  \
    -D WITH_FFMPEG=ON \
    -D WITH_XINE=ON \
    -D WITH_GSTREAMER=ON \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D PYTHON3_INCLUDE_DIR=/usr/include/python3.6m \
    -D PYTHON3_NUMPY_INCLUDE_DIR=/usr/local/lib/python3.6/dist-packages/numpy/core/include \
    -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.6/dist-packages \
    -D PYTHON3_LIBRARY=/usr/lib/arm-linux-gnueabihf/libpython3.6m.so \
    ../
$ make -j 6
$ sudo -H make install

### Checking the installation results
$ python3 -c "import cv2"
```

#### Torch2trt 

torch2trt is a PyTorch to TensorRT converter which utilizes the TensorRT Python API. 
Installation instructions are availale at: [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

#### Cloning a repository

Clone this repository as follows:

```
$ git clone https://github.com/deep-task/etri_sociality_recognizer_for_Osseri.git
```

#### Download model files.
* Download all the model files (.pth) from [here]().
* Move the model files to a folder  `socialactionrecog/models`.


## How to start client program
* In order to run this `ETRI_Recognition_Xavier (client)`, the server node must be started first.
* Before running the client, you need to modify the server's IP address and port number.
```
### Social_Main.py / (56) line
HOST = "192.168.0.6"
PORT = 9999
```
* Run Social_Main.py 
```
$ cd socialactionrecog
$ python3 Social_Main.py
```

## Example of returned json string
* A sample message is as follows.
```
recog_info = {
        "encoding": "UTF-8",
        "header": {
                "content": ["human_recognitiopn"],
                "source": "ETRI",
                "target": ["UOA", "UOS"],
                "timestamp": 0
        },
        "human_recognition": [{
            "face_roi": {
                    "x1": 113,
                    "x2": 124,
                    "y1": 145,
                    "y2": 149
                },
                # "gender": 1,                            // 0: female, 1: male
                # "age": -1,
                # "headpose": {
                #     "yaw": -7.2235464332,
                #     "pitch": 0.248783278923,
                #     "roll": 8,21489774892244
                # },
                # "glasses": False,
                "social_action": nSocialActionCode,     // -1: not recognized,
                                                        //  0:bitenail,                   
                                                        //  1: covering mouth with hands,
                                                        //  2: cheering up!, 3: finger heart sign,
                                                        //  4: OK sign, 5: crossing arms, 6: neutral
                                                        //  7: picking ears,
                                                        //  8: resting chin on a hand,
                                                        //  9: scratching head, 10: shake hands 
                                                        // 11: a thumb up, 12: touching nose
                                                        // 13: waving hand, 14: bowing 
                # "gaze": nInterested,                    //  0: aversion, 1: contact
                # "name": "",                             // identification result (currently not used)
                # "longterm_tendency": -1,                //  "passive" : 0,
                #                                         //  "neutral" : 1,
                #                                         //  "active : 2
                # "lognterm_habit": -1                    // Habit behavior index
                                                        // same as social_action
        }]
    }

```