[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZUww-RSx-dLpKzSTYrJ55v6Nz6KExxDm#scrollTo=6QNXsBtCt7Hb)

[![arXiv](https://img.shields.io/badge/arXiv-<2004.10934>-<COLOR>.svg)](https://arxiv.org/abs/2004.10934)


# YOLOv4 Darknet for Windows

Training the neural network for object detection. It uses to darknet to do this.

More details on Medium: 

* [YOLOv4](https://medium.com/@alexeyab84/yolov4-the-most-accurate-real-time-neural-network-on-ms-coco-dataset-73adfd3602fe?source=friends_link&sk=6039748846bbcf1d960c3061542591d7)

Main Repository is AlexeyAB's:

* [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet#neural-networks-for-object-detection)

## Table of Contents

 

## Requirements for Windows

* Visual Studio >= 2017: [VS](https://visualstudio.microsoft.com/tr/downloads/)

* CMake >= 3.18: [CMake](https://cmake.org/download/)

* OpenCV >= 2.4: [OpenCV](https://opencv.org/releases/) 
(on Windows set system variable OpenCV_DIR = C:\opencv\build - where are the include and x64 folders image)

* CUDA >= 10.2: [Cuda](https://developer.nvidia.com/cuda-toolkit-archive)

* cuDNN >= 8.0.2: [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)

* GPU with CC >= 3.0: [Control to GPU support](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)

 That is developer's(Alexey) minumum advices. There were no problems with the following versions.

* Visual Studio = 2019
* CMake = 3.20.5
* OpenCV = 4.5.0
* CUDA = 11.2
* cuDNN = 8.0.5

 You need to Python for run scripts or annotations. If you don't install before, you visit this page [setup Python](https://realpython.com/installing-python/).

  

## Darknet Installiation 

### -> First Step: VS 2019

* You download on link then install with marked checkbox.

<img src="/docs/vs.png" alt="Setup VS"/>


### -> Second Step: CUDA

* You install follow setup.
* After, control the system environment variables. Variables must be as on shown images.

<img src="/docs/cuda1.png" alt="Setup Cuda"/>
<img src="/docs/cuda2.png" alt="Setup Cuda"/>
<img src="/docs/cuda3.png" alt="Setup Cuda"/>


### -> Third Step: CuDNN

* You download the folder. Copy all files in folder then paste in this directory "`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`" as shown images.

<img src="/docs/cudnn1.png" alt="Setup Cudnn"/>
<img src="/docs/cudnn2.png" alt="Setup Cudnn"/>


### -> Fourth Step: OpenCV

* You download on link then extract to "`C:\opencv_4.5.0\`". 

<img src="/docs/opencv1.png" alt="Setup OpenCV"/>

* You editing system environment variables as shown images.


<img src="/docs/opencv2.png" alt="Setup OpenCV"/>
<img src="/docs/opencv3.png" alt="Setup OpenCV"/>
<img src="/docs/opencv4.png" alt="Setup OpenCV"/>

* If 'opencv_world' checkbox is not marked, mark it with  Cmake. Firstly clik configure then generate as shown image.

<img src="/docs/cmake1.png" alt="Cmake"/>


### -> Fifth Step: Build Darknet

* Download the my repository. I'm assuming it is in this directory, "`C:\Users\Name\Desktop\darknet\`"
* Copy `cudnn64_8.dll` in directory "`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`" then paste to darknet directory "`C:\Users\Name\Desktop\darknet\build\darknet\x64`".
* Copy `opencv_videoio_ffmpeg450_64.dll`, `opencv_world450.dll`, `opencv_world450d.dll` in directory "`C:\opencv_4.5.0\opencv\build\x64\vc15\bin`" then paste to darknet directory "`C:\Users\Name\Desktop\darknet\build\darknet\x64`".
* Open `darknet.vcxproj`, `yolo_cpp_dll.vcxproj`' with notepad in directory  "`C:\Users\Name\Desktop\darknet\build\darknet\`".
* Change CUDA and OpenCV version to your own version.
* Open `yolo_cpp_dll.sln`, `darknet.sln` with VS2019 and build it like shown image.

<img src="/docs/build1.png" alt="Build"/>
<img src="/docs/build2.png" alt="Build"/>
<img src="/docs/build3.png" alt="Build"/>
<img src="/docs/build4.png" alt="Build"/>
<img src="/docs/build5.png" alt="Build"/>
<img src="/docs/build6.png" alt="Build"/>
<img src="/docs/build7.png" alt="Build"/>
<img src="/docs/build8.png" alt="Build"/>
<img src="/docs/build9.png" alt="Build"/>
<img src="/docs/build10.png" alt="Build"/>

## Preparate to Dataset 

* If you haven't available dataset, you must creat it. 
* For this, I recommend collecting a minimum of 200 images per class to be detected. However, it would be useful to have at least 2000 images  in the independent from  number of classes.
* Now, you must label images. For this, I recommend Labelimg. Follow the [link](https://www.programmersought.com/article/1977864250/) for installation. Change saving format to YOLO and you are ready for annotation.
* Change the `classes.txt` that Labelimg generates to `_darknet.labels` when finish the labeling.
* Dataset needs to be split. You can use `split_data.py` in preprocess folder for splitting dataset as train, validation, test. All three must have `_darknet.labels`.
* Can use `bb_data_aug.py` for increase the train set.
* If you have available dataset, you can use `xml2yolo.py` in preprocess folder for .xml labels to yolo labels.
* After, copy images and annotations in train and valid folders. Paste it in "`C:\Users\Name\Desktop\darknet\build\darknet\x64\data\obj`".
*Copy images and annotations in test folder. Paste it in "`C:\Users\Name\Desktop\darknet\build\darknet\x64\data\test
* You must list of dataset as `train.txt`, `valid.txt`, `test.txt` for this, you can use `create_list_of_images.py` in this directory "`C:\Users\Name\Desktop\darknet\build\darknet\x64`".
* Then, you must create files named `obj.data`, `obj.names` in "`C:\Users\Name\Desktop\darknet\build\darknet\x64\data\`" as shown images.


Content of the file `data/obj.names` should be
```
NOK # name of first class
Kon # name of second class
              # third
              # all of them
```

Content of the file `data/obj.data` should be
```
classes = 1 #Number of classes 
train = C:/Users/Name/Desktop/darknet-master/build/darknet/x64/data/train.txt # Location of train set
valid = C:/Users/Name/Desktop/darknet-master/build/darknet/x64/data/valid.txt # Location of validation set, during test must be test.txt
names = C:/Users/Name/Desktop/darknet-master/build/darknet/x64/data/obj.names # Location of obj.names
backup = C:/Users/Name/Desktop/darknet-master/build/darknet/x64/backup # Location of saved weights
```

<img src="/docs/dataset1.png" alt="Dataset"/>
<img src="/docs/dataset2.png" alt="Dataset"/>

## Starting the Training

Everything is OK, no more obstacles to start training!

* If you use Anaconda, you open the `Anaconda Prompt` or if don't use, open the CMD.

* Powershell can also be used. I don't prefer.

* Now, if you don't open powershell, need to do one extra step. Move `tee.exe` where scripts to directory "`C:\Windows\System32`".

* Open the prompt and go to the directory where the repo is located with cd.
```
C:\Users\Name>cd C:\Users\Name\Desktop\darknet\build\darknet\x64
```
After, write and click enter
```
# This, starting the training with pretrained weight. Recommend! You move the pre-trained pretrained_model/yolov4.conv.137 from the repository to C:\Users\Name\Desktop\darknet\build\darknet\x64 in the repository.
 
darknet.exe detector train data/obj.data cfg/yolov4-custom.cfg yolov4.conv.137 -dont_show -map | tee results.log

# If you want to start from scratch try this 

darknet.exe detector train data/obj.data cfg/yolov4-custom.cfg -dont_show -map | tee results.log 

# If you want to continue where you left off if the training is interrupted, do this.

darknet.exe detector train data/obj.data cfg/yolov4-custom.cfg /backup/yolov4-custom_last.weights -dont_show -map | tee results.log
```
* Will be created training record with the `tee.exe`file above in "`C:\Users\Name\Desktop\darknet\build\darknet\x64`"

## Evaluate of Results

* For this, you must export the values ​​given in certain iterations in the result.log file to excel as shown images.

<img src="/evaluate/eval1.png" alt="Evaluate"/>

* Given `example.xlsx` excel table and `graph.ipynb` in evaluate folder for this.

<img src="/evaluate/eval2.png" alt="Evaluate"/>

<img src="/evaluate/eval3.png" alt="Evaluate"/>

* These are training values. We also need test values. For this, enter obj.data and make `valid.txt` to `test.txt.`

Content of the file `data/obj.data` should be for test results
```
classes = 1 #Number of classes 
train = C:/Users/Name/Desktop/darknet-master/build/darknet/x64/data/train.txt # Location of train set
valid = C:/Users/Name/Desktop/darknet-master/build/darknet/x64/data/test.txt # Location of test set
names = C:/Users/Name/Desktop/darknet-master/build/darknet/x64/data/obj.names # Location of obj.names
backup = C:/Users/Name/Desktop/darknet-master/build/darknet/x64/backup # Location of saved weights
```

* Then navigate to the "C:\Users\Name\Desktop\darknet\build\darknet\x64" directory on Prompt and run the code below.

<img src="/docs/conf1.png" alt="ConfMatrix"/>

```
#This creates mAP, recall,precision,f1-score and TP, FP, FN on test images. If you have one class, can create confusion matrix as shown images.

darknet.exe detector map data/obj.data cfg/yolov4-custom.cfg /backup/yolov4-custom_final.weight

#This creates prediction of test images as .txt. And, records in `\x64` folder.

darknet.exe detector test data/obj.data cfg/yolov4-custom.cfg /backup/yolov4-custom_final.weight -dont_show -ext_output < data/test.txt > result.txt

#This creates prediction of specified image and records as .jpg in `\x64` folder.

darknet.exe detector test data/obj.data cfg/yolov4-custom.cfg /backup/yolov4-custom_final.weight C:\Users\Name\Desktop\darknet\build\darknet\x64\test\sample.jpg
```

src="/docs/predict1.png" alt="Prediction"/>

src="/docs/predict2.png" alt="Prediction"/>


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
