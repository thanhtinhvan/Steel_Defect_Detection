# Steel Defect Detection

This is a DL project about Steel Defect Detection. I design to join the Kaggle competition: [Steel defect detection](https://www.kaggle.com/c/severstal-steel-defect-detection)
![OverallResult](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/0.jpg)
The data can get from the above competition link.

This project uses the [Mask R-CNN](https://github.com/matterport/Mask_RCNN) framework to detect steel defects in images

## Mask R-CNN Implementation
At this time (end of 2017), Facebook AI research has not yet released their implementation. [Matterport, Inc](https://matterport.com/) has graciously released a very nice python [implementation of Mask R-CNN](https://github.com/matterport/Mask_RCNN) on github using Keras and TensorFlow. This project is based on Matterport, Inc work.

## Sample images and my result:
There are 4 classes:
- Dot
![dot](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/class_dot_1.jpg)
- Line
![line](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/class_line_2.jpg)
- Surface Scratch
![surface](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/class_surfaceScratch_3.jpg)
- Deep Scratch
![deep](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/class_deep_4.jpg)

## Generate mask using Encode pixels:
In the dataset, the defect area was encoded in only 1 excel file. 
For example:
```
202054 10 202310 30 202565 51 202821 70 203076 91 203332 110 203587 131 203843 151 204098 162 204354 162 
204609 163 204865 163 205120 164 205376 164 205643 153 205919 132 206195 112 206471 92 206748 71 207024 51 
207300 31 207576 11
```
The image size is (1600x256x3). So we have 1600x256=409600 pixels location.
The first number is the pixel id and the next one is the length of the next pixel which consider as mask area. For example above "202054 10" mean pixel id from 202054, 202055, ...., 202064 (10 pixels) are considered as mask. 
Then if all pixel id we change value to black-value. From original, we will have:
![encode](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/encodePixels.jpg)

## Training
Training starting from using MS COCO model. 

## Results:
Model able to detect, classify the defect. The using visualize to draw exactly what model detect (not only bounding box).
![0](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/0.jpg)
- Surface scratch
![1](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/1.jpg)
- Surface scratch
![4](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/4.jpg)
- Surface scratch
![5](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/5.jpg)
- Surface scratch
![11](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/11.jpg)
- Surface scratch
![14](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/14.jpg)
- Line scratch
![12](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/12.jpg)
- Line scratch
![13](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/13.jpg)
- Line scratch
![2](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/2.jpg)
- Line scratch
![6](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/6.jpg)
- Line scratch
![7](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/7.jpg)
- Surface scratch and line scratch
![10](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/10.jpg)
- Deep defect
![3](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/3.jpg)
- Deep defect
![8](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/8.jpg)
- Deep defect
![9](https://github.com/thanhtinhvan/Steel_Defect_Detection/blob/master/Screenshots/9.jpg)

## Installation
### Requirement:
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in requirements.txt.
### Setup
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.
Install dependencies
```bash
pip3 install -r requirements.txt
```

## Usage
Open and run SDD.ipynb in jupyter notebook. It combined steps to train and test the model.

## Contributing and request
Pull requests are welcome. For major changes or can not run, please open an issue first to discuss what you would like to change or your issue. 
