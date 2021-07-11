# pytorch-FCN-mvtec
This is a toy example of implementing fully-convolutional-networks (FCN) on mvtec dataset as a segmentation task.

### 1 - Dataset
For convenience, all images (17 images in tatal, 15/2 are used to train/test the model) and their ground-truth masks are contained within the folder "data/". In fact, these images are called mvtec-Carpet-Test-Color, an industrial defect segmentation dataset available from "https://www.mvtec.com/company/research/datasets/mvtec-ad".

### 2- How to run this toy project?
Open your terminal and then navigate the right directory. Then execute `python train.py` to complete the training and testing process. As long as evertything goes successfully, you will see in the folder "results/" those predicted  masks for the 2 testing images, which looks like:

epoch 1 (left: input images, middle: the ground-truth masks, right: the predicted masks)

<img src="https://github.com/suzyi/pytorch-FCN-mvtec/blob/main/results/epoch_1.png" width=300px/>

epoch 100

<img src="https://github.com/suzyi/pytorch-FCN-mvtec/blob/main/results/epoch_100.png" width=300px/>

epoch 200

<img src="https://github.com/suzyi/pytorch-FCN-mvtec/blob/main/results/epoch_200.png" width=300px/>

epoch 300

<img src="https://github.com/suzyi/pytorch-FCN-mvtec/blob/main/results/epoch_300.png" width=300px/>

### 3 - Reference
+ All .py files are borrowed from [github-bat67: pytorch-FCN-easiest-demo](https://github.com/bat67/pytorch-FCN-easiest-demo) but I have re-wrote the onehot function.
