# InnovationLab3

This program is a Face/Gender/Age detector based on Yolov5. The Gender detection is based on the CelebA Dataset and the Age Detection is based on the Adience Dataset.
Tested and written on python 3.8.10

## Installation:

```pip install -r requirements.txt``` 

Add the following files to the Models folder:

Gender Detection Model: [Drive Link](https://drive.google.com/uc?id=1H8UzJURLl69GGZC9ZA9zZ1DSbCFP-55I&export=download)

Face Detection Model: [Drive Link](https://drive.google.com/uc?id=1L9CubLbwRkUPFh4rh9KnTeoSNKFrcNeO&export=download)

Age Detection Model: [Drive Link](https://drive.google.com/uc?id=1p3vxO-FtOwe-I_LECCB6CJuRzGp-EVv7&export=download)

## Execution:

to run the program:

```python detect.py```

to enable Gender/Age Detection add:

```--gen_det True --age_det True```

Webcam Source can be changed by adding the number of your desired Input-device(x):

```--source x```

