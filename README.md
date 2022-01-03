# InnovationLab3

This program is a Face/Gender/Age detector based on Yolov5. The Gender detection is based on the CelebA Dataset and the Age Detection is based on the Adience Dataset.
Tested and written on python 3.8.10

## Installation:

```pip install -r requirements.txt``` 

Add the following files to the Models folder:

Gender Detection Model: [Drive Link](https://drive.google.com/file/d/1H8UzJURLl69GGZC9ZA9zZ1DSbCFP-55I/view?usp=sharing)

Face Detection Model: [Drive Link](https://drive.google.com/file/d/1L9CubLbwRkUPFh4rh9KnTeoSNKFrcNeO/view?usp=sharing)

Age Detection Model: [Drive Link](https://drive.google.com/file/d/1p3vxO-FtOwe-I_LECCB6CJuRzGp-EVv7/view?usp=sharing)

## Execution:

to run the program:

```python detect.py```

to enable Gender/Age Detection add:

```--gen_det True --age_det True```

Webcam Source can be changed by adding the number of your desired Input-device(x):

```--source x```

