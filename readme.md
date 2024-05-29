# Simple Adversarial Noise

This program computes adversarial noise to be added to an image such that a resnet50 ImageNet classifier misclassifies the image as a different class.

### Setup Environment
Using your choice of python interpreter (conda, venv, pyenv etc...) for python>=3.10 run:
```
pip install requirements.txt
```

### Run the program
To run the program pass the path to an input image and the target class. E.g.
```
python generate_adversarial_image.py --image_path=assets/elephant.webp --target_class=1
```
The program can be run with no arguments for a quick demo.
Additional optional arguments/parameterisations can be found via:
```
python generate_adversarial_image.py --help
```