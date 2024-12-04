# CNN Creator and Classifier

## **Details**
- **Author**: Thomas Fairfield
- **Course**: ARTI 405: A.I. Architecture & Design
- **Instructor**: Professor Gogolin
- **Semester**: Fall 2024

## **Additional Details**
- **Project Objective**: Produce a streamlit web interface for creating Convolutional Neural Nets and using them to classify
- **Technologies**: Python
- **Libraries**: streamlit, tensorflow+cuda, numpy, matplotlib, pillow

## Installation and details

Windows terminal:

`wsl --install`

navigate to the directory with py file and datasets
install python3.10.12
using pip install streamlit, tensorflow+cuda, numpy, matplotlib, pillow

`streamlit run classifiercreate.py`

open browser and connect to localhost:8501 or 127.0.0.1:8501

select a folder with an image dataset.  Structure should be:  ./dataset/class1/images0... and ./dataset/class2/images0... for every class in the dataset.

use preprocessing to crop images to set resolution, or just use training to trainat selected input size and last layer neuron count.

use left panel to switch to classify and load modelimages are scaled to the model's input size automatically, classes are labeled 0,1,2,3,... 
