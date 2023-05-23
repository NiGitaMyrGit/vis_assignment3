# Assignment 3 - Using pretrained CNNs for image classification
The portfolio exam for Visual Analytics S22 consists of 4 projects; three class assignments and one self-assigned project. This is the repository for the third assignment in the portfolio.
## 1. Contributions
The code was produced by me, but with a lot of problem-solving looking into various Stack Overflow blog posts.
worth mentioning is this StackOverflow: https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
And this notebook https://www.kaggle.com/code/vencerlanz09/indo-fashion-classification-using-efficientnetb0.
Furthermore the `def plot_history`and def `save_report` was provided by our instructor Ross during the course

## 2. Initial assignment description by instructor

In the previous assignments involving classification, we were performing a kind of simple feature extraction on images by making them greyscale and flattening them to a single vector. This vector of pixel values was then used as the input for some kind of classification model.

For this assignment, we're going to be working with an interesting kind of cultural phenomenon - fashion. On UCloud, you have access to a dataset of *Indo fashion* taken from this [Kaggle dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset). There is a paper which goes along with it on *arXiv.org*, which you can read [here](https://arxiv.org/abs/2104.02830).

Your instructions for this assignment are short and simple:

- You should write code which **trains** a classifier on this dataset using a *pretrained CNN like VGG16*
- *Save* the *training* and *validation* **history plots**
- Save the **classification report**

## 2.1 Tips

- You should not upload the data to your repo - it's around 3GB in size.
  - Instead, you should *document in the README file where your data comes from*, how a user should find it, and where it should be saved in order for your code to work correctly.
- The data comes *already split into training, test, and validation datasets*. You can use these in a ```TensorFlow``` data generator pipeline like we saw in class this week - you can see an example of that [here](https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator).
- There are a lot of images, around 106k in total. Make sure to reserve enough time for running your code!
- The *image labels* are in the *metadata folder*, stored as *JSON files*. These can be read into ```pandas``` using ```read_json()```. You can find the documentation for that online.

## 3. Methods

# 4. Usage
This script was made using python 3.10.7, make sure this is your python version you run the script in. 
### 4.1 Installing packages
From the command line:
Clone this repository to your console by running the command `git clone https://github.com/NiGitaMyrGit/vis_assignment3.git`. This will copy the repository to the location you are currently in.
Then make sure you are located in the main folder, location can be changed by using the command `cd path\to\vis_assignment3`'. From here run the command `bash setup.sh` which will install all the required packages in order to run the script.

### 4.2 Dataset
For downloading this dataset, the easiest is to go to the Indo fashion dataset by clicking this link: [Kaggle dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset) and press the "Download" button in the top right corner.
This will downlaod a zip-file called "archive.zip". Place this in the the 'in' folder and unzip it by typing the command `unzip archive.zip`
### 4.3 running the script

## 5. Results - discussion
