instructions by Ross
# Using pretrained CNNs for image classification

In the previous assignments involving classification, we were performing a kind of simple feature extraction on images by making them greyscale and flattening them to a single vector. This vector of pixel values was then used as the input for some kind of classification model.

For this assignment, we're going to be working with an interesting kind of cultural phenomenon - fashion. On UCloud, you have access to a dataset of *Indo fashion* taken from this [Kaggle dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset). There is a paper which goes along with it on *arXiv.org*, which you can read [here](https://arxiv.org/abs/2104.02830).

Your instructions for this assignment are short and simple:

- You should write code which **trains** a classifier on this dataset using a *pretrained CNN like VGG16*
- *Save* the *training* and *validation* **history plots**
- Save the **classification report**

## Tips

- You should not upload the data to your repo - it's around 3GB in size.
  - Instead, you should *document in the README file where your data comes from*, how a user should find it, and where it should be saved in order for your code to work correctly.
- The data comes *already split into training, test, and validation datasets*. You can use these in a ```TensorFlow``` data generator pipeline like we saw in class this week - you can see an example of that [here](https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator).
- There are a lot of images, around 106k in total. Make sure to reserve enough time for running your code!
- The *image labels* are in the *metadata folder*, stored as *JSON files*. These can be read into ```pandas``` using ```read_json()```. You can find the documentation for that online.

## Contributions
This code was written by memyselfandI but with inspiration help functions from class, this StackOverflow: https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
and this notebook https://www.kaggle.com/code/vencerlanz09/indo-fashion-classification-using-efficientnetb0.

### Get the data
The easiest is to go to the Indo fashion dataset by clicking this link: [Kaggle dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset) and press the "Download" button in the top right corner.
This will downlaod a zip-file called "archive.zip". Place this in the the 'in' folder and unzip it by typing the command `unzip archive.zip`