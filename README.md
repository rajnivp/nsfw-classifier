# nsfw-classifier
## Description
This classifier is trained on resnet50 architecture for 30 epochs and it took 3 hours to train it 
with ~90% accuracy on my machine
## Prerequisites

- Python 3.x
## How to train a model
- Install and activate virtual enviroment
- Run pip install -r requirements.txt
- Run downloader.py to collect data for different folders
- Run train.py
## How to test a model
- Run plotting.ipynb to plot test probabilities
- To test on single image run following command<br>
  test.py path-to-image<br>
  alternativaly to test images on gui see below example<br><br>
![example](https://github.com/rajnivp/nsfw-classifier/blob/master/assets/classifier.gif)
