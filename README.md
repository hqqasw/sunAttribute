# sunAttribute
Multi-label Classification for Sun Attribute Dataset

How to run:
-------
1. Download the dataset from https://cs.brown.edu/~gen/Attributes/SUNAttributeDB_Images.tar.gz
2. Make a dir named "images" in "data/" and put the images in it
3. Run train.py to train a model (default is vgg16-based model)
4. Run test.py to test your trained model and get recall and precision

Code Structure:
------- 
* `train.py`: a script for training
* `test.py`: a script for testing
* `classifier/dataset`: parse the SUN Attribute Database
* `classifier/models`: different base models, including reset and vgg
* `classifier/utils`: some basic codes, including metrics, data transformer, data pre-processer and so on
* `classifier/trainer.py`: a trainer for training
* `calssidier/evaluator.py`: an evaluator for evaluation and testing
