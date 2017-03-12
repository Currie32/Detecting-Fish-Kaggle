# Detecting-Fish-Kaggle

This work is for the "The Nature Conservancy Fisheries Monitoring" Kaggle Competition: https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring. The objective of this competition is to predict which type of fish is in each image. The dataset is quite large, so if you want to download it, visit: https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data

I used two different models to make my predictions for this competition. Model1 is a deeper and wider convolution neural network (CNN), and model2 is shallower and narrower. Model1 uses a 80-20 train-split split, and model2 uses Kfold with five splits. Model2 typically performed better than model1. To create my best submission, I averaged the predictions of my best performing models. This is a type of ensemble, or "wisdom of the crowd" approach. Currently, I rank in the top 16% of competitions.

To view my models most easily, click on the .ipynb links.

Note: I used Keras to build my CNNs.
