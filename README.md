# Earthquakes Forecast System Data Mining Project

## Overview 
Earthquakes are a major disaster and unpredictability leads to more destruction in terms of human life and material losses. There was a serious debate about predictions of two simultaneous perspectives and predictability of earthquakes. While a group of researchers spent their resources and efforts on this task, the other group of researchers saw it as impossible to predict. It is a fact that the seismologist community has not been able to develop methods of predicting earthquakes despite more than a century of efforts. Researchers and scientists from around the world have adopted different new and hybrid algorithms using artificial intelligence techniques in the last decade. They were used to predict earthquake challenges such as depth, location, size and time. These algorithms and techniques may include conventional methods or may create new algorithms with a mixed technique in combination with nature-inspired algorithms. Popular methods include: `Pattern Recognition Neural Network (PRNN)`, `Recurrent Neural Network (RNN)`, `Random Forest` and `Linear Programming Boost (LPBoost)` ensemble decision trees community for earthquake prediction. 

The core idea of this work is to predict earthquakes of magnitude 0.1 and 7 using Random Forest Algorithm approaches. A model will be created using our training kit using random forest machine learning technique. The number of decision trees will be determined in order to calculate the performance of this model and to reach an appropriate level. The parameters to be used in this method are time, latitude, longitude, size and degree. Finally, recall, precision, f-measure and matrix confusion, which are classification metrics, will be calculated. 

---

## Starting
A `jupyter notebook`, a server client application, was used to edit and run the codes. In our project python language is used and Pandas, an open source BSD library that provides data structures and data analysis tools from python libraries, is also used. Pandas allows you to open csv and text files and read the data in it.

---

## Description of Various Datasets
The dataset given by Kaggle consists of 8 columns and approximately 24004 million rows.

![veriornek](https://user-images.githubusercontent.com/48765779/77943205-87c24100-72c5-11ea-9ed8-0c01f26ff09a.png)

---

## Project Library
``` r
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
%matplotlib inline
```
---

## Data Pre-Processing
`Degree` feature is added as an additional column to the data. 
The degree feature is assigned data to express the connection with the magnitude data.
The average of the magnitudes corresponding to each degree was calculated.
The data is divided into two data sets: `train` and `test`.
In the datasets, properties with a value of 0, empty or not used are deleted (index, no, date, time, place).
`Degree` property has been deleted from test dataset.

---

## Data Visualization
The frequency of the magnitude data was calculated and plotted the graph.
`Degree` datas plotted as boxplot.
Latitude and degree data are grouped according to magnitude data.

---

## Modelling
`Random forest` was used as classifier. One of the major problems of decision trees, which is the traditional methods, is overfitting. To solve this problem, the random forest model selects and sets different sub-sets from the data set. With this method, many decision trees are formed and each decision tree makes an individual estimation. For classification problems, the most votes are selected from these estimates.

![random forest](https://user-images.githubusercontent.com/48765779/77943227-914ba900-72c5-11ea-9ccf-5ff522299a9b.jpg)

---

## Performance Metrics For Classification
Finally, the desired confusion `matrix`, `accuracy`, `recall`, `precision` and `f1 score` were calculated. In order to calculate these values, `sklearn.metrics` module is used and various python libraries are added to the code (confusion_matrix, accuracy_score, precision_score, f1_score). 

---

## Structure of The Solution Proposed-Components
![data](https://user-images.githubusercontent.com/48765779/77942600-83495880-72c4-11ea-84e3-c39db479528a.png)

---

## Results and Their Interpretation
This study presents a random forest model which can predict the magnitude of the earthquake. The whole magnitude range was divided into four classes: 0.49 to 3.5, 3.5 to 4.3, 4.3 to 5.3, 5.3 to 6.5 and 6.5 to 7.5, respectively. Decision trees are supervised machine learning techniques and have been used for classiﬁcation and regression purposes. Random forest refers to the large number of decision trees, merged through bootstrap aggregating or bagging. The choice of number of decision trees is decided on the basis of experimentation on dataset. In our case, 360 numbers of trees have been chosen for creating ensemble on the basis of experimentation. Since majority of earthquake events in the present dataset corresponds to class “Moderate” (earthquakes of having magnitude between 3.5 and 4.3), there is a significant level of class imbalance present in the dataset and sample number of class `Great` and class `Super` are smaller than other classes. As a result of our report, the accuracy rate of our model, which we used Random Forest classification, was `0.610997`.

---

## Project Team
- [ ] <a href="https://github.com/iremnazcay" target="_blank">İremnaz ÇAY</a>  
- [ ] <a href="https://github.com/sinemertass">Sinem Sena Ertaş</a>
- [ ] <a href="https://github.com/mervebosna" target="_blank">Merve Boşna</a> 
