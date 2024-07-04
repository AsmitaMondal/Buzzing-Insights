# Buzzing-Insights :honeybee: :leaves:

## Abstract :disguised_face: 📝

This project presents a comprehensive exploration of honeybee ecology and honey production, incorporating image data analysis, predictive modeling, and ecological inference. Through meticulous data collection and analysis, insights into the intricate interactions between honeybees, flowering plants, and honey production dynamics have been gained, with accuracies exceeding 90% for binary as well as multiclass classification models. 

Notably, image classification techniques using Convolutional Neural Networks facilitated the identification of pollen-carrying honeybees with an accuracy of 91%, while predictive models for honey price estimation, particularly utilizing XGBoost, demonstrated exceptional performance with an R2 score of 0.99. Further, comparative studies were implemented for regression models as well as dimensionality reduction effects for optimization of the algorithms. 

The findings from this project hold significant implications for biodiversity conservation, sustainable agriculture practices, and the overall sustainability and profitability of the honey production industry. The project was carried out in the Kaggle Notebook powered by the TensorFlow GPU. 

## Team :couple: :people_hugging:

This project which was done by a team of 2 contains the entire documentation of the process. The links to the respective notebooks can be found in the documentation PDF itself. 

## Objectives  🥅

 The project aims to achieve the following objectives:
 - Explore the honey production dataset to understand its structure, variables, and trends.
 - Predict both honey purity and price using linear regression models, assessing their accuracy and performance.
 - Compareand contrast the performance of various regression models `(Lasso, Ridge, XGBoost, Gradient Boost)` with linear regression for predicting honey purity and price, evaluating their strengths and weaknesses.
 - Classify pollen and non-pollen carrying honeybees based on images using `Convolutional Neural Networks`.
 - Compareand contrast the performance of models trained on `dimensionality-reduced images` with those trained on the original dataset, assessing their predictive accuracy and computational efficiency.
 - Apply `logistic regression` to identify the sex of bees interacting with certain plants and determine the collection method of the bees, analyzing their associations with different variables.

 ## Requirement Specifications :luggage: :computer:
 
 ### Software: 🧮
 - Python with necessary libraries installed- Kaggle Notebook
 - TensorFlow GPU accelerated
   
 ### Hardware: :computer_mouse:
 - RAM: 8GBorabove
 - CPU: 8th Generation Intel® Core™ i3 Processor or above
 - GPU: 8GBor above
 - OS: 64-bit
 - Graphics Card: Integrated

## Summary of Findings 	:mag_right: 📜

### Pollen-Carrying Honeybee Image Classification :camera: :bee:

 The [Honey Bee Pollen](https://www.kaggle.com/datasets/ivanfel/honey-bee-pollen) dataset from Kaggle was used as a resource.

The entire flow of code can be found in the [documentation](https://github.com/AsmitaMondal/Buzzing-Insights/tree/main/Documentation).

 - The test accuracy of the model without `PCA` was **91.63%** which shows an incredible
 learning rate and performance by the model.
- The test accuracy of the model with PCA was **84.18%** which is also remarkable.
 - The time taken for the model to train on data without PCA was approximately **347.88**
 seconds.
- The time taken for the model to train on data with PCA was **2.53** seconds approximately.
  
 The slight decrease in test accuracy observed for the model with PCA preprocessing, achieving
 **84.18%**, compared to the model without PCA, which reached **91.63%**, can be attributed to
 several factors. ⏬
1. Firstly, PCA involves dimensionality reduction, which compresses the original
 high-dimensional image data into a lower-dimensional space by capturing the most important
 variations. While this reduction in dimensionality can lead to computational efficiency and
 potentially mitigate overfitting, it may also result in a loss of some information. 🖥️
2. Secondly, PCA operates on the assumption that the most significant
 variations in the data can be captured by linear combinations of the original features. However,
 complex and nonlinear relationships present in image data may not be fully captured by linear
 transformations alone. As a result, the PCA-transformed features may not fully represent the
 intricate patterns and structures present in the original images, leading to a slight decrease in
 classification accuracy. :chart_with_upwards_trend:

However, it was found that the training time for PCA was indeed much
 less compared to the other model. This fulfils the purpose of dimensionality reduction with
 respect to time complexity enhancement. ⏳

 ### Honey Price and Purity Prediction 💸

  The [Predict Price and Purity of Honey](https://www.kaggle.com/datasets/stealthtechnologies/predict-purity-and-price-of-honey) dataset was taken from Kaggle. 

  The entire flow of code can be found in the [documentation](https://github.com/AsmitaMondal/Buzzing-Insights/tree/main/Documentation).

For Price Prediction: 💰
- Our analysis revealed that boosting models excelled amongst the five regression models evaluated.
- Notably, the `XGBoost Regressor` demonstrated exceptional performance, achieving a testing MAE (Mean Absolute Error) of **0.011929**, surpassing the `Gradient Boost Regressor's` testing MAE of **0.012565**. This translates to impressive accuracy in both training and testing phases, solidifying `XGBoost Regressor` as the optimal model for this task.

For Purity Prediction 🍯
- Our analysis once again revealed that boosting models excelled amongst the five regression models evaluated.
- Notably, the `XGBoost Regressor` demonstrated exceptional performance, achieving a testing MAE (Mean Absolute Error) of **0.003413**, surpassing the `Gradient Boost Regressor's` testing MAE of **0.018926**. 

### Bee- Plant Interaction Analysis 🐝🪴


The [Bee-Plant Interaction Dataset](https://figshare.com/articles/dataset/Dataset_of_wild_bees_and_their_forage_resources_along_livestock_grazing_gradient_of_northern_Tanzania/21550545/3) was used for the analysis.

The entire flow of code can be found in the [documentation](https://github.com/AsmitaMondal/Buzzing-Insights/tree/main/Documentation).

- For all three datasets with missing value imputation, without missing value imputation and removal of missing values- the accuracy of the `Logistic Regression` model to classify the sex of the bee was 95%. This implies that the effect of imputation or removal did not affect the overall model building. In other words, the dataset was extremely well formed to work even without much preprocessing required. The mean cross validation score was 0.94. The individual fold scores were  **[0.94805195 0.94981413 0.94981413 0.94795539 0.89962825]**. 🥬❓
- The same pattern was seen for collection method classification using `Logistic Regression`. The training data achieved an accuracy of 98% while the testing data achieved an accuracy of 97%. The model displayed excellent performance without any overfitting or underfitting. The mean cross validation score was 0.98. The individual fold scores were  **[0.97588126 0.98327138 0.99442379 0.96096654 0.97769517]**. 🍀 🔬
- In a similar manner, to classify bee families based on species name, the training and testing data for all the three copies achieved a remarkable accuracy of **99.50%**. This is because the family of bee species becomes obvious when the species name is known. Both `Decision Tree` and `Random Forest` classifiers performed the same here. The cross-validation accuracy for both the models was **77.58%**. 🐝🍯

**K-MODES ANALYSIS** 🔍

1. For grazing intensity clustering, the KModes was run from a range of 1 to 5 clusters. The `Elbow method` showed inertia slow-down at k=3. In other words, the graph formed an elbow at k=3. This result was verified using the `silhouette score` which was the highest for 3 clusters (0.1898). 
This aligned with our dataset which had 3 categories for grazing intensity: *low, medium and high*. 🍀

2. For season clustering, the KModes was run from a range of 1 to 5 clusters. The `Elbow method` showed inertia slow-down at k=3. In other words, the graph formed an elbow at k=3. This result was verified using the silhouette score as well as the `silhouette graph`. Even though the silhouette score for k=3 was lesser than k=2, the visualization depicted that the cluster widths as well as the criteria for the clusters to lie above the average line was fulfilled by k=3 in a better way than k=2. A clustering solution with a lower silhouette score exhibiting better visual characteristics in terms of cluster separation, compactness, and class width, is a more suitable solution, despite the lower silhouette score.
This aligned with our dataset which had 3 categories for season: *dry season, short rain, long rain*. ⛈️

## Future Scope 🎊

- Incorporating interdisciplinary approaches that integrate advanced machine learning techniques with ecological and environmental datasets could offer new insights into honeybee ecology, pollination dynamics, and honey production systems. 💻
- Exploring emerging technologies such as remote sensing and drone-based monitoring could provide novel opportunities for monitoring honeybee populations and their interactions with the environment, ultimately contributing to the conservation and sustainability of pollinator ecosystems. 🤖

