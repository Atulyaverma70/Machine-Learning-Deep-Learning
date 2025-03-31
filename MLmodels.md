# Models of Machine Learning
Machine learning is a branch of AI that enables computers to learn from data and give prediction on new data.
## Linear Regression
* Linear regression is a type of supervised machine learning algorithm that computes the linear relationship between the dependent variable and one or more independent features by fitting a linear equation to observed data.
* It is used for regression task.
* residual is actual value minus predicted value.
* **goal in this-** to find the best fit line.
*  best fit line is that which has minimum error.
* **USE-** House Price Prediction, sale forecast.
* how can you mesure error in this?
    * Mean squrare Error - formula same as variance
    * Mean asolute error 
    * Root mean squre error- formula same as standard deviation


## Decision Tree
* is a type of supervised learning algorithm used for both classification and regression task.
* Decision Tree has flow chart like structure used to make decisions.
* **Working:** Splits data recursively based on feature values to reduce impurity in the subsets, creating branches until each leaf has a homogeneous outcome.
* **Splitting Criteria:** Gini Impurity, Entropy/Information Gain, Variance Reduction
* It has disadvantages of overfitting.
* Evaluation Metrics used to find error in Regression: MSE, MAE, RMSE
* Evaluation Metrics used to find error in classification: Accuracy, Precision, Recall, F1 Score (for classification), Mean Squared Error (for regression).
* spliting done using gini index **(range 0 to 0.5)** and Entropy(information gain)**(range 0 to 1**).
    * gini index=1-p^2 (used when classes are imbalanced)
    * Entropy=-plog(p) (used when useful when understanding uncertainity is important)
* **Real life use-** Medical Diagnosis, Product Recomendation.

## Random Forest
* Random forest is an ensemble learning method **(combines multiple weak model to create a stronger)** that combines multiple decision trees to improve accuracy and reduce overfitting.
* It use Bagging technique to reduce the overfitting.
    * Bagging (Bootstrap Aggregating) is a technique that trains multiple weak models independently (often decision trees) on different random subsets of the data, then combines their outputs to produce a more accurate and stable result.
* **Working:**
    * Training Phase: Multiple decision trees are created independently, each making its own predictions.
    * Prediction Phase: For classification, the final output is the majority vote across all trees. For regression, it’s the average of the tree outputs.
* **Common Applications:**
    * Financial Modeling: Fraud detection, credit scoring.
    * Healthcare: Disease prediction, medical diagnosis.
    * E-commerce: Customer segmentation, product recommendation.
    * Agriculture: Crop prediction, soil quality analysis.

## AdaBoost
* AdaBoost (Adaptive Boosting) is a machine learning technique that combines several simple models (called weak learners) to create one strong model that makes more accurate predictions.
* Key Concepts in AdaBoost (Simplified):
    * Sequential Training: Models are trained one by one, each focusing on correcting the mistakes of the previous model by giving more weight to misclassified points.

    * Weight Adjustment: Weights start equal, then increase for misclassified points and decrease for correctly classified ones, making the model pay more attention to difficult cases.

    * Model Combination: Each model's contribution to the final prediction is weighted by its accuracy, with the overall result being a weighted majority vote.

    * Mathematical Concept: AdaBoost minimizes errors based on weighted instances, creating a model that combines each weak learner's contributions to improve accuracy.
* Applications- face detection, text classification(spam detection), Medical diagnosis.

## Logistic Regression
Logistic regression is the machine learnig model used for classification task.
* It uses sigmoid function to predict the output.
* Decision Boundary: A threshold (often 0.5) is used to decide the final class label. If the output probability is greater than 0.5, it's classified as one class (e.g., 1), otherwise as the other (e.g., 0).
* Logistic Regression uses a cost function (log-loss) to measure the error between predicted probabilities and actual labels.
    * This cost function is minimized using algorithms like Gradient Descent to find the best parameters.
* Application- email spam classfication.

## Support Vector Machine(SVM)
* It is the supervised machine learning algorithm used for classification task.
* Support Vector Machine (SVM), the primary goal is to find the best hyperplane that divides the data points of different classes as distinctly as possible.
* Applications- Image Classification(used to categorize cat and and dog).

## K-nearest neighbour
It is a supervised machine learning algorithm used for classification task.
* **approach**- In KNN, when a new data point arrives, the algorithm calculates the distance between this point and all existing points in the dataset. It then identifies the K nearest neighbors (closest points) and assigns the new data point to the class (or predicts a value) based on the majority class of these neighbors (or average of values for regression).
* In this we use Ecludian Distance, Manhatten distance,etc.
* Applicaations- Recommendation system, Diesease prediction.

## K means clustering
It is the unsupervised learning it is used for clustering a dataset in K distinct groups.
* Key Concepts:
   *  K: The number of clusters the algorithm should form. This needs to be predefined.
   * Centroid: The center of a cluster, often calculated as the mean of all points in that cluster.
   * Euclidean Distance: A common distance metric used to determine how far points are from the centroids.   
* Aplication- anomaly detection, Image compression.

## Collaborative filtering 
It is the unsupervised machine learning technique commonly used in recommendation systems 
* how it work-
    * data collection
    * Similarity Calculaition
    * Prediction
    * Recomendation
* similarity measures
    * Cosine similarity- high cosine value indicates high similarity.
    * Ecludian distance- smaller distance indicates high similarity.
    * Pearsen Correlation- value closer to +1 indicates strong positive correlation.
* Application-
    * Movie Reccommendations
    * Ecommerce
    * Social media
* It has sparsity issue when data is insufficient.

<hr> 

## Gradient Boosting
It is Machine learning technique that is used for building predictive model. It is ensemble learning method 
* How it works?
    * sequential learning
    * Algorithm optimize the loss functions like-
        * mean square error regression 
        * log loss for classification
* Key features
    * Focus on optimizing errors
    * Customizable loss functions
* Advantages
    * high predictive accuracy
    * work well with structured and unstructured data
    * can handles complex relationship in data
* disadvantages
    * can slow in large dataset
    
<hr>

## XGBoost
* It is optimized machine learning algorithm based on the Gradient Boosting framework.
* it is widdely used for supervised learning tasks.
* It is not an algorithm.
* It is Library that is based on gradient Boosting.
* Key features
    * Do Parallel processing : Utilizes CPU and GPU for faster computations.
    * Handling issing Data 
* Used for common uses:
    * Ranking 
    * Classification 
    * Regression

## Difference between Gradient boosting and adaboost
* Gradient boosting focuses mainely on minimizing the loss to get better output.
* Adaboost is focuses on reweighting samples based on there error.


## Extra Notes

* **Ensemble Learning method is two type-**
    * Bagging- Is an technique that train multiple weak model independently and result is average of all.
        * used in Random forest.
    * Boosting- Builds models sequentially, with each new model focusing on correcting errors made by the previous one.
        * used in adaboost.
* **Why Ensemble Learning Works**
    * Reduces bias
    * Reduces Variance
    * Improves Generalization- means work better on unseen data.
*  **Applications of Ensemble Learning**
    * Fraud detection
    * Image and text classification
    * Predictive analytics (e.g., in finance and healthcare)
    * Recommendation systems

## Activation function
Activation functions introduce non-linearity into the model, enabling it to learn complex patterns, which ultimately helps in reducing prediction errors.
* Sigmoid function
    * range 0 to 1
    * s- shape curve
    * used for binary classification

* Softmax function
    * range 0 to 1
    * its multiclass extension of sigmoid
    * used for multiclass classification
    * slow for large data

* Tanh(hyperbolic curve)
    * range -1 to 1
    * s-shape cureve
    * used as hidden layer in neural network

* ReLU(Rectified Linear)
    * range 0 to infinite
    * curve is linear for positive value and zero for negative
    * used as hidden layer in deep networks

* Leaky Relu
    * range -infinite to infinite

## Confusion Matrix
It is a essential tool for analyzing performance in classification model.

* Accuracy= TP+TN/TP+TN+FP+FN
* Precision= TP/TP+FP(proportion of positive prediction that are actually currect)
* TPR or Recall= TP/TP+FN (proportion of actual positive that were correctly identifies)
* FPR=FP/FP+TN
* F1-score=2*(Precision*Recall)/Precision+Recall

* log-loss= -1/n(ylogp+(1-y)log(1-p))

## Z score
* Z=(X(data point)-μ(mean))/sd
* It is use to detect the outlier or anomalies.
* if Z-score is greater then +3 and less the -3 then it is outliear

* why we use mean and median in that?
    * it tells the central tendency of data


# feature extraction technique
## PCA (principal component analysis)->unsupervised machine learning technique
* Principal Component Analysis (PCA) is a dimensionality reduction technique used in machine learning and statistics to simplify large datasets while preserving essential patterns and structures. It is widely used for feature extraction, noise reduction, and visualization of high-dimensional data.
* Benifits
    * faster execution of algo
    * visualization
* Applications
    * image compression
    * Noise reduction
    * feature extraction
    * Data visualization
* steps to perform that 
    1. standardize the data
    2. finds the covarience matrix
    3. finds eigen value and eigen vector
    4. sort eigen value in descending order and choose top k eigen vectors

# feature selection
Extracts the some important feture from feture.
* why we need this
    * curse of dimensionality
    * computational complexity
    * iterpretability
* they are different types:
    * filter based technique
        1. dupliate fetures(drop the duplicate features)
        2. varience threshold(worked on two features constant and quasi constant)
        3. correlation technique(pearsen correlation cofficient technique)
            * its range between -1(string inverse linear relation ship) to 1(strong positive relationship)
        4. ANOVA(used when input is numerical column and output is categroical)
        5. ch-square(used when both data categorical input and output)
    


