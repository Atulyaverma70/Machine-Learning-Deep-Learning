# Deep learning
Deep learning is a subset of machine learning that uses neural networks with multiple layers to automatically learn and extract complex patterns from large datasets
## ANN(Artificial Neural Network)
* inspired by human brain.
* worked on fixed input size.
* **layers** it has one input and one output layer and muliple hidden layer.
* **Structure**- ANN are made up of neuron that take input data,apply weight,biases, and activation function.
* **Applications:** ANNs can handle tasks like classification, regression, and even basic feature extraction.

## CNN(Convolutional Neural Network)
Convolutional Neural Networks (CNNs) are a type of neural network specifically designed for processing and analyzing grid-like data structures, such as images and videos.
* there are three layers in this-
    * Convolutional layer- are filters(filters are used to extract features)
        * it is used to detect edges(change in intensity). we uses filters/kernel
        * we uses padding and stride also
             * **Padding** it involves adding extra pixels around the edges of the input images. If we dont use padding in image we loss edges information.
            * **Stride**  stride is the number of pixels the convolutional filter moves (or "strides") across the input image during convolution.
                * if we increse stride then it didnt capture low level feature.
        * from convolutional layer we get feature map
    * Pooling layer
        * why- to drop some features in the feature map.
        * this helps to prevent the network from becoming too large.
        * there are different types of pooling
            * max
            * Min
            * Average

    * fully connected layer(ANN)
* CNN is inspired from visual cortex(dimaag ka part hota hai jise use karke hum dekh pate hai)
* Application-
    * face recognition 
    * self driving cars

## RNN(Recurent Neural Network)
Its a type of sequential model use to work on sequential data.
* Work good in NLP related areas.
* Type of RNN-
    * Many to one
        * input sequence data
        * output is non sequential data->scaler(0,1)
        * used in sentiment analysis- movie review
    * one to many
        * normal non sequential data
        * output sequential data
        * application- image captioning
        * music generation
    * many to many
        * input sequential data
        * output sequential data
        * aplication- google translation
    * one to one 


* application-
    * sentiment analysis
    * used in e- commerce to check the reviews positive or neagative
    * predicting next word
    * image caption generation
    * google translation

## LSTM(Long short term memory)
Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture that is particularly effective for tasks involving sequential data, such as time series forecasting, natural language processing, and speech recognition.
* LSTMs have two states—cell state and hidden state—whereas RNNs have only a single hidden state.
    * The cell state maintains information over time, while the hidden state is used for calculations in LSTM.
* there are three gates in LSTM architecture-
    * Forget gate- Decide what information to discard from the memory cell.
    * Input Gate- Determines what new information should be added to the memory cell.
    * Output state- Controls what information is sent to the next hidden state.
* Applications- 
    * Test generation
    * Speech Recognition
    * language translation
    * Anomaly Detection











## extra Notes
* Difference between RGB and Grayscale(black and white)- is that RGB has three chanels and Grayscale has one chanel.
* If we use ANN for the task like NLP related we have to lots of Unneccessory computations that why we prefer RNN rather then ANN.
* If we use ANN in swquential data the problem occurs-
    * text input varying size.
    * zero padding- unneccessary coomputation.
    * Prediction problem.



    ## vanishing gradient problem
The vanishing gradient problem occurs during the training of deep neural networks when gradients (the updates used in backpropagation to optimize weights) become extremely small, approaching zero. 

## Batch Normalization
that help in speedup the neural network training.
* intermal covariate shift
    * change in distribution of network activation ue to the change in network parameter during training phase
* Advantages
    * hyper parameter more stable
    * training rate faster
    * regularizer
    * reduces the weight initialization impact


## how to improve the performance of a Neural Network
* by fine tuning the hyper parameters
    * by increasing no of hidden layers
    * by incresing the neurons per layer
    * increase the epoch

## problems with neural netwok
* vanishing graidient and exploading gradient
* Not enough data
* slow training 
* overfitting

## epoch
An epoch is one complete pass of the entire dataset through the neural network during training.

## early stopping
Early stopping is a regularization technique used in machine learning and deep learning to prevent overfitting by stopping the training process when the model's performance on a validation set starts to degrade.

## generators
* it is useful for large amount of data.
* It dividies the whole data in multiple chunks then train one by one data due to the less computational power and large amount of data.

## Data Augmentation
* it is a technique creating new data from existiong data.
* resuces the overfitting.


