# Image Classification

_**Ensemble Learning**_

Ensemble learning is one of the various prototypes of deep learning which is advantageous in various machine learning applications. Ensemble method can be thought of as a machine learning model that involves a set of various individual models working parallelly on the same dataset to produce a better accuracy result as compared to any of the individual models working alone . This ensemble model comprises of three individual models – Convolutional Neural Network(CNN), Long Short Term Memory networks(LSTM) and Multi-layer Perceptron Network (MLP).. We would be using this powerful tool to categorise images in the Fashion MNIST dataset which comprises of 60,000 training dataset images and 10,000 test images.

_**Data Set**_

1. Fashion MNIST dataset comprises of 60,000 test set images each of which has 784 features (i.e. 28×28 pixels).
2. Each pixel has a value between 0 to 255 where 0 is for white and 255 signifies black.
3. All the images in the dataset have light gray background((hexadecimal color: #fdfdfd) and the products fall in different categories like -  men, women, kids and neutral.
4. All the images of Fashion MNIST dataset  are classified into one of the following 10 categories - t-shirts, trousers, pullovers, dresses, coats, sandals, shirts, sneakers,     bags, and ankle boots.

_**CNN Architecture Used**_

1. The architecture we experimented with consists of 3 layers: 2 Convolutional, 2 Pooling and 2 Fully connected layers.
2. Filters of dimensions 5x5 and 3x3 are used in the 2 convolutional layers to produce feature maps.
3. ReLU activation function is used in these two layers to add non-linearities. 
4. Each convolutional layer is followed by a Max Pooling layer to downsample the data.
5. A dense layer of 128 neurons with ReLU activation is added to make the network more sophisticated. 
6. A fully connected layer with softmax activation serves as the output layer.

_**Long Short-Term Memory (LSTM)**_

1. LSTM models have evidently outperformed RNNs on learning context-free and context-sensitive languages.
2. The key difference between traditional RNNs and LSTMs is that later can avoid the long-term dependency issue and work for either long or short time spans
3. An LSTM unit is composed of a cell, an input gate, an output gate and a forget gate.
4. The cell is responsible for collecting the value and the three gates modulate the flow of information into and out of the cell.
5. LSTM networks are expedient to carry out classification, processing and prediction chores based on time series data.

_**MLP Architecture Used**_

1. Each input vector is mapped to a class label, belonging to one of the categories. Since MLP needs the input as a vector, images need to be pre-processed. 
2. Each 28x28 grayscale image has 28*28 = 764 pixel values. 
3. Therefore, each image is flattened and 764 features contribute as the input vector for the MLP. 
4. The MLP model has 3 dense layers with 512 neurons and ReLU activation, each followed by a dropout layer to prevent overfitting. 
5. The output layer is a fully connected layer with softmax activation. 

_**Ensemble Voting Classifier**_

1. The three individual base models have softmax activation function in the output layer. The trained models are used to predict the probabilities of the 10 classes for the image data. For selecting the class predicted by the ensemble classifier, two different techniques are used: Hard voting and Soft voting. 
2. Hard voting predicts the class which was predicted by the majority of the base models. In the case when each of the models predicts a different class and there is no unique mode, the class with the highest probability is selected. 
3. Soft voting takes the mean of probabilities of each class for the three base models, and predicts the class with the highest mean. We used weighted mean of the probabilities for soft voting in order to vary the contribution of the individual models in the ensemble.

_**Softmax Function**_

1. The softmax function, also known as softargmax or normalized exponential function is a function that takes as input a vector of K real numbers, and normalizes it into a probability distribution consisting of K probabilities. 
2. That is, prior to applying softmax, some vector components could be negative, or greater than one; and might not sum to 1; but after applying softmax, each component will be in the interval and the components will add up to 1, so that they can be interpreted as probabilities.
3. Furthermore, the larger input components will correspond to larger probabilities.
4. Softmax is often used in neural networks, to map the non-normalized output of a network to a probability distribution over predicted output classes.

_**Experimental Results**_

1. The proposed algorithm was implemented on all the above mentioned corpora and the results obtained are discussed in this section. 
2. The experiments were performed using Python 3.6 on a Dell hex core (2.4 GHz.) laptop with Nvidia GeForce GTX 1060 GPU. 
3. Under the given conditions, experimental evidence shows the ensemble algorithm performed quite robustly in almost all the cases and outperformed the established deep learning models in terms of accuracy. 
4. The accuracy obtained by individual models on test data are: 0.8687 for MLP, 0.8904 for LSTM and 0.9141 for CNN.
5. The accuracy for Ensemble model using hard voting comes out to be 0.9064 while it reaches 0.9178 for weighted soft voting.

The experiments show that CNN performs the best among the individual base learners for image classification of the Fashion MNIST dataset, while MLP performs the worst. The hard voting technique which takes the majority of the predictions is better than LSTM and MLP, however doesn’t match up to the accuracy of CNN. The soft averaging technique is comparable to the CNN in terms of performance. The soft voting technique is modified with weights associated with each of the models in the ratio 1:1:2 for MLP, LSTM and CNN respectively. The modified architecture shows improvement in accuracy and even outperforms the CNN, which is the most practiced image classification technique. 














