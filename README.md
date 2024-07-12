# EMNIST
# Section 1-Introduction
## Brief Overview of The Problem:
In this project, we aim to assess the performance of Multilayer Perceptron (MLP) and Convolutional Neural
Networks (CNNs) using the EMNIST Balanced dataset, a carefully curated subset of the Extended MNIST dataset.
This dataset has been meticulously designed to ensure equitable distributions across various alphanumeric
characters, including handwritten digits (0-9) and uppercase letters (A-Z). Each image is represented as a pixel
value matrix, accompanied by corresponding labels indicating the category. With a substantial dataset
comprising 112,800 records in the training set, 18,800 records in the test set, and balanced classes spanning 47
categories, we've meticulously prepared the data by segregating dependent features and labels for both sets.
These datasets were converted to numpy uint8 type for compatibility and a validation set of 10,000 records was
extracted from the training data. Utilising one-hot encoding for labels, we normalised the dependent features
by dividing them by 255 for both MLP and CNN models. While the MLP architecture reshaped the data into
(number of records, 28, 28) format, the CNN architecture required a format of (number of records, 28, 28, 1) to
accommodate convolutional layers.
Python Libraries Used:
In our analysis, we harnessed a variety of Python libraries, leveraging specific functionalities from each as
detailed below:
1. Pandas: We employed Pandas for seamlessly reading the dataset stored in the .CSV format using the
‘pd.read_csv()’ function. This facilitated the conversion of our data into DataFrame objects, allowing for efficient
manipulation and exploration.
2. Matplotlib: Matplotlib played a crucial role in our analysis by enabling the generation of feature versus count
plots. These visualizations provided invaluable insights into the distribution of each feature within our dataset,
aiding in exploratory data analysis.
3. Seaborn: We utilised Seaborn for plotting the loss and accuracy curves for each of our models. This allowed
us to visually track the performance of our models over epochs, facilitating the identification of trends and
patterns.
4. Scikit-learn (or sklearn): Scikit-learn proved to be instrumental in our analysis by providing access to a plethora
of evaluation metric functions. These functions enabled us to conduct comprehensive analyses of model
performance and compare the performance of different models effectively.
5. Tensorflow, Keras: Extensively leveraging Tensorflow and Keras, we built our neural network models and
utilized these libraries for hyperparameter tuning. The flexibility and robustness of these libraries enabled us to
construct, train, and evaluate complex neural network architectures efficiently.
The Structure of MLP & CNNs:
Baseline Model for MLP:
The provided deep neural network architecture is tailored for image classification tasks. It begins with a flattening
input layer for 28x28 images, followed by four dense layers employing ReLU activation functions to capture
intricate patterns. These layers vary in units from 128 to 16, promoting feature extraction while controlling
complexity. Regularization techniques like L1 and L2 are applied to prevent overfitting, complemented by
dropout layers (10%-20%) after each dense layer. The final layer, with 47 units, utilizes softmax activation, offering
a probability distribution over classes. This design prioritizes balancing model complexity and generalization,
utilizing dropout and regularization for enhanced robustness against overfitting.
Baseline Model for CNN:
The provided code outlines a Convolutional Neural Network (CNN) model designed for image classification tasks.
It initializes a Sequential model object and starts with a convolutional layer (CONV_1) featuring 32 filters of size
3x3, ReLU activation, and 'same' padding to preserve spatial dimensions. Following this, a max-pooling layer
(POOL_1) with a 2x2 window size is added for down-sampling. Next, another convolutional layer (CONV_2) is
introduced, employing 64 filters of size 3x3 with ReLU activation and 'same' padding, succeeded by another max-
pooling layer (POOL_2). The model then flattens the output to prepare for fully connected layers (FC_1 and FC_2)
comprising 128 and 64 neurons respectively, both utilizing ReLU activation. Lastly, the output layer (FC_3)
comprises 47 neurons with softmax activation for classifying into 47 categories, mirroring the number of output
classes. The model summary provides a comprehensive overview of the architecture, detailing the parameters
and output shapes at each layer.
Section 2-Rationale of Design
2.1 Rationale Design of MLP:
Chosen Hyperparameters/Techniques for tuning:
In the MLP model building function, various hyperparameters and techniques are used for tuning and
exploration. These include the number of layers, neurons per layer, activation functions, and dropout rates,
3
offering flexibility for experimenting with different architectures. Activation functions like ReLU, Tanh, Leaky
ReLU, and ELU address non-linearity and vanishing gradient issues. Dropout regularization prevents overfitting
by randomly dropping neurons. L1 and L2 regularization, along with batch normalization, stabilize and accelerate
training. Optimization algorithms like Adam, SGD, RMSprop, and Adadelta are explored for efficiency and
convergence speed. Overall, these choices aim to enhance generalization performance and training efficiency.
Hyperparameter Tuning Results-best MLP model:
After tuning the hyperparameters, the best model that we got, is-
{'num_layers': 3,'neurons_0': 93, 'activation_0': 'leaky_relu', 'dropout_0': 0.7, 'regularization': False, 'batch
normalization': False, 'dropout': False, 'optimizer': 'adam', 'neurons_1': 8, 'activation_1': 'relu', 'dropout_1':
0.1, 'neurons_2': 8, 'activation_2': 'relu', 'dropout_2': 0.1}.
The model begins with an input layer that flatten the 28x28 input images, then configuration defines a three
hidden layers MLP model. The first layer has 93 neurons with Leaky ReLU activation and a dropout rate of 0.7.
The next two layers consist of 8 neurons each with ReLU activation and a dropout rate of 0.1. Regularization
techniques and batch normalization are disabled. The Adam optimizer is used for gradient descent. The final
layer is an output layer with 47 units, each representing a class in the classification task, and utilizes the softmax
activation function to output a probability distribution over the classes.
Training with Adaptive Learning Rate:
Employing learning rate schedulers, such as learning rate scheduler and reduce lr on plateau technique,
optimises learning rate dynamics during training. These strategies adjust the learning rate based on validation
loss trends, ensuring steady convergence. Although both techniques were explored, minimal performance
differences were observed. After total of 20 epochs, the model with the learning rate scheduler achieved 81.67%
accuracy and 78.78% validation accuracy, while the Reduce LR on Plateau model attained 80.40% accuracy and
78.34% validation accuracy. Consequently, the model with the learning rate scheduler was chosen for evaluation
on the test dataset.
Impact of Techniques on Model Performance:
The integration of hyperparameter tuning and optimization techniques significantly enhanced the MLP model's
performance. These methods aid in preventing overfitting by dynamically adjusting the learning rate, facilitating
more effective convergence towards the global optimum. Furthermore, by mitigating overfitting risks, these
techniques bolster the model's generalization capacity to unseen data, resulting in improved validation accuracy
and reduced validation loss. A notable improvement is observed when comparing the baseline and the optimized
models. The baseline model achieved an accuracy of 62.15% and a validation accuracy of 77.50%, while the
tuned model demonstrated substantially higher performance, with an accuracy of 81.67% and a validation
accuracy of 78.78%, both after the same number of epochs (20 epochs).
Addressing Overfitting and Underfitting:
Throughout the training process, the MLP model encountered intermittent challenges with overfitting,
particularly in its initial phases. To counteract this issue, we implemented regularization techniques like dropout,
aimed at introducing variability and preventing the model from excessively memorizing noise within the training
data. Additionally, the integration of batch normalization played a crucial role in stabilizing the training dynamics
and mitigating overfitting risks by standardizing input data distributions. By delicately balancing the model's
capacity and leveraging effective regularization strategies, the MLP model effectively managed the complexities
of overfitting and underfitting, ultimately leading to enhanced generalization performance on the validation set.
2.2 Rationale Design of CNN:
Chosen Hyperparameters/Techniques for tuning:
In crafting the Convolutional Neural Network (CNN) model, a diverse array of hyperparameters and techniques
offer avenues for exploration and refinement. This adaptable architecture allows for adjustments in layer
configurations, filter sizes, and activation functions, fostering experimentation. Techniques like dropout
regularization and batch normalization combat overfitting and stabilize training dynamics. Optimization
algorithms such as Adam, SGD, RMSprop, and Adadelta influence training efficiency. Overall, these meticulous
selections aim to enhance performance, resilience, and efficiency in training and generalization.
Hyperparameter Tuning Results-best MLP model:
After tuning the hyperparameters, the best model that we got, is-
{'conv_input_filter': 64, 'conv_input_kernel': 5, 'conv_input_padding': 'valid', 'conv__input_activation': 'elu',
'conv_filter': 104, 'conv_kernel': 5, 'conv_activation': 'relu', 'conv_dropout': 0.12245673371633485, 'dropout':
True, 'num_dense_layers': 2, 'dense_activation0': 'relu', 'Dense_neurons0': 96, 'dense_dropout0':
0.1565447071282482, 'regularization': True, 'batch normalization': False, 'optimizer': 'rmsprop',
'dense_activation1': 'relu', 'Dense_neurons1': 112, 'dense_dropout1': 0.0827728976888206,
'dense_activation2': 'elu', 'Dense_neurons2': 72, 'dense_dropout2': 0.09705184313484912,'dense_activation3':
'tanh', 'Dense_neurons3': 80, 'dense_dropout3': 0.06241696994094974, 'dense_activation4': 'leaky_relu',
'Dense_neurons4': 72, 'dense_dropout4': 0.102298394949072, 'L1_0': 0.0, 'L2_0': 0.0, 'L1_1': 0.0, 'L2_1': 0.0}.
