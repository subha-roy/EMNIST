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
The model begins with an input layer featuring 64 filters and a 5x5 kernel size with ELU activation, the model
progresses through a convolutional layer with 104 filters, employing ReLU activation and a dropout rate of
12.25%. Subsequent pooling layers, assumed to have default configurations, are followed by dense layers with
varying neuron counts and activation functions, including ReLU, ELU, and Tanh. Dropout regularization is applied
across these layers with dropout rates ranging from 6.24% to 15.65%. Batch normalization is disabled, and the
RMSprop optimizer is employed for gradient descent. Overall, this architecture aims to extract hierarchical
features from input images, leveraging dropout to prevent overfitting and optimizing learning dynamics with
RMSprop to enhance classification performance.
Training with Adaptive Learning Rate:
The use of learning rate schedulers, like learning rate scheduler and reduce lr on plateau technique collectively
optimize learning rate dynamics in training. These adaptive strategies adjust the learning rate based on validation
loss trends, ensuring stable convergence towards the optimal solution. Learning rate schedulers prevent
overshooting or stagnation, while Reduce LR on Plateau helps navigate local minima, enhancing generalization
to unseen data. Although, after exploring both the techniques, we won’t much of difference in the performance,
For the reduce lr on plateau, we observed the accuracy is 87.45% & val_accuracy is 87.05% after just total of 8
epochs, and for the learning rate scheduler, the accuracy is 87.39% & val_accuracy is 83.91 % after 8 epochs. So
afterwards, I used the model with learning rate scheduler to check the performance on the test dataset.
Impact of Techniques on Model Performance:
The integration of hyperparameter tuning and optimisation techniques has significantly elevated the
performance of the CNN model. These strategies play a pivotal role in averting overfitting by dynamically
adjusting the learning rate, thereby facilitating more effective convergence towards the global optimum.
Furthermore, by mitigating the risk of overfitting, these techniques bolster the model's capacity to generalise to
unseen data, resulting in heightened validation accuracy and diminished validation loss. Comparing the baseline
model with the optimized variant reveals a substantial enhancement in performance. Initially, the baseline model
exhibited an accuracy of 92.29% and a validation accuracy of 86.12% after 10 epochs. In contrast, the tuned
model achieved an accuracy of 87.45% and a validation accuracy of 87.05% after merely 8 epochs. Additionally,
it's noteworthy that the tuned model demonstrates superior convergence in terms of both loss and validation
loss.
Addressing Overfitting and Underfitting:
Throughout the CNN model's training process, occasional instances of overfitting were observed, particularly in
the initial training stages. To counter this challenge, various regularization techniques, including dropout, were
implemented. Dropout introduces randomness by deactivating neurons during training, thereby preventing the
model from memorizing noise present in the training data. Additionally, the incorporation of batch normalization
played a crucial role in stabilizing the training process. By normalizing the input data distributions, batch
normalization mitigated the risk of overfitting, ensuring more stable and consistent training dynamics. Through
a delicate balance of model capacity and the integration of regularization techniques, the CNN model effectively
tackled overfitting and underfitting issues, resulting in notable improvements in generalization performance on
the validation set.
Section 3- Results
i) During the training iterations of both the Multilayer Perceptron (MLP) and Convolutional Neural Network (CNN)
models, we observed adaptive fluctuations and plateaus in the training loss and accuracy curves over multiple
epochs. While the CNN model demonstrated smoother convergence in its training loss and accuracy curves,
indicative of efficient optimization and model convergence, the MLP model exhibited fluctuations and gaps
between the training and validation loss and accuracy curves. Despite these disparities, both models showcased
an overall trend of decreasing training loss and increasing training accuracy, suggesting effective learning from
the training data. Moreover, the CNN model demonstrated robust performance on unseen data, whereas the
MLP model displayed moderate performance on the test set. Overall, both models effectively learned and
generalized from the training data, with the CNN model's smoother convergence implying its superior ability to
capture complex patterns and make accurate predictions.
ii) The provided code offers a visual journey into the predictions rendered by both the MLP and CNN models for
six samples sourced from the EMNIST dataset. Each sample's display showcases the predicted and true labels,
their colours—blue for correct predictions and red for incorrect ones—serving as vivid indicators of prediction
accuracy. This visual exploration provides a tangible glimpse into the models' prowess in discerning handwritten
characters. Aligning predicted and true labels signifies accurate predictions, while disparities between them
unveil areas of potential challenge. This visual assessment serves as a valuable tool for evaluating the models
adeptness in distinguishing between various characters, offering nuanced insights into their strengths and
potential limitations in character recognition tasks. By meticulously scrutinizing the predicted outcomes, we
glean invaluable insights into the models' performance dynamics, enriching our understanding of their
capabilities.
iii) In assessing the performance disparities between Multilayer Perceptron (MLPs) and Convolutional Neural
Networks (CNNs), it's imperative to scrutinise their unique characteristics and operational intricacies. MLPs excel
in deciphering intricate patterns from structured data, proving invaluable for tasks involving tabular and
sequential data analysis. However, they struggle with high-dimensional data, particularly images, where they
stumble due to their inability to comprehend spatial relationships between pixels, thus compromising their
effectiveness in image-related tasks. Conversely, CNNs are purposefully engineered to surmount this limitation
by leveraging convolutional layers to adeptly extract local patterns and hierarchical representations within
images. This hierarchical feature extraction empowers CNNs to adeptly capture spatial hierarchies, with lower
layers discerning rudimentary features like edges and higher layers synthesizing them to identify complex
patterns such as objects or shapes. Despite this advantage, CNNs entail computational overhead, especially with
increasing network depth and complexity. Moreover, their efficacy hinges on substantial labelled data for
effective training, posing challenges in data-scarce scenarios. Furthermore, interpreting the learned
representations within CNNs can be challenging, complicating the comprehension of model decisions and
insights. Overall, while both MLPs and CNNs exhibit strengths and weaknesses, the latter's innate capacity to
comprehend spatial relationships and extract features renders them more adept at image classification tasks.
This inherent superiority elucidates why the CNN model outperformed the MLP in the analysis, showcasing
enhanced accuracy and superior generalization in image classification endeavours.
Section 4: Conclusions
Embarking on this project has been an illuminating journey, delving deep into the intricate realm of machine
learning. It has afforded me invaluable insights into two fundamental architectures: the Multilayer Perceptron
(MLP) and the Convolutional Neural Network (CNN). Through meticulous exploration and experimentation, I've
cultivated a profound understanding of how various elements such as hyperparameters, activation functions,
regularization techniques, and optimization algorithms intricately shape the efficacy and convergence patterns
of these models. Upon reflection, I've come to appreciate the indispensable role of methodical experimentation
and continual parameter adjustment in optimizing model performance. While both the MLP and CNN models
have shown promise, I recognize the vast potential for improvement. Moving forward, my dedication lies in fine-
tuning hyperparameters and delving into more sophisticated architectures to elevate model accuracy and
convergence.
Immersed in this project, I've experienced a journey of enriching learning, deepening my understanding of neural
network structures, hyperparameter tuning, and meticulous model assessment. Scrutinizing predicted outcomes
from both MLP and CNN models has yielded invaluable insights into their respective strengths and limitations,
particularly in character recognition tasks. Reflecting on the journey thus far, I've realized the critical importance
of interpreting model predictions to extract actionable insights for refinement and practical application. Looking
ahead, I'm eager to apply these lessons to address intricate machine learning challenges and continually refine
my skills in model development and deployment.
This project has significantly broadened my horizons in crafting and evaluating machine learning models,
particularly MLP and CNN architectures. Exploring a diverse range of hyperparameters, activation functions,
regularization techniques, and optimization algorithms has deepened my insight into how these components
interact to influence model performance and convergence dynamics. This newfound knowledge fuels my passion
for relentless exploration and innovation in the dynamic field of machine learning.
Reference:
1. Keras API Docs - https://keras.io/api/
2. Keras Tuner - https://keras.io/keras_tuner/
3. Scikit-Learn API - https://scikit-learn.org/stable/modules/classes.html
4. Data Source - https://www.kaggle.com/datasets/crawford/emnist


