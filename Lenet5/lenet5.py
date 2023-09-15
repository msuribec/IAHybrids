import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
import tensorflow.compat.v1 as tf
from sklearn.metrics import classification_report


class lenet5Model:
    """Class that defines the LeNet-5 model
    Parameters
    ----------
    train_x : pandas.DataFrame
        Training data
    train_y : pandas.Series
        Training labels
    valid_x : pandas.DataFrame
        Validation data
    valid_y : pandas.Series
        Validation labels
    test_x : pandas.DataFrame
        Test data
    learning_rate : float
        Learning rate for the optimizer
    batch_size : int
        Batch size for training
    num_epochs : int
        Number of epochs to train for
    """
    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, learning_rate=0.0001, batch_size=128, num_epochs=1000):
        self.num_epochs = num_epochs
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.learning_rate = learning_rate
        self.batch_size = batch_size


        # Create placeholder for model input and label.
        # Input shape is (minbatch_size, 28, 28)
        self.X = tf.placeholder(tf.float32, [None, 28, 28], name="X")
        self.Y = tf.placeholder(tf.int64, [None, ], name="Y")

        logits = self.CNN(self.X)
        print(logits)
        softmax = tf.nn.softmax(logits)

        # Convert our labels into one-hot-vectors
        labels = tf.one_hot(indices=tf.cast(self.Y, tf.int32), depth=10)

        # Compute the cross-entropy loss
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels))

        # Use adam optimizer to reduce cost
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.cost)


        # For testing and prediction
        self.predictions = tf.argmax(softmax, axis=1)
        correct_prediction = tf.equal(self.predictions, self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Initialize all the variables
        

    def train(self):
        """Function to train the model
        Returns
        -------
        test_pred : np.ndarray
            Array with the predictions for the test set
        losses : list
            List with the loss for each epoch
        accuracies : list
            List with the accuracy for each epoch
        
        """
        init = tf.global_variables_initializer()

        losses = []
        accuracies = []
        # Running the model
        with tf.Session() as sess:

            sess.run(init)

            for epoch in range(self.num_epochs):
                num_samples = self.train_x.shape[0]
                num_batches = (num_samples // self.batch_size) + 1
                epoch_cost = 0.
                i = 0
                while i < num_samples:
                    batch_x = self.train_x.iloc[i:i+self.batch_size,:].values
                    batch_x = batch_x.reshape(batch_x.shape[0], 28, 28)

                    batch_y = self.train_y.iloc[i:i+self.batch_size].values

                    i += self.batch_size

                    # Train on batch and get back cost
                    _, c = sess.run([self.train_op, self.cost], feed_dict={self.X:batch_x, self.Y:batch_y})
                    epoch_cost += (c/num_batches)

                # Get accuracy for validation
                valid_accuracy = self.accuracy.eval(
                    feed_dict={self.X:self.valid_x.values.reshape(self.valid_x.shape[0], 28, 28),
                            self.Y:self.valid_y.values})

                print ("Epoch {}: Cost: {}".format(epoch+1, epoch_cost))
                print("Validation accuracy: {}".format(valid_accuracy))
                losses.append(epoch_cost)
                accuracies.append(valid_accuracy)

            test_pred = self.predictions.eval(feed_dict={self.X:self.test_x})
            return test_pred,losses,accuracies



    def CNN(self,X):
        """Function that defines the CNN architecture
        Parameters
        ----------
        X : tf.placeholder
            Placeholder for the input data
        Returns
        -------
        logits : tf.Tensor
            Tensor with the logits for the output layer
        """

        # Reshape input to 4-D vector
        input_layer = tf.reshape(X, [-1, 28, 28, 1]) # -1 adds minibatch support.

        # Padding the input to make it 32x32. Specification of LeNET
        padded_input = tf.pad(input_layer, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")

        # Convolutional Layer #1
        # Has a default stride of 1
        # Output: 28 * 28 * 6

        conv1 = tf.layers.conv2d(
          inputs=padded_input,
          filters=6, # Number of filters.
          kernel_size=5, # Size of each filter is 5x5.
          padding="valid", # No padding is applied to the input.
          activation=tf.nn.relu)

        # Pooling Layer #1
        # Sampling half the output of previous layer
        # Output: 14 * 14 * 6
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2
        # Output: 10 * 10 * 16
        conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=16, # Number of filters
          kernel_size=5, # Size of each filter is 5x5
          padding="valid", # No padding
          activation=tf.nn.relu)

        # Pooling Layer #2
        # Output: 5 * 5 * 16
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Reshaping output into a single dimention array for input to fully connected layer
        pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])

        # Fully connected layer #1: Has 120 neurons
        dense1 = tf.layers.dense(inputs=pool2_flat, units=120, activation=tf.nn.relu)

        # Fully connected layer #2: Has 84 neurons
        dense2 = tf.layers.dense(inputs=dense1, units=84, activation=tf.nn.relu)

        # Output layer, 10 neurons for each digit
        logits = tf.layers.dense(inputs=dense2, units=10)

        return logits


class runLenet5:
    """Class that runs the LeNet-5 model
    """
    def __init__(self):
        

        self.create_folders_if_not_exist(['Results'])
        PATH_VALIDATION_LABELS = 'Data/t10k-labels-idx1-ubyte'
        PATH_VALIDATION_IMG = 'Data/t10k-images-idx3-ubyte'
        PATH_TRAIN_LABELS = 'Data/train-labels-idx1-ubyte'
        PATH_TRAIN_IMG = 'Data/train-images-idx3-ubyte'
        

        self.val_img, self.validation_labels = self.read_mnist_data(PATH_VALIDATION_IMG, PATH_VALIDATION_LABELS)
        self.train_img, self.train_labels = self.read_mnist_data(PATH_TRAIN_IMG, PATH_TRAIN_LABELS)

        self.X_test, self.Y_test = self.read_course_data('Data/numbers.csv')

        self.save_results()

    def create_folders_if_not_exist(self,folder_paths):
        """Function that creates folders if they don't exist
        Parameters
        ----------
        folder_paths : list
            List with the paths of the folders to create
        """
        for folder_path in folder_paths:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

    def read_mnist_data(self, file_path_imgs, file_path_labels):
        """Function that reads the MNIST data   
        Parameters
        ----------
        file_path_imgs : str
            Path to the images file
        file_path_labels : str
            Path to the labels file
        Returns
        -------
        images : pandas.DataFrame
            Dataframe with the images
        labels : pandas.Series
            Series with the labels
        """
        images = idx2numpy.convert_from_file(file_path_imgs)
        labels = idx2numpy.convert_from_file(file_path_labels).astype("int64")
        aux_img = images.reshape(*images.shape, 1)
        images = self.normalize_data(aux_img,0,255)
        images = pd.DataFrame(images.reshape(images.shape[0], images.shape[1]*images.shape[2]))
        labels = pd.Series(labels)
        return images, labels
       
    def read_course_data(self, path):
        """Function that reads the course data
        Parameters
        ----------
        path : str
            Path to the course data
        Returns
        -------
        X_test : np.ndarray
            Array with the images
        Y_test : np.ndarray
            Array with the labels
        """
        data = pd.read_csv(path, index_col = 0)
        data = data[data['label'] != 'X']
        data_array = data.to_numpy()
        
        X_test = data_array[:,:-1]
        Y_test = data_array[:,-1].astype('int64')
        X_test =  np.asarray(X_test).astype('float32')
        X_test = X_test.reshape(X_test.shape[0],28,28)
        return X_test, Y_test


    def save_results(self):
        """Function that runs the model and saves the results
        """
        tf.disable_v2_behavior()
        l5 = lenet5Model(self.train_img, self.train_labels, self.val_img, self.validation_labels, self.X_test, learning_rate=0.01, batch_size=250, num_epochs=5)
        self.test_pred,losses,accuracies = l5.train()

        self.plot_values('Loss per epoch in validation', 'epoch', 'loss', losses, 'Results/lenet5_loss.png')
        self.plot_values('Accuracy per epoch in validation', 'epoch', 'accuracy', accuracies, 'Results/lenet5_accuracy.png')
        accuracies = self.calculate_accuracy(self.Y_test,self.test_pred)
        print(accuracies)
        
        confusion_matrix_np = self.calculate_confusion_matrix(self.Y_test, self.test_pred)
        class_names = np.unique(self.Y_test)
        self.plot_confusion_matrix(confusion_matrix_np, class_names)
        print(classification_report(self.Y_test, self.test_pred,target_names=[str(x) for x in class_names]))

 
    def plot_values(self, title, xlabel, ylabel, values, filename):
        """Function that plots a graph and saves it to a file
        Parameters
        ----------
        title : str
            Title of the graph
        xlabel : str
            Label for the x axis
        ylabel : str
            Label for the y axis
        values : list
            List with the values to plot
        filename : str
            Path to the file where the graph will be saved
        """
        plt.style.use('ggplot')
        plt.figure()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(values)
        plt.savefig(filename)
        plt.close()

    def normalize_data(self,data,min_,max_):
        """Function that normalizes the data
        Parameters
        ----------
        data : np.ndarray
            Array with the data to normalize
        min_ : float
            Minimum value of the data
        max_ : float
            Maximum value of the data
        Returns
        -------
        data : np.ndarray
            Array with the normalized data
        """
        return (data.astype("float32") -  min_) / (max_ - min_)
    
    def calculate_accuracy(self,Y_test,test_pred):
        """Function that calculates the accuracy for each class and the total accuracy
        Parameters
        ----------
        Y_test : np.ndarray
            Array with the true labels
        test_pred : np.ndarray
            Array with the predicted labels
        Returns
        -------
        accuracies : dict
            Dictionary with the accuracy for each class
        """
        correct_preds = {}
        accuracies = {}
        totals = {}
        
        for clase in range(10):
            correct_preds[str(clase)] = 0
            totals[str(clase)] = 0

        for i in range(len(Y_test)):
            totals[str(Y_test[i])] = totals[str(Y_test[i])] + 1
            if Y_test[i] == test_pred[i]:
                correct_preds[str(Y_test[i])] = correct_preds[str(Y_test[i])] + 1
        for clase in correct_preds:
            accuracies[clase] = correct_preds[clase]/ totals[clase]
        return accuracies

    def calculate_confusion_matrix(self,true_labels, predicted_labels):
        """Function that calculates the confusion matrix
        Parameters
        ----------
        true_labels : np.ndarray
            Array with the true labels
        predicted_labels : np.ndarray
            Array with the predicted labels
        Returns
        -------
        confusion_matrix : np.ndarray
            Array with the confusion matrix
        """ 
        num_classes = len(np.unique(true_labels))
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

        for i in range(len(true_labels)):
            true_label = true_labels[i]
            predicted_label = predicted_labels[i]
            confusion_matrix[true_label, predicted_label] += 1

        return confusion_matrix
    
    def plot_confusion_matrix(self,confusion_matrix, class_names):
        """Function that plots the confusion matrix
        Parameters
        ----------
        confusion_matrix : np.ndarray
            Array with the confusion matrix
        class_names : np.ndarray
            Array with the class names
        """
        
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        fmt = 'd'
        thresh = confusion_matrix.max() / 2.
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(j, i, format(confusion_matrix[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if confusion_matrix[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('Results/lenet5_confusion.png')
        plt.close()


if __name__ == "__main__":
    c1 = runLenet5()