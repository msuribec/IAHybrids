import matplotlib.pyplot as plt
from itertools import product as cartesian_product
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import sympy as sm
import numpy as np
import os
from perceptron import Perceptron


class Comparison:
    """
    Class to compare different multi layer perceptrons
    ...

    Attributes
    ----------
    data : np.array
        Numpy array (N x M) of the entire data set
    Xindex : [int]
        List of indices of the columns of the features or independent variables
    Yindex : [int]
        List of indices of the columns of the outputs or dependent variables
    hl_max : float
        maximum number of hidden layers, we iterate from 1 to hl_max
    hn_max: float
        maximum number of neurons per hidden layer, we iterate from 1 to hn_max
    max_epochs : int
        maximum number of epochs for the training
    var_index : int
        index of last variable
    seed : int
        seed for the random number generator
    
    """
    
    def __init__(self,data, Xindex, Yindex, hl_max, hn_max, etas, max_epochs, type_autoencoder,var_index, seed = None):

        self.data = data
        self.Xindex = Xindex
        self.Yindex = Yindex

        self.n_vars = self.data.shape[1] -1 
        self.var_index = [i for i in range(var_index)]



        self.hl_max = hl_max
        self.hn_max = hn_max
        self.etas = etas
        self.max_epochs = max_epochs

        if seed is None:
            self.seed = int(np.pi*10**9)

        self.type_autoencoder = type_autoencoder
        self.results_path = 'Results' + type_autoencoder
        paths_to_create = [self.results_path] + [self.results_path + '/'+ stage for stage in ['Training', 'Validation', 'Test','csv']]
        self.create_paths(paths_to_create)

        np.random.seed(self.seed)

        self.normalize()
        self.random_sample()

        self.global_args = (
            self.X_train,
            self.Y_train,
            self.X_validation,
            self.Y_validation,
            self.X_test
        )

        
        list_hn = list(range(1,hn_max+1))
        mlp_params_list = [etas,list_hn]
        self.mlp_params_combinations = list(cartesian_product(*mlp_params_list))
    
        self.results = {}
        self.save_models_results()
        self.plot_results()
        self.mlp_keras()
        
        

    def normalize(self):
        """Function to normalize the data"""
        self.norm_data = (self.data - np.min(self.data, axis=0)) / (np.max(self.data, axis=0) - np.min(self.data, axis=0))

    
    def random_sample(self):
        """Function to randomly sample the data and split the data set into training, validation and test sets"""

        indices = np.arange(self.norm_data.shape[0])
        (
            data_train,
            data_tv,
            self.indices_train,
            indices_tv,
        ) = train_test_split(self.norm_data, indices, test_size=0.4, random_state=self.seed)
        indices_in_tv = np.arange(data_tv.shape[0])
        (
            data_test,
            data_validation,
            indices_test_in_tv,
            indices_val_in_tv,
        ) = train_test_split(data_tv, indices_in_tv, test_size=0.5, random_state=self.seed)

        self.index_test = indices_tv[indices_test_in_tv]
        self.indices_val  = indices_tv[indices_val_in_tv] 

        self.data_train = data_train
        self.data_val = data_validation
        self.data_test = data_test

        self.X_train = data_train[:, self.Xindex]
        self.Y_train = data_train[:, self.Yindex]

        self.X_test  =  data_test[:, self.Xindex]
        self.Y_test  =  data_test[:, self.Yindex]
        
        self.X_validation = data_validation[:, self.Xindex]
        self.Y_validation = data_validation[:, self.Yindex]


        self.Y_test = self.Y_test.reshape(self.X_validation.shape[0], self.Y_validation.shape[1])

        self.Y_train_sorted = self.X_train[np.argsort(self.indices_train)]
        self.Y_train_sorted = self.Y_train[np.argsort(self.indices_train)]

        self.X_test_sorted = self.X_test[np.argsort(self.index_test)]
        self.Y_test_sorted = self.Y_test[np.argsort(self.index_test)]

        self.X_validation_sorted = self.X_validation[np.argsort(self.indices_val)]
        self.Y_validation_sorted = self.Y_validation[np.argsort(self.indices_val)]

    
    def save_models_results(self):

        """ Function to save the results of the models in a dictionary"""
        for hl in range(1, self.hl_max + 1):
            results_dics = [{} for i in range(9)]
            for (lr,hn) in self.mlp_params_combinations:
                args = self.global_args + (hl,hn, self.results_path)
                mlp = Perceptron(*args)
                mlp.save_results(results_dics, self.max_epochs,lr, self.var_index, self.data_train)
            self.results[str(hl)] = results_dics

    def plot_comparison(self,data,indexes, label_names, title, x_label, y_label, filepath, size, plot_pred = False, obs=None):
        """ Function to plot comparisons between models
        Parameters
        ----------
        data : dict
            Dictionary with the data to plot, the dictionary contains the name of the model as key and the predicted values as values
        indexes : [str]
            List of the names of the models to plot
        label_names : [str]
            List of the labels of the models to plot (best, avg, worst for example)
        title : str
            Title of the plot
        x_label : str
            Label of the x axis
        y_label : str
            Label of the y axis
        filepath : str
            Path to save the image of the plot
        size : tuple
            Size of the plot
        plot_pred : bool
            Boolean that signifies if the plot is of the predictions or not
        obs : np.array
            Array of the observed values, only used if plot_pred is True
        """

        for feature in range(data[indexes[0]].shape[1]):
            plt.style.use('ggplot')
            fig, ax = plt.subplots(1,1,figsize=size,sharex=True)
            if plot_pred:
                ax.plot(obs[:,feature], ".-", label="Observations", color='b')
            for label_name, index in zip(label_names,indexes):
                label_i = f"{index} ({label_name})"
                ax.plot(data[index][:,feature], ".-", label=label_i, color='c')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.legend(framealpha=1)
            # fig.tight_layout()
            fig.suptitle(title)
            
            if 'png' in filepath:
                split_filepath = filepath.split('.')
                newfilepath = split_filepath[0] + f' feature {feature}'
            else:
                newfilepath = filepath
            # plt.show()
            plt.savefig(newfilepath + '.png', bbox_inches='tight')
            plt.close()



    def plot_comparison_works_plot_best(self,data,indexes, label_names, title, x_label, y_label, filepath, size, plot_pred = False, obs=None):
        """ Function to plot comparisons between models
        Parameters
        ----------
        data : dict
            Dictionary with the data to plot, the dictionary contains the name of the model as key and the predicted values as values
        indexes : [str]
            List of the names of the models to plot
        label_names : [str]
            List of the labels of the models to plot (best, avg, worst for example)
        title : str
            Title of the plot
        x_label : str
            Label of the x axis
        y_label : str
            Label of the y axis
        filepath : str
            Path to save the image of the plot
        size : tuple
            Size of the plot
        plot_pred : bool
            Boolean that signifies if the plot is of the predictions or not
        obs : np.array
            Array of the observed values, only used if plot_pred is True
        """
        

        for feature in range(data[indexes[0]].shape[1]):
            plt.style.use('ggplot')
            fig, ax = plt.subplots(1,1,figsize=size,sharex=True)
            if plot_pred:
                ax.plot(obs[:,feature], ".-", label="Observations")
            for label_name, index in zip(label_names,indexes):
                label_i = f"{index} ({label_name})"
                print(data[index])
                dat_f = data[index].reshape(*obs.shape)
                ax.plot(dat_f[:,feature], ".-", label=label_i)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.legend(framealpha=1)
            # fig.tight_layout()
            fig.suptitle(title)
            
            if 'png' in filepath:
                split_filepath = filepath.split('.')
                newfilepath = split_filepath[0] + f' feature {feature}'
            else:
                newfilepath = filepath
            # plt.show()
            plt.savefig(newfilepath + '.png', bbox_inches='tight')
            plt.close()
      

    def set_params(self,type_plot, stage, name):
        """ Function to set the parameters of the plot
        Parameters
        ----------
        type_plot : str
            Type of plot to make, can be 'error', 'local_grad' or 'pred'
        stage : str
            Stage of the data to plot, can be 'training', 'validation' or 'test'
        name : str
            name of the model
        Returns
        -------
            plot_title : str
                Title of the plot
            x_label : str
                Label of the x axis
            y_label : str
                Label of the y axis
            filepath : str
                Path to save the image of the plot
        """
        
        if type_plot == 'error':
            x_label = "epochs"
            y_label = r"$\mathcal{E}_{av}$"
            fig_title = 'Energy error'
            if name:
                plot_title = r"$\mathcal{E}_{av}$" + f" ({stage})"
        elif type_plot == 'local_grad':
            x_label = "epochs"
            y_label = r"$\delta_k$"
            fig_title = "Local output gradients"
            plot_title = r"Avg $\delta_k$" + f" ({stage})"
        else:
            x_label = "p"
            y_label = "Y"
            fig_title = "Pred vs obs"
            plot_title = f"Comparison of predicted and observed values"
        file_path = f"{self.results_path}/{stage}/{fig_title} in {stage}" 
        if name is not None:
            hl = name[-1]
            plot_title = plot_title + f"with {hl} hidden layer(s)"
            file_path = file_path + f" hl={hl}.png"
        return plot_title, x_label, y_label, file_path

    
    def sort_by_value(self,dic):
        """ FUnciton to sort a dictionary by its values"""
        return dict(sorted(dic.items(), key=lambda x:x[1]))
    

    def rank_models(self,last_errors, n_weight_matrices):
        """ Function to rank the models by their simplicity and error
        Parameters
        ----------
        last_errors : dict
            Dictionary with the name of the model as key and the error of the last epoch as value
        n_weight_matrices : dict
            Dictionary with the name of the model as key and the number of weight matrices as value
        Returns
        -------
        best : str
            Name of the best model
        avg : str
           Name of the average model
        worst : str
            Name of the worst model
        """
        sorted_by_errors = self.sort_by_value(last_errors)
        sorted_params = {}
        for key in sorted_by_errors:
            sorted_params[key] = n_weight_matrices[key]
        sorted_simplicity = self.sort_by_value(sorted_params)
        ordered_models = list(sorted_simplicity.keys())
        indexes = [0,-len(ordered_models)//2-1,-1]
        best = ordered_models[indexes[0]]
        avg = ordered_models[indexes[1]]
        worst = ordered_models[indexes[2]]
        return best, avg, worst

    
    def plot_results(self):
        """Function that finds the best, average and worst model, plots the results of the predictions,
        saves the errors of the models and makes comparison plots between models with the same number of
        hidden layers
        """
        test_errors_pairs = []
        test_errors = []
        validation_errors = []
        val_errors = []
        val_preds = []
        test_preds = []
        hl_list = []

        for hl in self.results:
            predictions_models = self.results[hl][4]

            for key in predictions_models:
                pred_not_sorted = predictions_models[key]
                y_pred = pred_not_sorted[np.argsort(self.index_test)]
                test_error = np.mean(np.power(self.Y_test_sorted - y_pred,2))
                test_errors.append(test_error)
                test_errors_pairs.append((key, test_error))
                test_preds.append(y_pred)
            predictions_models_vals = self.results[hl][7]

            for key in predictions_models_vals:
                
                pred_not_sorted_val = predictions_models_vals[key]
                y_pred_val = pred_not_sorted_val[np.argsort(self.indices_val)]

                error_val = np.mean(np.power(self.Y_validation - y_pred_val,2))
                val_preds.append(y_pred_val)
                val_errors.append(error_val)
                hl_list.append(hl)
                validation_errors.append((key, error_val))
            
        self.plot_best(hl_list,val_errors, val_preds, validation_errors, 'validation')
        self.plot_best(hl_list,test_errors, test_preds, test_errors_pairs, 'test')


        test_errors
        df = pd.DataFrame(test_errors_pairs, columns=['model', 'test_error'])
        df.to_csv(f'{self.results_path}/csv/test_errors.csv', index=True)
        df = pd.DataFrame(validation_errors, columns=['model', 'validation_error'])
        df.to_csv(f'{self.results_path}/csv/validation_errors.csv', index=True)

    def plot_best(self, hl_list, val_errors, val_preds, validation_errors, stage):
        """Function to plot the predictions of the best model
        Parameters
        ----------
        hl_list : [int]
            List of the number of hidden layers of the models
        val_errors : [float]
            List of the validation errors of the models
        val_preds : [np.array]
            List of the predictions of the models
        validation_errors : [dic]
            List of dictionaries with model names and validation val_errors
        stage : str
            Stage of the data to plot, can be 'validation' or 'test'
        """

        sort_val = np.argsort(val_errors)

        best_i = sort_val[0]
        avg_i = sort_val[len(sort_val)//2 -1]
        worst_i = sort_val[-1]

        self.best_hl = hl_list[best_i]

        best_model = val_preds[best_i]
        avg_model = val_preds[avg_i]
        worst_model = val_preds[worst_i]

        

        best_model_name = validation_errors[best_i][0]
        avg_model_name = validation_errors[avg_i][0]
        worst_model_name = validation_errors[worst_i][0]


        data = {
            best_model_name: best_model,
            avg_model_name: avg_model,
            worst_model_name: worst_model
        }
        indexes = [best_model_name,avg_model_name,worst_model_name]
        label_names = ['best', 'avg', 'worst']

        size = (7,7)

        plot_title, x_label, y_label, filepath = self.set_params('pred', stage, None)
        self.plot_comparison(data,indexes, label_names, plot_title, x_label, y_label, filepath, size, plot_pred = True, obs=self.Y_validation)

        data_best = {
            best_model_name: best_model
        }

        self.best_model_name = best_model_name
        indexes = [best_model_name]
        label_names = ['best']
        plot_title = 'Predicted values of the best model vs real observations'
        x_label = 'P'
        y_label = 'Y'
        filepath = f'{self.results_path}/Test/best pred.png'
        self.plot_comparison(data_best,indexes, label_names, plot_title, x_label, y_label, filepath, size, plot_pred = True, obs=self.Y_validation)
        

    def create_paths(self, directories):
        """ Function to create directories to save the results"""
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def mlp_keras(self):
        """ Function to reshape data run a MLP with the best model and save results"""
        results_best_model = self.results[str(self.best_hl)][8][self.best_model_name]

    
        y_train = self.data_train[:,-1].reshape(-1,1)

        x_val = self.data_val[:, self.var_index]
        y_val = self.data_val[:,-1].reshape(-1,1)

        x_test = self.data_test[:, self.var_index]
        y_test = self.data_test[:,-1].reshape(-1,1)

        self.save_mlp_keras_results(results_best_model,y_train,x_val, y_val, x_test, y_test)

    def save_mlp_keras_results(self,X_train, Y_train, X_validation, Y_validation, X_test, Y_test):
        """ Function to run a MLP with the best model and save the results
        Parameters
        ----------
        X_train : np.array
            Numpy array of the features of the training set
        Y_train : np.array
            Numpy array of the outputs of the training set
        X_validation : np.array
            Numpy array of the features of the validation set
        Y_validation : np.array
            Numpy array of the outputs of the validation set
        X_test : np.array
            Numpy array of the features of the test set
        Y_test : np.array
            Numpy array of the outputs of the test set    
        """
        max_epochs = 50
        eta = 0.2

        input_shape = (X_train.shape[1],)
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(4, input_shape=input_shape, activation='sigmoid'))
        model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # Configure the model and start training
        model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=eta), metrics=['mean_squared_error'])
        H = model.fit(X_train, Y_train, epochs=max_epochs, batch_size=50, verbose=1, validation_data=(X_validation, Y_validation))

        predictions = model.predict(X_test, batch_size=64)
        mse_test = np.mean(np.square(predictions - Y_test))
        print(mse_test)


        plt.style.use('ggplot')
        plt.figure()
        epoch_values = list(range(max_epochs))
        plt.plot(epoch_values, H.history['loss'], label='training')
        plt.plot(epoch_values, H.history['val_loss'], label='validation')
        plt.title('Loss in each epoch')
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{self.results_path}/loss_results_Best{self.type_autoencoder}.png', bbox_inches='tight')
        plt.close()

        plt.style.use('ggplot')
        plt.figure()
        epoch_values = list(range(max_epochs))
        plt.plot(epoch_values, H.history['mean_squared_error'], label='Train')
        plt.plot(epoch_values, H.history['val_mean_squared_error'], label='Validation ')
        plt.title('Error in each epoch')
        plt.xlabel('epoch')
        plt.ylabel('Error')
        plt.legend()
        plt.savefig(f'{self.results_path}/error_results_Best{self.type_autoencoder}.png', bbox_inches='tight')
        plt.close()
    
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(Y_test, label='obs')
        plt.plot(predictions, label='predictions ')
        plt.title(f'Predictions of Best MLP in {self.type_autoencoder} (MSE = {mse_test:.5f})')
        plt.xlabel('p')
        plt.ylabel('Y')
        plt.legend()
        plt.savefig(f'{self.results_path}/predictions_Best{self.type_autoencoder}.png', bbox_inches='tight')
        plt.close()

        self.save_test_errors(mse_test)


    def save_test_errors(self, line):
        """ Function to save the test error of the best model
        Parameters
        ----------
        line : float
            Test error of the best model
        """
        file_name = f'{self.results_path}/prediction_error_Best{self.type_autoencoder}.txt'
        
        # Open the file in write mode (creates the file if it doesn't exist)
        with open(file_name, "w") as file:
            # Write the number to the file as a string
            file.write(str(line))


def get_data():
    """ Function to get the data from the file data.txt
    Returns
    -------
    data : np.array
        Numpy array (N x M) of the entire data set
    Xindex : [int]
        List of indices of the columns of the features or independent variables
    Yindex : [int]
        List of indices of the columns of the outputs or dependent variables
    """
    data = np.loadtxt('data.txt', delimiter=',')
    Xindex = [0,1,4]
    Yindex = [3]
    return data, Xindex, Yindex

if __name__ == "__main__":
    # Let's compare autoencoders with low and high dimension
    data, Xindex, Yindex = get_data()
    Comparison(data, Xindex, Xindex ,1,3,[0.01,0.02,0.03,0.001],50, ' Reduction', 3)
    Comparison(data, Xindex, Xindex ,1,10,[0.01,0.02,0.03,0.001],50,' Expansion', 4)