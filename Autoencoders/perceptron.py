import matplotlib.pyplot as plt
from itertools import product as cartesian_product
import pandas as pd
import sympy as sm
import numpy as np
import os

class Perceptron:
    """
    Class that represents a multi layer perceptron
    ...

    Atributos
    ----------
    X_train : np.array
        Numpy matrix (N x K) of the training data, N is the number of datapoints and K is the number of features
    Y_train : N x M
        Numpy matrix (N x M) of the outputs of the training data, N is the number of datapoints and M is the number of outputs
    X_validation : [int]
        Numpy matrix (Nv x K) of the training data, Nv is the number of datapoints and K is the number of features
    Y_validation : [int]
        Numpy matrix (Nv x M) of the outputs of the validation data, Nv is the number of datapoints and M is the number of outputs
    X_test: [int]
        Numpy matrix (Nt x K) of the training data, Nt is the number of datapoints and K is the number of features
    hl : int
        Number of hidden layers
    hn : int
        Number of neurons per hidden layer
    """
    
    def __init__(self, X_train, Y_train, X_validation, Y_validation, X_test, hl, hn, results_path):

        # initialize parameters receibed

        self.results_path = results_path

        self.hl = hl
        self.hn = hn

        self.X_validation = X_validation
        self.Y_validation = Y_validation
        self.X_test = X_test

        self.X_train = X_train
        self.Y_train = Y_train
        self.Y_dtrain = sm.Matrix(Y_train).T
        
        self.n_dim = X_train.shape[1]
        self.N = X_train.shape[0]
        self.inputs = self.n_dim
        self.M = self.Y_dtrain.shape[0]
        self.Yd_train = np.reshape(Y_train,(self.N,self.M))
        self.model_name = f"MLP hl = {hl} hn= {hn}"
        

        # Iniitialize neuron and layer numbers
        self.neurons_input = self.inputs
        self.layers_hidden = hl
        self.neurons_hidden = [hn] * hl
        self.outputs = self.M
        self.weights_dimensions = [self.inputs ,self.neurons_input] + self.neurons_hidden + [self.outputs]
        self.neurons_numbers = [self.neurons_input] + self.neurons_hidden + [self.outputs]
        self.n_layers = self.layers_hidden + 2

        self.bias = [0.7] * self.n_layers

        # Compute the number of weights per layer and the total number of weights in the network

        num_weights_layer = []
        for l in range(self.n_layers):
            num_weights_layer.append(self.weights_dimensions[l]*self.weights_dimensions[l+1])
        self.num_weights_layer = num_weights_layer
        self.num_weights = sum(num_weights_layer)

        # Name each layer to label plots

        self.name_layers()

        # Initialize lists to save results

        self.validation_errors =[]
        self.validation_avg_energy_errors=[]
        self.validation_local_gradients=[]
        self.validation_delta_ks =[]


    def name_layers(self):
        """
        Auxiliar function that initializes a list with names for the layers
        """
        self.layer_names = []
        for l in range(self.n_layers):
            type_layer = ''
            if l == 0:
                type_layer = 'input'
            elif l == self.n_layers-1:
                type_layer = 'output'
            else:
                type_layer = 'hidden'
            l_name = f"{type_layer} layer"
            if type_layer == 'hidden': 
                l_name = l_name + str(l)
            self.layer_names.append(l_name)

    def get_energy(self,e_vector):
        """Function that returns the instantaneous energy error

        Parameters
        ----------
        e_vector : numpy ndarray
            Error matrix 
        
        Returns
        ----------
        int: intantaneous energy error
            
        """
        return 0.5 * (sum(e_vector**2))

    def forward(self, x, weights):
        """ Function that performs a forward step.
        It calculates the local field and the activation value and propagates forward

        Parameters
        ----------
        X : numpy ndarray
            Input matrix
        weights: numpy ndarray
            Weights matrix 

        Returns
        ----------
        Y: numpy ndarray of outputs of the last layer
        Yi: numpy ndarray of outputs of each layer
        Vi: numpy ndarray of local fields of each layer
        impulses: numpy ndarray of inputs of each layer
        """   
        Vi = []
        phi_i_v_i = x
        Yi = []
        for i in range(self.n_layers):
            wi = weights[i]
            vi = np.dot(phi_i_v_i,wi)
            vi = vi + self.bias[i]
            Vi.append(vi)
            phi_i_v_i = self.num_phi(vi,i)
            Yi.append(phi_i_v_i)
        Y = Yi[-1]
        impulses = [x] + Yi
        return Y, Vi, Yi, impulses
    

    def num_phi(self, x, layer):
        """ Activation function
        It returns the layer's activation function applied to a specific value
        Parameters
        ----------
        X : numpy ndarray
            argument to aply the activation function to
        layer: int
            number of layer
        Returns
        ----------
        The layer's activation function applied to x
        """   
        if layer == 0:
            return 1 / (1 + np.exp(-x))
        elif layer == self.n_layers -1:
            return 1 / (1 + np.exp(-x))
        else:
            return 1 / (1 + np.exp(-x))


    def num_dphi(self, x, layer):
        """ Derivative of activation function
        It returns the derivative of the layer's activation function applied to a specific value

        Parameters
        ----------
        X : numpy ndarray
            argument to aply the derivative of the activation function to
        layer: int
            number of layer
        
        Returns
        ----------
        The derivative of the layer's activation function applied to x
        """   
        if layer == 0:
            return x * (1 - x)
        elif layer == self.n_layers -1:
            return x * (1 - x)
        else:
            return x * (1 - x)



    def gradient_descent(self, initial_values,epochs, eta, tol = 0.01):
        """ Function that implements the gradient descent algorithm

        Parameters
        ----------
        initial values : [numpy ndarray]
            list of weight matrices (one per layer)
        epochs: int
            maximum epochs
        eta: float
            learning rate
        tol: float
            tolerance for stopping condition
        
        """   
        self.model_name = self.model_name + f' eta = {eta}'
        self.epochs = epochs
        
        assert len(initial_values) == self.n_layers, "not enough initial weight matrices were passed"
        energy_errors_av = []
        errors = []
        param_values = [np.zeros((self.epochs,self.num_weights_layer[l])) for l in range(self.n_layers)]
        dif_values = [np.zeros((self.epochs,self.num_weights_layer[l])) for l in range(self.n_layers)]
        local_grad_values = [np.zeros((self.epochs, 1)) for i in range(self.n_layers)]
        mean_delta_k_output = []
    
        error_it = 10000
        it = 0
        W = initial_values

        while it < epochs and error_it > tol:
            # forward
            Y, Vi, Yi, impulses = self.forward(self.X_train,W)
            local_gradients_it = []
            # batch back propagation starting from the output layer
            for layer in range(self.n_layers-1,-1,-1): 
                if layer == self.n_layers -1:
                    d = self.Yd_train
                    error = d-Y
                    energy = self.get_energy(error)
                    avg_energy_error_it = np.mean(energy)
                    errors = np.append(errors,error)
                    energy_errors_av = np.append(energy_errors_av,avg_energy_error_it)
                else:
                    wi = W[layer+1]
                    error = np.dot(delta_k,wi.T)
                dphi_vi = self.num_dphi(Yi[layer], layer)
                delta_k = error * dphi_vi
                local_gradients_it.append(delta_k)
                if layer == self.n_layers - 1:
                    mean_delta_k_output = np.append(mean_delta_k_output,np.mean(delta_k))
                local_grad_values[layer][it] = np.mean(delta_k)
                   
            # update_weights
            for layer in range(self.n_layers):
                index = self.n_layers - 1-  layer
                impulse = impulses[layer]
                delta_k  = local_gradients_it[index]
                
                dJdw = impulse.T.dot(delta_k)

                # print('--------------')
                # print(W[layer])
                W[layer] = W[layer] + eta*dJdw
                # print(dJdw)
                # print(W[layer])
                param_values[layer][it,:] = W[layer].flatten().tolist()

            self.validation(W) 

            it+=1
        

        self.param_values = param_values
        self.dif_values = dif_values
        self.local_gradients_array = local_grad_values

        self.errors = errors
        self.avg_energy_errors_training = energy_errors_av
        self.training_weights = W
        self.mean_delta_k_output = mean_delta_k_output

    def validation(self, W):
        """ Function that finds the predicted output on the validation set,
        it also finds one gradient to test the behaviour on the validation set.
        Training does not occur here, the model does not learn from the validation set

        Parameters
        ----------
        W : [numpy ndarray]
            weight matrix 
        """   
        N_validation = self.Y_validation.shape[0]
        Y_validation = np.reshape(self.Y_validation,(N_validation,self.M))
        #foward
        Y, Vi, Yi, impulses = self.forward(self.X_validation,W)
        #backward to find error and local gradient
        layer = self.n_layers - 1
        validation_error = Y_validation -Y 
        instantaneous_energy = 0.5 * (sum(validation_error**2))
        av_error = np.mean(instantaneous_energy)
        delta_k = validation_error * self.num_dphi(Y,layer)
        mean_delta_k = np.mean(delta_k)

        self.validation_errors = np.append(self.validation_errors,validation_error)
        self.validation_avg_energy_errors = np.append(self.validation_avg_energy_errors, av_error)
        self.validation_local_gradients = np.append(self.validation_local_gradients, delta_k)
        self.validation_delta_ks = np.append(self.validation_delta_ks, mean_delta_k)

    def test(self, X_test):
        """ Fucntion that finds the predicted output on the test set
        Parameters
        ----------
        X_test : [numpy ndarray]
            Matrix of testing data
        """   
        Y, _,_,_ = self.forward(X_test,self.training_weights)
        self.Y_hat_test = Y

    def final_validate(self):
        """ Function that finds the predicted output on the validation set
        """   
        Y, _,_,_ = self.forward(self.X_validation,self.training_weights)
        self.y_hat_val = Y
        

    def graph(self):
        """ Auxiliary function to call other methods and generate graphs """  
        self.graph_errors()
        self.graph_gradients()
        
    def graph_gradients(self):
        """ Auxiliary function to graph local gradients """  
        assert len(self.local_gradients_array) == self.n_layers
        grads_array = np.hstack(self.local_gradients_array)
        df_grads = [pd.DataFrame(grads_array,columns=self.layer_names)]

        title = [r'Average local gradient $\delta_k$']
        filepath = f'{self.results_path}/Training/Gradients {self.model_name}.jpg'
        x_label = ['epoch']
        y_label = [r'$\delta_k$']
        self.graph_dfs(df_grads, title,x_label, y_label,filepath)
        
    def graph_dfs(self, df_list,titles,x_labels, y_labels,filepath, size= (5,5)):
        """ Auxiliary function to graph multiple dataframes

        Parameters
        ----------
        df_list : [df]
            List of dataframes to plot
        titles: [str]
            list of titles
        x_labels: [str]
            list of labels along the x axis
        y_labels: [str]
            list of labels along the y axis
        filepath: str
            filepath to save the image
        size: (float,float)
            size of the image
        """  
        num_dfs = len(df_list)
        plt.style.use('ggplot')
        fig, ax = plt.subplots(num_dfs,1,figsize=size,sharex=True)
        if num_dfs == 1:
            df = df_list[0]
            df.plot(marker=".",rot=45,ax=ax,legend=True, title= titles[0])
            ax.set_xlabel(x_labels[0])
            ax.set_ylabel(y_labels[0])

            fig.tight_layout()
            plt.savefig(filepath,bbox_inches='tight')
            plt.close()
        else:
            for l in range(num_dfs):
                df = df_list[l]
                df.plot(marker=".",rot=45,ax=ax[l],legend=False, title= titles[l])
                ax[l].set_xlabel(x_labels[l])
                ax[l].set_ylabel(y_labels[l])
                lines0, labels0 = [sum(x, []) for x in zip(*[ax[l].get_legend_handles_labels()])]
                
            fig.tight_layout()
            plt.savefig(filepath)
            plt.close()

    def graph_errors(self):
        """ Auxiliary function to graph errors """  
        df_av_instantaneous_energy = [pd.DataFrame(self.avg_energy_errors_training,columns=['$\mathcal{E}_{av}$'])]
        title = ['Average Instantaneous energy error $\mathcal{E}_{av}$']
        filepath = f'{self.results_path}/Training/Errors {self.model_name}.jpg'
        x_label = ['epoch']
        y_label = ['$\mathcal{E}_{av}$']
        self.graph_dfs(df_av_instantaneous_energy, title,x_label, y_label,filepath)
        
    def save_results(self, list_dics, max_epochs,eta,  indexes, data):
        """ Auxiliary function to run training, validation and testing and 
        save relevant results of the model to a list of dictionaries
        
        Parameters
        ----------
        list_dics : [dict]
            List of dictionaries were the results of the model will be saved
        max_epochs: int
            maximum number of epochs
        eta: float
            learning rate
        
        """  
        self.initialize_gradient_descent(max_epochs,eta)
        self.test(self.X_test)
        self.final_validate()
        self.graph()
    
        list_dics[0][self.model_name] = self.avg_energy_errors_training
        list_dics[1][self.model_name] = self.validation_avg_energy_errors
        list_dics[2][self.model_name] = self.mean_delta_k_output
        list_dics[3][self.model_name] = self.validation_delta_ks
        list_dics[4][self.model_name] = self.Y_hat_test
        list_dics[5][self.model_name] = self.validation_avg_energy_errors[-1]
        list_dics[6][self.model_name] = self.num_weights
        list_dics[7][self.model_name] = self.y_hat_val
        list_dics[8][self.model_name] = data[:, indexes]
 


    def initialize_gradient_descent(self,epochs,eta):
        """ Auxiliary function to initialize weights in the hyercube [-1,1]
        and initialize gradient descent
        
        Parameters
        ----------
        list_dics : [dict]
            List of dictionaries were the results of the model will be saved
        max_epochs: int
            maximum number of epochs
        eta: float
            learning rate
        """  
        initial_values = []
        for i in range(self.n_layers):
            sizei = (self.weights_dimensions[i], self.weights_dimensions[i+1])
            wi = np.random.uniform(low=-1, high=1,size=sizei)
            assert self.num_weights_layer[i] == wi.shape[0]*wi.shape[1]
            initial_values.append(wi)
        self.gradient_descent(initial_values, epochs, eta)
