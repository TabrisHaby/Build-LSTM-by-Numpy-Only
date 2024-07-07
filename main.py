# LSTM from Scratch
# How to program LSTM without using Pytorch or tensorflow.
# Scrpt by Haby @ 2024/07/05

import random
import numpy as np

def sigmoid(x):
    sigmoid = 1/(1+np.exp(-x))
    return float(sigmoid) # float for sigmoid fucntion

def sigmoid_derivative(x):
    sigmoid_derivative = x*(1-x)
    return sigmoid_derivative

def tanh_derivative(x):
    tanh_derivative = 1-x**2
    return tanh_derivative

def random_array(a, b, *args):
    np.random.seed(42)  # Fix the seed for reproducibility
    return np.random.rand(*args) * (b - a) + a 

class LSTM_Parameter:
    def __init__(self, cell_count, dim) -> None:
        """
        initialize parameters in LSTM
        There are 2 paramenters: 
        cell_count: cell count in LSTM
        dim: dim of input
        """
        self.cell_count = cell_count
        self.dim = dim
        # this is total input length
        input_length = cell_count + dim 

        # y = Wx + b, where x is input, W is weight, b is bias
        # weights for all gates
        self.weight_cell = random_array(0.5, -0.5, input_length, cell_count)
        self.weight_output = random_array(0.5, -0.5, cell_count, dim)
        self.weight_forget = random_array(0.5, -0.5, input_length, cell_count)
        self.weight_input = random_array(0.5, -0.5, input_length, cell_count)
        # bias for all gates
        self.bias_cell = random_array(0.5, -0.5, cell_count)
        self.bias_output = random_array(0.5, -0.5, dim)
        self.bias_forget = random_array(0.5, -0.5, cell_count)
        self.bias_input = random_array(0.5, -0.5, cell_count)
        # we need to consider weights and bias for all gates for backpropagation
        self.weight_cell_grad = np.zeros((cell_count, input_length))
        self.weight_output_grad = np.zeros((cell_count, input_length))
        self.weight_forget_grad = np.zeros((cell_count, input_length))
        self.weight_input_grad = np.zeros((cell_count, input_length))
        
        self.bias_cell_grad = np.zeros(cell_count)
        self.bias_output_grad = np.zeros(cell_count)
        self.bias_forget_grad = np.zeros(cell_count)
        self.bias_input_grad = np.zeros(cell_count)

    def parameters_update(self, lr = 0.3):
        """_summary_
        update all parameters for weights and bias based on gradient
        initialize all parameters from backpropagation as 0 <- this is important!!

        Args:
            lr (float, optional): learning rate. Defaults to 0.3.
        """
        # update all parameters for forward propagation
        self.weight_cell -= lr * self.weight_cell_grad
        self.weight_output -= lr * self.weight_output_grad
        self.weight_forget -= lr * self.weight_forget_grad
        self.weight_input -= lr * self.weight_input_grad
        self.bias_cell -= lr * self.bias_cell_grad
        self.bias_output -= lr * self.bias_output_grad
        self.bias_forget -= lr * self.bias_forget_grad
        self.bias_input -= lr * self.bias_input_grad

        # initialize all parameters for back propagation as 0, mimic the structure of forward propagation
        self.weight_cell_grad = np.zeros_like(self.weight_cell)
        self.weight_output_grad = np.zeros_like(self.weight_output)
        self.weight_forget_grad = np.zeros_like(self.weight_forget)
        self.weight_input_grad = np.zeros_like(self.weight_input)
        self.bias_cell_grad = np.zeros_like(self.bias_cell)
        self.bias_output_grad = np.zeros_like(self.bias_output)
        self.bias_forget_grad = np.zeros_like(self.bias_forget)
        self.bias_input_grad = np.zeros_like(self.bias_input)


class Cell_State:
    def __init__(self, cell_count, dim) -> None:
        """_summary_
        initialize cell state in LSTM, including:
        gate: gate in LSTM
        input_gate: input gate in LSTM
        forget_gate: forget gate in LSTM
        cell_state: cell state in LSTM
        output_gate: output gate in LSTM
        hidden_state: hidden state in LSTM
        hidden_state_grad: gradient of hidden state in LSTM
        hidden_state_cell_grad: gradient of cell state in LSTM

        Args:
            cell_count (_type_): count of cell in LSTM
            dim (_type_): dimension of input
        """
        self.gate = np.zeros(cell_count)
        self.input_gate = np.zeros(cell_count)
        self.forget_gate = np.zeros(cell_count)
        self.cell_state = np.zeros(cell_count)
        self.output_gate = np.zeros(cell_count)
        self.hidden_state = np.zeros(cell_count)
        self.hidden_state_grad = np.zeros(cell_count)
        self.hidden_state_cell_grad = np.zeros(cell_count)

class Node:
    def __init__(self, last_param, lstm_state):
        """_summary_
        initialize node in LSTM

        Args:
            last_param (_type_): parameters in nodes in LSTM,
            lstm_state (_type_): state in nodes in LSTM
        """
        self.last_param = last_param
        self.lstm_state = lstm_state
        self.input_length = None

    def current_node(self, input, prev_cell_state=None, prev_hidden_state=None):
        """_summary_
        calculate current node in LSTM
        based on previous cell state and previous hidden state

        y = Wx + b, where x is input, W is weight, b is bias

        Args:
            input (_type_): input
            prev_cell_state (_type_): cell state from previous node, V(0) by dafault
            prev_hidden_state (_type_): hidden state from previous node, h(0) by dafault
        """
        self.prev_cell_state = prev_cell_state
        self.prev_hidden_state = prev_hidden_state

        if prev_cell_state is None:
            prev_cell_state = np.zeros_like(self.lstm_state.cell_state)
        if prev_hidden_state is None:
            prev_hidden_state = np.zeros_like(self.lstm_state.hidden_state)
        
        # concatenate input and previous hidden state
        self.input_length = np.hstack([input, prev_hidden_state])

        # output of all gates
        # activation function: tanh for state gate and sigmoid for all other gates
        self.state.gate = np.tanh(np.dot(self.last_param.weight_cell, self.input_length) + self.last_param.bias_cell)
        self.state.input_gate = sigmoid(np.dot(self.last_param.weight_input, self.input_length) + self.last_param.bias_input)
        self.state.forget_gate = sigmoid(np.dot(self.last_param.weight_forget, self.input_length) + self.last_param.bias_forget)
        self.state.output_gate = sigmoid(np.dot(self.last_param.weight_output, self.input_length) + self.last_param.bias_output)

        self.state.cell_state = np.tanh(self.state.gate) * self.state.input_gate + self.state.forget_gate * prev_cell_state
        self.state.hidden_state = self.state.output_gate * np.tanh(self.state.cell_state)
        
    def current_node_grad(self, prev_grad_hidden, prev_grad_cell):
        """_summary_
        calculate gradient of current node in LSTM, including:
        grad_state_cell: gradient of cell state in LSTM
        grad_input_input: gradient of input gate in LSTM
        grad_forget_input: gradient of forget gate in LSTM
        grad_output_input: gradient of output gate in LSTM
        grad_cell_input: gradient of cell state in LSTM
        grad_input: gradient of input in LSTM

        Args:
            prev_grad_hidden (_type_): gradient of hidden state from previous node
            prev_grad_cell (_type_): gradient of cell state from previous node
        """
        grad_state_cell = prev_grad_cell + prev_grad_hidden * self.state.output_gate * tanh_derivative(np.tanh(self.lstm_state.cell_state))
        
        # d(gate)/d(x) = d(gate)/d(cell_state) * d(cell_state)/d(x)
        grad_input_input = self.lstm_state.gate * grad_state_cell * sigmoid_derivative(self.lstm_state.input_gate)
        grad_forget_input = self.prev_cell_state * grad_state_cell * sigmoid_derivative(self.lstm_state.forget_gate)
        grad_output_input = np.tanh(self.lstm_state.cell_state) * prev_grad_hidden * sigmoid_derivative(self.lstm_state.output_gate)
        grad_cell_input = self.lstm_state.input_state * grad_state_cell * tanh_derivative(self.lstm_state.gate)

        # update gradient of all parameters
        self.last_param.weight_cell_grad += np.outer(self.input_length, grad_cell_input)
        self.last_param.bias_cell_grad += grad_cell_input
        self.last_param.weight_input_grad += np.outer(self.input_length, grad_input_input)
        self.last_param.bias_input_grad += grad_input_input
        self.last_param.weight_forget_grad += np.outer(self.input_length, grad_forget_input)
        self.last_param.bias_forget_grad += grad_forget_input
        self.last_param.weight_output_grad += np.outer(self.input_length, grad_output_input)
        self.last_param.bias_output_grad += grad_output_input

        # update gradient of input
        grad_input = np.zeros_like(self.input_length)
        grad_input += np.dot(self.last_param.weight_input.T, grad_input_input)
        grad_input += np.dot(self.last_param.weight_forget.T, grad_forget_input)
        grad_input += np.dot(self.last_param.weight_output.T, grad_output_input)
        grad_input += np.dot(self.last_param.weight_cell.T, grad_cell_input)
        
        # update gradient of cell state and hidden state
        self.lstm_state.grad_cell_state = grad_state_cell * self.lstm_state.forget_gate
        self.lstm_state.grad_hidden_state = grad_input[self.last_param.dim:]


class LSTMnetwork:
    def __init__(self, lstm_param) -> None:
        """_summary_
        initialize LSTM network

        Args:
            lstm_param (_type_): parameters in LSTM
        """
        self.lstm_param = lstm_param
        self.lstm_node_list = []
        self.x_list = []

    def y_list(self, y_list, loss_layer):

        assert len(y_list) == len(self.x_list)
        idx = len(self.x_list) - 1
        loss = 0

        # calculate gradient and loss of last node
        loss += loss_layer.loss(self.lstm_node_list[idx].lstm_state.hidden_state, y_list[idx])
        grad_hidden = loss_layer.loss_grad(self.lstm_node_list[idx].lstm_state.hidden_state, y_list[idx])
        grad_cell = np.zeros(self.lstm_param.cell_count)
        self.lstm_node_list[idx].current_node_grad(grad_hidden, grad_cell)
        idx -= 1

        # calculate gradient and loss of other nodes
        while idx >= 0:
            loss += loss_layer.loss(self.lstm_node_list[idx].lstm_state.hidden_state, y_list[idx])
            grad_hidden = loss_layer.loss_grad(self.lstm_node_list[idx].state.hidden_state, y_list[idx])
            grad_hidden += self.lstm_node_list[idx+1].lstm_state.grad_hidden_state
            grad_cell = self.lstm_node_list[idx+1].lstm_state.grad_cell_state
            self.lstm_node_list[idx].current_node_grad(grad_hidden, grad_cell)
            idx -= 1

        return loss
    
    def x_list_cleaned(self):
        self.x_list = []

    def x_list_append(self, x):
        """_summary_
        append x to x_list
        if x_list is longer than lstm_node_list, append new node
        else create a new lstm state

        when idx == 0, then this is the first node
        else calculate gradient and loss of previous node

        Args:
            x (_type_): input
        """
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_node_list):
            lstm_state = Cell_State(self.lstm_param.cell_count, self.lstm_param.dim)
            self.lstm_node_list.append(Node(self.lstm_param, self.lstm_node_list[-1].lstm_state))

        idx = len(self.x_list) - 1
        if idx == 0:
            self.lstm_node_list[idx].current_node(x)
        else:
            prev_cell_state = self.lstm_node_list[idx-1].lstm_state.cell_state
            prev_hidden_state = self.lstm_node_list[idx-1].lstm_state.hidden_state
            self.lstm_node_list[idx].current_node(x, prev_cell_state, prev_hidden_state)

class LossLayer:
    @staticmethod
    def loss(y, y_hat):
        return np.sum(np.square(y - y_hat))

    @staticmethod
    def loss_grad(y, y_hat):
        grad = -2 * (y - y_hat)
        return grad
    

def main():
    """_summary_
    test for LSTM network 
    """
    np.random.seed(42)
    # LSTM parameters
    cell_count = 100
    dim = 50

    # initialize LSTM network
    lstm_param = LSTM_Parameter(cell_count, dim)
    lstm_network = LSTMnetwork(lstm_param)
    loss_layer = LossLayer()

    # target data
    y_list = [-1, 0.2, 0.5, 0.8, -0.3]

    # training Data
    input_val_arr = [np.random.random(dim) for _ in range(len(y_list))]

    # training model
    for i in range(1000):
        print("start iteration")

        for idx in range(len(y_list)):
            lstm_network.x_list_append(input_val_arr[idx])
        
        loss = lstm_network.y_list(y_list, loss_layer)
        print("Loss is {}".format(loss))

        # update parameters
        lstm_param.apply_grad(lr = 0.1) 

        # clear data
        lstm_network.x_list_cleaned()

        print("end iteration")



if __name__ == "__main__":
    main()