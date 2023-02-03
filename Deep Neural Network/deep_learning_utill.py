# This file is written by Alireza Kamyab
# Copy right 2023;
import numpy as np
import matplotlib.pyplot as plt

def initialize_parameters(layer_dims):
    """
    arguments:
        layer_dims: A list that contains the size of each layer
        
    returns:
        dict: parameters for each layer
    """
    
    params = {}
    L = len(layer_dims)
    for l in range(1, L):
        # This uses He initialization
        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return params


def initialize_velocity(parameters):
    v = {}
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        v['dW' + str(l)] = np.zeros_like(parameters['W' + str(l)])
        v['db' + str(l)] = np.zeros_like(parameters['b' + str(l)])
    
    return v


def initialize_adam(parameters):
    v = {}
    s = {}
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        v['dW' + str(l)] = np.zeros_like(parameters['W' + str(l)])
        v['db' + str(l)] = np.zeros_like(parameters['b' + str(l)])
        s['dW' + str(l)] = np.zeros_like(parameters['W' + str(l)])
        s['db' + str(l)] = np.zeros_like(parameters['b' + str(l)])
    
    return v, s


def linear_forward(A, W, b):
    """
    arguments:
        A: activations from previous layer or inputs, shape=(size of l-1, m)
        W: weights from layer l, shape=(size of layer l, size of layer l-1)
        b: intercepts from layer l, shape(size of layer l, 1)
        
    returns:
        Z: forward calculation
        cache: The same arguments useful for backward propagation
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    
    return Z, cache


def sigmoid(Z):
    """
    arguments:
        Z: a matrix calculated in "linear_forward" step
        
    returns:
        A: activation function of sigmoid(Z)
        cache: The same arguments useful for backward propagation
    """
    cache = Z
    A = 1 / (1 + np.exp(-Z))
    
    return A, cache


def relu(Z):
    """
    arguments:
        Z: a matrix calculated in "linear_forward" step
        
    returns:
        A: activation function of relu(Z)
        cache: The same arguments useful for backward propagation
    """
    cache = Z
    A = np.maximum(0, Z)
    
    return A, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    arguments:
        A: activations from previous layer or inputs, shape=(size of l-1, m)
        W: weights from layer l, shape=(size of layer l, size of layer l-1)
        b: intercepts from layer l, shape(size of layer l, 1)
        activation: A string that specifies what type of activiation function to use (sigmoid or ReLU)
    
    returns:
        A: activation calculated
        cache: to be used for backward propagation
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    else:
        A, activation_cache = Z, Z
    
    cache = (linear_cache, activation_cache)
    
    return A, cache


def forward_prop(X, parameters, keep_prob=1.):
    """
    arguments:
        X: numpy array, shape=(input size, input examples i.e m)
        parameters: output of initialize_parameters_deep
        keep_prob: the probability of which each perceptron is kept
        
    returns:
        AL: activation value from the output layer
        caches: Caches from calling Linear activation forward, the size is L since there are L layers
    """
    caches = []
    L = len(parameters) // 2
    A_prev = X
    
    for l in range(1, L):
        Wl = parameters['W' + str(l)]
        bl = parameters['b' + str(l)]
        A, cache = linear_activation_forward(A_prev, Wl, bl, 'relu')
        caches.append(cache)
        
        # Regularization : Drop out
        D = np.random.rand(A.shape[0], A.shape[1])
        D = (D <= keep_prob).astype(int)
        A = A * D
        A /= keep_prob
        
        A_prev = A
        
    WL = parameters['W' + str(L)]
    bL = parameters['b' + str(L)]
    AL, cache = linear_activation_forward(A_prev, WL, bL, 'sigmoid')
    caches.append(cache)
    
    return AL, caches


def compute_frobenius_norm(parameters):
    L = len(parameters) // 2
    summation = 0
    for l in range(1, L + 1):
        W_kj = parameters['W' + str(l)]
        summation += np.sum(np.square(W_kj))
    return np.sqrt(summation)


def compute_derivative_frobenius_norm(parameter, parameters):
    l2norm = compute_frobenius_norm(parameters)
    return parameter / l2norm


def compute_cost(AL, Y, parameters=None, lambda_=0., epsilon=1e-15):
    """
    arguments:
        AL: nparray; the output of the neural network
        Y: array of labels
        parameters: dictionary of the parameters containing Ws and bs to calculate ||W||F
        lambda_: is a scalar as regularization parameter
    returns:
        decimal number specifing the cost
    """
    Y = Y.reshape(AL.shape)
    m = Y.shape[1]
    AL = np.clip(AL, epsilon, 1 - epsilon)
    cost = np.multiply(-Y, np.log(AL)) - np.multiply(1 - Y, np.log(1 - AL))
    cost = np.sum(cost, axis=1) / (m)
    
    if parameters is not None:
        reg_cost = (lambda_ * compute_frobenius_norm(parameters) ** 2) / (2 * m)
        cost += reg_cost
        
    return np.squeeze(cost)


def sigmoid_backward(dA, cache):
    """
    arguments:
        dA: post-activation gradient, of any shape
        cache: it is "Z" that we stored to use later in back prop
    
    returns:
        derivative of loss function with respect to Z
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    
    return dZ


def relu_backward(dA, cache):
    """
    arguments:
        dA: post-activation gradient, of any shape
        cache: it is "Z" that we stored to use later in back prop
    
    returns:
        derivative of loss function with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    
    return dZ


def linear_backward(dZ, cache):
    """
    arguments:
        dZ: gradient of the cost with respect to the linear output i.e Z
        cache: the tupe (A_prev, W, b)
    
    returns:
        dW: gradient of the cost with respect to the weights i.e W
        db: gradient of the cost with respect to the intercepts i.e b
        dA_prev: gradient of the cost with respect to the activation function in the previous layer
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.matmul(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.matmul(W.T, dZ)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    arguments:
        dA: post-activation gradient for current layer l
        cache: tuple contaning (linear_cache, activation_cache)
        activation: activation function go to be used in this layer
        
    returns:
        dW: gradient of the cost with respect to the weights i.e W
        db: gradient of the cost with respect to the intercepts i.e b
        dA_prev: gradient of the cost with respect to the activation function in the previous layer
    """
    
    (linear_cache, activation_cache) = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def get_mini_batches(X, y, mini_batch_size = 64, random=True):
    """
    Partitions the X and y into mini-batches
    
    arguments:
        X: ndarray contaning m examples that are to be partitioned
        y: (1, m) array containing labels that are to be partitioned in parallel to X
        mini_batches_size: scalar (int) specifying the size of each partition
        random: (boolean) shuffels the X and y
        
    returns:
        (list) containing partitions
    """
    
    m = X.shape[1]
    mini_batches = []
    
    if random:
        permutations = np.random.permutation((m))
        X = X[:, permutations]
        y = y[:, permutations].reshape((1, m))
        
    full_partitions = m // mini_batch_size
    
    for p in range(full_partitions):
        X_part = X[:, p * mini_batch_size : (p + 1) * mini_batch_size]
        y_part = y[:, p * mini_batch_size : (p + 1) * mini_batch_size]
        
        mini_batch = (X_part, y_part)
        mini_batches.append(mini_batch)
        
    if m % mini_batch_size != 0:
        X_part = X[:, full_partitions * mini_batch_size :]
        y_part = y[:, full_partitions * mini_batch_size :]
        
        mini_batch = (X_part, y_part)
        mini_batches.append(mini_batch)
    
    return mini_batches


def backward_prop(AL, Y, caches, parameters, lambda_=0, keep_prob=1., epsilon=1e-15):
    """
    arguments:
        AL: predicted outputs from the neural network
        Y: correct values
        caches: cached values in each step of linear forward and linear activation forward
        parameters: dictionary of the parameters containing Ws and bs to calculate ||W||F
        lambda_: is a scalar as regularization parameter
    returns:
        gradients that are --> dA_prev, dW, db for each layer
    """
    grads = {}
    Y = Y.reshape(AL.shape)
    L = len(caches)
    m = Y.shape[1]
    
    current_cache = caches[L - 1]
    AL = np.clip(AL, epsilon, 1 - epsilon)
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    dA_prev, dW, db = linear_activation_backward(dAL, current_cache, 'sigmoid')
    grads['dW' + str(L)] = dW
    grads['db' + str(L)] = db
    grads['dA' + str(L - 1)] = dA_prev
    
    for l in reversed(range(L - 1)):
        current_cache = caches[l] # caches[l] means, caches from layer l+1
        dA_prev, dW, db = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, 'relu')
        grads['dW' + str(l + 1)] = dW
        grads['db' + str(l + 1)] = db
        grads['dA' + str(l)] = dA_prev
        
    
    # This does not calculate d||W||F/dW it calculates d||W||F^2/dW
    for l in range(1, L + 1):
        grads['dW' + str(l)] += lambda_/m * parameters['W' + str(l)]
        
        
    # Regularization : Dropout
    for l in range(1, L):
        grad_A = grads['dA' + str(l)]
        D = np.random.rand(grad_A.shape[0], grad_A.shape[1])
        D = (D <= keep_prob).astype(int)
        grad_A = grad_A * D
        grads['dA' + str(l)] = grad_A
        
    return grads


def update_parameters(params, grads, learning_rate):
    """
    arguments
        params: is the parameters that we initialized
        grads: gradients that are calculated in "L_model_backward"
        learning_rate: a decimal illustrating how much of the gradient should effect params
    
    return:
        updated parameters
    """
    L = len(params) // 2
    parameters = params.copy()
    
    for l in range(1, L + 1):
        parameters['W' + str(l)] = params['W' + str(l)] - learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] = params['b' + str(l)] - learning_rate * grads['db' + str(l)]
    
    return parameters


def update_parameters_momentum(params, grads, learning_rate, v, beta):
    """
    arguments
        params: is the parameters that we initialized
        grads: gradients that are calculated in "L_model_backward"
        learning_rate: a decimal illustrating how much of the gradient should effect params
        v: velocities
        beta: scalar, momentum hyperparameter
    
    return:
        updated parameters using momentum
        v: velocities
    """
    
    L = len(params) // 2
    for l in range(1, L + 1):
        v['dW' + str(l)] = beta * v['dW' + str(l)] + (1 - beta) * grads['dW' + str(l)]
        v['db' + str(l)] = beta * v['db' + str(l)] + (1 - beta) * grads['db' + str(l)]
        
        params['W' + str(l)] -= learning_rate * v['dW' + str(l)]
        params['b' + str(l)] -= learning_rate * v['db' + str(l)]
        
    return params, v


def update_parameters_adam(parameters, grads, learning_rate, t, v, s, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam
    
    Arguments:
        parameters -- python dictionary containing your parameters:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients for each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        t -- Adam variable, counts the number of taken steps
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
        learning_rate -- the learning rate, scalar.
        beta1 -- Exponential decay hyperparameter for the first moment estimates 
        beta2 -- Exponential decay hyperparameter for the second moment estimates 
        epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
        parameters -- python dictionary containing your updated parameters 
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    v_corrected, s_corrected = {}, {}
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        v['dW' + str(l)] = beta1 * v['dW' + str(l)] + (1 - beta1) * grads['dW' + str(l)]
        v['db' + str(l)] = beta1 * v['db' + str(l)] + (1 - beta1) * grads['db' + str(l)]
        v_corrected['dW' + str(l)] = v['dW' + str(l)] / (1 - np.power(beta1, t))
        v_corrected['db' + str(l)] = v['db' + str(l)] / (1 - np.power(beta1, t))
        
        s['dW' + str(l)] = beta2 * s['dW' + str(l)] + (1 - beta2) * np.power(grads['dW' + str(l)], 2)
        s['db' + str(l)] = beta2 * s['db' + str(l)] + (1 - beta2) * np.power(grads['db' + str(l)], 2)
        s_corrected['dW' + str(l)] = s['dW' + str(l)] / (1 - np.power(beta2, t))
        s_corrected['db' + str(l)] = s['db' + str(l)] / (1 - np.power(beta2, t))
        
        parameters['W' + str(l)] -= learning_rate *  v_corrected['dW' + str(l)] / (np.sqrt(s_corrected['dW' + str(l)]) + epsilon)
        parameters['b' + str(l)] -= learning_rate * v_corrected['db' + str(l)] / (np.sqrt(s_corrected['db' + str(l)]) + epsilon)
    
    return parameters, v, s
        
# Learning rate decay
def update_learning_rate(epoch_num, decay_rate, alpha_0):
    """
    Updates Learning rate according to epoch number
    
    arguments:
        epoch_num: number of passes through traning set (scalar)
        decay_rate: dacay rate (scalar)
        alpha_0: initial value of learning_rate (scalar)
        
    returns:
        new value of alpha
    """
    
    return alpha_0 / (1 + decay_rate * epoch_num)


def schedule_learning_rate(epoch_num, decay_rate, alpha_0, time_interval=100):
    """
    Updates Learning rate according to epoch number
    
    arguments:
        epoch_num: number of passes through traning set (scalar)
        decay_rate: dacay rate (scalar)
        alpha_0: initial value of learning_rate (scalar)
        time_interval -- Number of epochs where you update the learning rate.
        
    returns:
        new value of alpha
    """
    
    return alpha_0 / (1 + decay_rate * np.floor(epoch_num / time_interval))


# Model
def model(X, y, layer_dims, optimizer, mini_batch_size=64, alpha=0.0001, num_epochs=1000, beta=0.9, beta1=0.9, 
          beta2=0.999, epsilon=1e-8, verbos=True, decay_rate=1, decay=None):
    """
    Arguments:
        X -- input data, of shape
        Y -- true "label" vector, of shape (1, number of examples)
        layers_dims -- python list, containing the size of each layer
        optimizer: the name of the optimizer
                gd: Gradient descent
                momentum: Gradient descent with momentum
                adam: Adam optimizer
        mini_batch_size -- the size of a mini batch
        alpha -- the learning rate, scalar.
        num_epochs -- number of epochs
        beta -- Momentum hyperparameter
        beta1 -- Exponential decay hyperparameter for the past gradients estimates 
        beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
        epsilon -- hyperparameter preventing division by zero in Adam updates
        verbos -- True to print the cost every 1000 epochs
        decay_rate -- Rate which alpha decays
        decay -- decay function used to decay alpha

    Returns:
        parameters -- python dictionary containing your updated parameters 
        costs -- python list containing all the costs on each epoch
    """
    
    costs = []
    m = X.shape[1]
    y = y.reshape((1, m))
    parameters = initialize_parameters(layer_dims)
    alpha_0 = alpha
    
    if optimizer == 'gd': None
    elif optimizer == 'momentum':
        v = initialize_velocity(parameters)
    elif optimizer == 'adam':
        t = 0
        v, s = initialize_adam(parameters)
    
    for i in range(1, num_epochs + 1):
        mini_batches = get_mini_batches(X, y, mini_batch_size, random=True)
        net_cost = 0
        for mini_batch in mini_batches:
            X_part, y_part = mini_batch
            AL, caches = forward_prop(X_part, parameters)
            grads = backward_prop(AL, y_part, caches, parameters)
            net_cost += compute_cost(AL, y_part)
        
            if optimizer == 'gd':
                parameters = update_parameters(parameters, grads, alpha)
            elif optimizer == 'momentum':
                parameters, v = update_parameters_momentum(parameters, grads, alpha, v, beta)
            elif optimizer == 'adam':
                t += 1
                parameters, v, s = update_parameters_adam(parameters, grads, alpha, t, v, s, beta1, beta2, epsilon)
        
        cost_avg = net_cost / (m / mini_batch_size)
        costs.append(cost_avg)
        
        if decay:
            alpha = decay(i, decay_rate, alpha_0)
            
        if verbos and i % 100 == 0:
            if not decay: print(f'\rEpoch {i}: cost is {cost_avg}', end='')
            else: print(f'\rEpoch {i}: cost is {cost_avg} -- alpha is {alpha}', end='')
            if i % 1000 == 0:
                print('')
            
    return parameters, costs

# Gradient Test Part
def vector_to_dictionary(vector, parameters):
    """
    Convert a vector into a dictionary of parameters
    Arguments:
    vector -- The vector to convert
    parameters -- A dictionary of parameters, containing the shapes of the parameters
    
    Returns:
    dictionary -- A dictionary of parameters from the vector
    """
    L = len(parameters) // 2
    index = 0
    for l in range(1, L + 1):
        shape = parameters['W' + str(l)].shape
        size = np.prod(shape)
        parameters['W' + str(l)] = np.reshape(vector[index:index + size], shape)
        index += size
        
        shape = parameters['b' + str(l)].shape
        size = np.prod(shape)
        parameters['b' + str(l)] = np.reshape(vector[index:index + size], shape)
        index += size
    return parameters


def dict_to_vector(params):
    """Flatten dictionary to vector."""
    L = len(params) // 2
    vec = np.array([])
    for l in range(1, L + 1):
        vec = np.concatenate((vec, params['W' + str(l)].flatten()))
        vec = np.concatenate((vec, params['b' + str(l)].flatten()))
    return vec


def grads_to_vec(grads, L):
    """Flatten gradients to vector."""
    vec = np.array([])
    for l in range(1, L + 1):
        vec = np.concatenate((vec, grads['dW' + str(l)].flatten()))
        vec = np.concatenate((vec, grads['db' + str(l)].flatten()))
    return vec


def gradient_check(X, Y, parameters, grads, epsilon=1e-7):
    """
    Implement gradient check
    
    Arguments:
    X -- Input data, shape (input size, number of examples)
    Y -- True labels, shape (output size, number of examples)
    parameters -- Output of initialize_parameters_deep
    grads -- Output of backward_propagation
    epsilon -- Small epsilon value for numeric approximation of gradient
    
    Returns:
    difference -- Difference between numeric gradient and backprop gradient
    """
    parameters_values = dict_to_vector(parameters)
    num_parameters = parameters_values.shape[0]
    grad = grads_to_vec(grads, len(parameters) // 2).reshape(-1, 1)
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    for i in range(num_parameters):
        thetaplus = np.copy(parameters_values)                                      
        thetaplus[i] = thetaplus[i] + epsilon   
        AL, _ = forward_prop(X, vector_to_dictionary(thetaplus, parameters))                 
        J_plus[i] = compute_cost(AL, Y, parameters)     
        
        thetaminus = np.copy(parameters_values)                                     
        thetaminus[i] = thetaminus[i] - epsilon                               
        AL, _ = forward_prop(X, vector_to_dictionary(thetaminus, parameters))                
        J_minus[i] = compute_cost(AL, Y, parameters)                                            
        
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)                    
    
    numerator = np.linalg.norm(grad - gradapprox)                                   
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                
    difference = numerator / denominator   
    
    if difference > 2 * epsilon:
        print("There is a mistake in the backward propagation! difference = " + str(difference))
    else:
        print("Your backward propagation works perfectly fine! difference = " + str(difference))
    
    return difference


# Plotting
def plot(X_in, y_in, params, norm, x_min=-2, x_max=2, y_min=-2, y_max=2):
    plt.figure()
    
    x = np.linspace(x_min, x_max, 1000)
    y = np.linspace(y_min, y_max, 1000)
    X, Y = np.meshgrid(x, y)
    xx, yy = X.flatten().reshape(1, -1), Y.flatten().reshape(1, -1)
    data = np.vstack((xx, yy)).T
    data_n = norm.transform(data)
    Z, _ = forward_prop(data_n.T, params)
    plt.contourf(X, Y, Z.reshape(X.shape), cmap='OrRd')
    plt.colorbar()
    
    plt.scatter(X_in[y_in == 1, 0], X_in[y_in == 1, 1], alpha=0.75, label='Y = 1', c='yellow');
    plt.scatter(X_in[y_in == 0, 0], X_in[y_in == 0, 1], alpha=0.75, label='Y = 0', c='red');
    
    ax = plt.gca()
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    plt.legend()