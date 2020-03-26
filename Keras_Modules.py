import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt

from pylab import rcParams
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, UpSampling2D
from keras.layers import Flatten, Activation


def load_data():
    dataset1 = np.load(r'C:\Users\Darya\Documents\GitHub\CERN\images_10000.npy')
    dataset2 = np.load(r'C:\Users\Darya\Documents\GitHub\CERN\images_10000-15000.npy')
    dataset = np.concatenate([dataset1, dataset2], 0)
    dataset = dataset[:, 20:105, 20:105]
    X_train = dataset[:int(0.85 * dataset.shape[0])]
    X_test = dataset[int(0.85 * dataset.shape[0]):]
    X_train[X_train < 1.e-3] = 0.
    return X_train, X_test


def rand_weight(l1, l2):
    return tf.Variable(tf.random.normal([l1, l2]))

def rand_bias(l):
    return tf.Variable(tf.random.normal([l]))

def inner_prod1(l1, l2, l3):
    return tf.nn.relu(tf.add(tf.matmul(l1, l2), l3))

def inner_prod2(l1, l2, l3):
    return tf.matmul(l1, l2) + l3

def plot_img(X):
    plt.figure()
    #plt.subplot(1, 5, i+1)
    ax = plt.gca()
    im = plt.imshow(X, vmin=1.e-3, cmap='viridis', norm=LogNorm(), alpha=0.9)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, label='E', fraction=0.055, pad=0.04)
    plt.axis('off')
    plt.show()
    
def get_output_layer(n_nodes_inpl, n_nodes_hl1, n_nodes_hl2, n_nodes_outl):
    # 85 * 85 *140 weights and 140 biases in layer 1 
    hidden_1_layer_vals = {'weights':rand_weight(n_nodes_inpl, n_nodes_hl1), 'biases':rand_bias(n_nodes_hl1)}

    # 140*140 weights and 140 biases for layer 2 
    hidden_2_layer_vals = {'weights':rand_weight(n_nodes_hl1, n_nodes_hl2), 'biases': rand_bias(n_nodes_hl2)}

    # 140*85*85 weights and 85*85 biases for the output
    output_layer_vals = {'weights':rand_weight(n_nodes_hl2,n_nodes_outl),'biases':rand_bias(n_nodes_outl)}

    # the input layer for images of shape 85x85 
    input_layer = tf.placeholder('float', [None, 85*85])

    # inner product of the input and the weights + the biases for layer 1 
    layer_1 = inner_prod1(input_layer, hidden_1_layer_vals['weights'], hidden_1_layer_vals['biases'])
    # inner product of output of layer 1 and the weights + the biases for layer 2 
    layer_2 = inner_prod1(layer_1, hidden_2_layer_vals['weights'], hidden_2_layer_vals['biases'])

    # inner product of output of layer 2 and the weights + the biases for the output layer  
    output_layer = inner_prod2(layer_2, output_layer_vals['weights'], output_layer_vals['biases'])
    return input_layer, output_layer


def initialize(shape, output_layer, learn_rate):
    output_true = tf.placeholder('float', [None, shape])
    # the cost function
    meansq = tf.reduce_mean(tf.square(output_layer - output_true))
    # the optimizer 
    optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(meansq)
    return output_true, meansq, optimizer

def initialize_sess():
    init = tf.global_variables_initializer()
    #sess = tf.Session()
    sess = tf.InteractiveSession()
    sess.run(init)
    return sess

def encoder_decoder(img_shape1, img_shape2, num_channels):
    # The encoder 
    input_img = Input(shape=(85, 85, 1))  
    x = Conv2D(16, (6, 6), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (6, 6), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(1, (6, 6), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # The decoder 
    x = Conv2D(1, (6, 6), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (6, 6), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (6, 6), activation='relu', padding = 'same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (4, 4), activation='sigmoid', padding='valid')(x)
    return input_img, decoded