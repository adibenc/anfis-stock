import time
# import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt
from anfis import ANFIS


# Mackey-Glass series computation
def mackey(n_iters):
    x = np.zeros((n_iters,))
    x[0:30] = 0.23 * np.ones((30,))
    t_s = 30
    for i in range(30, n_iters - 1):
        a = x[i]
        b = x[i - t_s]
        y = ((0.2 * b) / (1 + b ** 10)) + 0.9 * a
        x[i + 1] = y
    return x


# Generate dataset
D = 4  # number of regressors
T = 1  # delay
N = 2000  # Number of points to generate
mg_series = mackey(N)[499:]  # Use last 1500 points
data = np.zeros((N - 500 - T - (D - 1) * T, D))
labels = np.zeros((N - 500 - T - (D - 1) * T,))

for t in range((D - 1) * T, N - 500 - T):
    data[t - (D - 1) * T, :] = [mg_series[t - 3 * T], mg_series[t - 2 * T], mg_series[t - T], mg_series[t]]
    labels[t - (D - 1) * T] = mg_series[t + T]
trainx = data[:labels.size - round(labels.size * 0.3), :]
trainy = labels[:labels.size - round(labels.size * 0.3)]
testx = data[labels.size - round(labels.size * 0.3):, :]
testy = labels[labels.size - round(labels.size * 0.3):]


def anfisTrain(num_epochs, trainx, trainy, testx, testy, fis = None):
    if fis == None:
        fis = ANFIS(n_inputs=D, n_rules=m, learning_rate=alpha)
    # Initialize session to make computations on the Tensorflow graph
    with tf.Session() as sess:
        # Initialize model parameters
        sess.run(fis.init_variables)
        trn_costs = []
        val_costs = []
        time_start = time.time()
        for epoch in range(num_epochs):
            #  Run an update step
            trn_loss, trn_pred = fis.train(sess, trainx, trainy)
            # Evaluate on validation set
            val_pred, val_loss = fis.infer(sess, testx, testy)
            if epoch % 10 == 0:
                print("Train cost after epoch %i: %f" % (epoch, trn_loss))
            if epoch == num_epochs - 1:
                time_end = time.time()
                print("Elapsed time: %f" % (time_end - time_start))
                print("Validation loss: %f" % val_loss)
                # Plot real vs. predicted
                pred = np.vstack((np.expand_dims(trn_pred, 1), np.expand_dims(val_pred, 1)))
                plt.figure(1)
                plt.plot(mg_series)
                plt.plot(pred)
            trn_costs.append(trn_loss)
            val_costs.append(val_loss)
        # Plot the cost over epochs
        plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(np.squeeze(trn_costs))
        plt.title("Training loss, Learning rate =" + str(alpha))
        plt.subplot(2, 1, 2)
        plt.plot(np.squeeze(val_costs))
        plt.title("Validation loss, Learning rate =" + str(alpha))
        plt.ylabel('Cost')
        plt.xlabel('Epochs')
        # Plot resulting membership functions
        fis.plotmfs(sess)
        plt.show()

# ANFIS params and Tensorflow graph initialization
m = 16  # number of rules
alpha = 0.01  # learning rate
# Training
num_epochs = 100

anfisTrain(num_epochs, trainx, trainy, testx, testy,
    ANFIS(n_inputs=D, n_rules=m, learning_rate=alpha)
)