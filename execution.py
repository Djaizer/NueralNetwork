import numpy as np
import sys

from NueralNetwork import NueralNetwork


def MSE(y, Y):
    return np.mean((y - Y) ** 2)


train = [
    ([0, 0, 0], 0),
    ([0, 0, 1], 1),
    ([1, 0, 0], 1),
    ([0, 1, 0], 0),
    ([0, 1, 1], 0),
    ([1, 0, 1], 1),
    ([1, 1, 0], 0),
    ([1, 1, 1], 1),
]

epochs = 5000
learning_rate = 0.01




network = NueralNetwork(learning_rate=learning_rate)

for e in range(epochs):
    inputs_ = []
    correct_predictions = []
    for input_stat, correct_predict in train:
        network.train(np.array(input_stat), correct_predict)
        inputs_.append(np.array(input_stat))
        correct_predictions.append(np.array(correct_predict))

    train_loss = MSE(network.predict(np.array(inputs_).T), np.array(correct_predictions))
    sys.stdout.write("\rProgress: {}, Training losses: {}".format(str(100 * e/float(epochs))[:4], str(train_loss)[:5]))


    for input_stat, correct_predict in train:
            print("For input: {} the prediction is: {}, expected: {}".format(
                str(input_stat),
                str(network.predict(np.array(input_stat)) > .5),
                str(correct_predict == 1)))


print("etwork.weights_0_1: ", network.weights_0_1)

print("etwork.weights_1_2: ", network.weights_1_2)





