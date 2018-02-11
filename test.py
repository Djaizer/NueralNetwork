import numpy as nm
girl = 0.0
vodka = 0.0
rain = 1.0

def activation_function(res):
 if res >= 0.5:
    return 1
 else:
     return 0


def predict(vodka, rain, girl):
    income = nm.array([vodka, rain, girl])
    neuron_1 = [0.25, 0.25, 0]
    neuron_2 = [0.5, -0.4, 0.9]
    layer_1 = nm.array([neuron_1, neuron_2])
    layer_3 = nm.array([-1, 1])

    print("layer_1: " + str(layer_1))

    after_1_layer = nm.dot(layer_1, income)
    print("after_1_layer: " + str(after_1_layer))

    layer_2 = nm.array([activation_function(x) for x in after_1_layer])
    print("layer_2: " + str(layer_2))

    result = nm.dot(layer_3, layer_2)
    print("result : " + str(result))

    return activation_function(result) == 1

print(predict(vodka, rain, girl))
