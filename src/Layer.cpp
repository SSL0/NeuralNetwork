//
// Created by ssl0 on 4/2/20.
//

#include "Layer.h"

Layer::Layer(int numOfNeuron) {
    for(int neuron = 0; neuron < numOfNeuron; neuron++){
        neurons.emplace_back(Neuron());
    }
}