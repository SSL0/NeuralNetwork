//
// Created by ssl0 on 4/2/20.
//

#pragma once

#include <iostream>
#include <vector>

#include <Neuron.h>

using namespace std;

class Layer {
public:
    Layer(int numOfNeuron);

    double sum = 0.0;

    vector<Neuron> neurons;
    Neuron biasNeuron;
};
