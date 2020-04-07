//
// Created by ssl0 on 4/2/20.
//

#include "Neuron.h"

Neuron::Neuron() = default;


void Neuron::createRandWeights(int num) {
    for(int i = 0; i < num; i++){
        weights.push_back(getRandVal());
    }
}

double Neuron::sigmoidFunc(double x) {
    return 1 / (1 + exp(-x));
}

void Neuron::createWeights(vector<double> weightsArr) {
    weights = move(weightsArr);
}

double Neuron::getRandVal() {
    return randGenerator(eng);
}
