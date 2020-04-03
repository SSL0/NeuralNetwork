//
// Created by ssl0 on 4/2/20.
//

#include "Neuron.h"

Neuron::Neuron() = default;


void Neuron::CreateRandWeights(int num) {
    for(int i = 0; i < num; i++){
        weights.push_back((double)rand() / RAND_MAX);
    }
}

double Neuron::SigmoidFunc(double x) {
    return 1 / (1 + exp(-x));
}

void Neuron::CreateWeights(vector<double> weightsArr) {
    weights = weightsArr;
}
