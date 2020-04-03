//
// Created by ssl0 on 4/2/20.
//

#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(vector<int> neuronsForEach) {
    // Create layers
    for(int layerIndx = 0; layerIndx < neuronsForEach.size(); layerIndx++) {
        layers.emplace_back(Layer(neuronsForEach[layerIndx]));

        // Create weights for neurons on previous layer
        if (layerIndx > 0) {
            for(Neuron &neuron : layers[layerIndx - 1].neurons){
                neuron.CreateRandWeights(neuronsForEach[layerIndx]);
            }
        }
    }
    return;
}

double NeuralNetwork::Predict(vector<double> input) {
    // Insert value to INPUT LAYER
    for(int neuronIndx = 0; neuronIndx < input.size(); neuronIndx++){
        layers[0].neurons[neuronIndx].Result = input[neuronIndx];
    }

    // Calculate other LAYERS
    for(int layer = 1; layer < layers.size(); layer++){
        for(int neuron = 0; neuron < layers[layer].neurons.size(); neuron++){
            double result = 0;
            for(Neuron &prevLayerNeuron : layers[layer - 1].neurons){
                result += prevLayerNeuron.weights[neuron] * prevLayerNeuron.Result;
            }
            layers[layer].neurons[neuron].Result = Neuron::SigmoidFunc(result);
        }
    }
    return layers.back().neurons.back().Result;
}

void NeuralNetwork::TrainBPE(vector<TrainingType> trainingData, int numOfEpoch, double learningRate) {
    vector<double> errors;
    for(int i = 0; i < numOfEpoch; i++){
        for(TrainingType &set : trainingData){
            for(int layer = layers.size() - 1; layer > 0; layer--){
                for(int neuron = 0; neuron < layers[layer].neurons.size(); neuron++){
                    for(Neuron &prevLayerNeuron : layers[layer - 1].neurons){
                        double actual = Predict(set.first);
                        double error = actual - set.second;
                        errors.push_back(error * error);
                        double deltaWeight = actual * (1 - actual) * error;
                        prevLayerNeuron.weights[neuron] -= prevLayerNeuron.Result * deltaWeight * learningRate;
                        int a = 0;
                    }
                }
            }
        }
        double obj = 0;
        for(double &err : errors){
            obj += err;
        }
        cout << "MSE: " << obj / errors.size() << endl;
    }
}