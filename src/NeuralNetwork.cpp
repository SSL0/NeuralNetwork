//
// Created by ssl0 on 4/2/20.
//

#include <NeuralNetwork.h>

NeuralNetwork::NeuralNetwork(const vector<int>& neuronsForEach) {
    // Create layers
    for(size_t i = 0; i < neuronsForEach.size(); i++) {
        layers.emplace_back(Layer(neuronsForEach[i]));
        if (i == 0) continue;

        // Create weights for bias on previous LAYER
        layers[i - 1].biasNeuron.createRandWeights(neuronsForEach[i]);

        // Create weights for neurons on previous LAYER
        for(Neuron &neuron : layers[i - 1].neurons) {
            neuron.createRandWeights(neuronsForEach[i]);
        }
    }
}

vector<Neuron>& NeuralNetwork::Predict(const vector<double>& input) {
    // Insert value to INPUT LAYER
    for(size_t i = 0; i < input.size(); i++){
        layers[0].neurons[i].result = input[i];
    }

    // Calculate other LAYERS
    GoThoughtLayers(1, layers.size(), [this](size_t currentLayer, size_t currentNeuron){
        double result = 0;
        for(Neuron &prevLayerNeuron : layers[currentLayer - 1].neurons){
            result += prevLayerNeuron.weights[currentNeuron] * prevLayerNeuron.result;
        }
        result += layers[currentLayer - 1].biasNeuron.weights[currentLayer];
        layers[currentLayer].neurons[currentNeuron].result = Neuron::sigmoidFunc(result);
    });
    return layers.back().neurons;
}

void NeuralNetwork::TrainBPE(const vector<TrainType>& trainData, int numOfEpoch, double learningRate) {
    // Array for MSE
    vector<double> errors;
    for(int i = 0; i <= numOfEpoch; i++) {
        for(const TrainType& set : trainData){
            // Make error for last neuron
            layers.back().neurons.back().error = Predict(set.first).back().result - set.second;
            // Back Propagation Error
            GoThoughtLayers(layers.size() - 1, 0, [this, &set, &learningRate, &errors](size_t currentLayer, size_t currentNeuron){
                for(Neuron &prevLayerNeuron : layers[currentLayer - 1].neurons){
                    double error  = layers[currentLayer].neurons[currentNeuron].error;
                    double actual = layers[currentLayer].neurons[currentNeuron].result;

                    // Add square error
                    errors.emplace_back(error * error);

                    double deltaWeight = (actual * (1 - actual)) * error;
                    prevLayerNeuron.weights[currentNeuron] -= prevLayerNeuron.result * deltaWeight * learningRate;

                    prevLayerNeuron.error = prevLayerNeuron.weights[currentNeuron] * deltaWeight;
                }
            });

        }

        printf("\rTrain progress: [%d%%] | MSE: %f", i * 100 / numOfEpoch, ComputeMSE(errors));
    }
}

void NeuralNetwork::GoThoughtLayers(size_t start, size_t end, const function<void(size_t, size_t)>& action) {
    int direction = 0;
    if(start < end) direction = 1;
    else direction = -1;

    for(size_t layer = start; (start < end) ? layer < end : layer > end; layer += direction){
        for(size_t neuron = 0; neuron < layers[layer].neurons.size(); neuron++){
            action(layer, neuron);
        }
    }
}

double NeuralNetwork::ComputeMSE(const vector<double>& errors) {
    double errorAverage = 0.0;
    for(double err : errors){
        errorAverage += err;
    }
    return errorAverage / errors.size();
}
