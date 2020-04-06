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

vector<double>& NeuralNetwork::predict(const vector<double>& input) {
    // Insert value to INPUT LAYER
    for(size_t i = 0; i < input.size(); i++){
        layers[0].neurons[i].result = input[i];
    }

    // Calculate other LAYERS
    goThoughtLayers(1, layers.size(), [this](size_t currentLayer, size_t currentNeuron) {
        double result = 0;
        for (Neuron &prevLayerNeuron : layers[currentLayer - 1].neurons) {
            result += prevLayerNeuron.weights[currentNeuron] * prevLayerNeuron.result;
        }
        result += layers[currentLayer - 1].biasNeuron.weights[currentLayer];
        layers[currentLayer].neurons[currentNeuron].result = Neuron::sigmoidFunc(result);
    });

    results.clear();

    for(Neuron& neuron : layers.back().neurons){
        results.emplace_back(neuron.result);
    }

    return results;
}

void NeuralNetwork::trainBP(int numOfEpoch, double learningRate) {
    // Array for MSE
    vector<double> errors;
    File out("errors.txt");
    for(int epoch = 0; epoch <= numOfEpoch; epoch++) {
        for(const TrainType& set : trainingData){
            predict(set.first);

            // Make error for neurons on LAST LAYER
            for(size_t i = 0; i < layers.back().neurons.size(); i++){
                layers.back().neurons[i].error =  set.second[i] - layers.back().neurons[i].result;
            }

            // Back Propagation Error
            goThoughtLayers(layers.size() - 1, 0,[this, &set, &learningRate, &errors](size_t currentLayer, size_t currentNeuron) {
                double error = layers[currentLayer].neurons[currentNeuron].error;
                double actual = layers[currentLayer].neurons[currentNeuron].result;

                // Add square error
                errors.emplace_back(error * error);

                for (Neuron &prevLayerNeuron : layers[currentLayer - 1].neurons) {
                    double deltaWeight = (actual * (1 - actual)) * error;

                    prevLayerNeuron.weights[currentNeuron] += prevLayerNeuron.result * deltaWeight * learningRate;
                    prevLayerNeuron.error = prevLayerNeuron.weights[currentNeuron] * deltaWeight;

                    layers[currentLayer - 1].biasNeuron.weights[currentNeuron] += deltaWeight * learningRate;
                }
            });

        }
        out.write(to_string(computeMSE(errors) * 100));
        printf("\rTrain progress: [%d%%] | Errors: %f%% MSE: %f", epoch * 100 / numOfEpoch, computeMSE(errors) * 100, computeMSE(errors));
    }

    cout << endl << "Results after train: " << endl;
    for(auto & set : trainingData){
        predict(set.first);

        // Print string type:
        // Expected: [ %f...%f ] Output: [ %f...%f ]
        cout << "Output: [ ";
        for(double& value : results){
            cout << fixed << value << ' ';
        }
        cout << "] Expected: [ ";
        for(double& value : set.second){
            cout << fixed << value << ' ';
        }
        cout << ']' << endl;
    }
}

// Private

void NeuralNetwork::goThoughtLayers(size_t start, size_t end, const function<void(size_t, size_t)>& action) {
    int direction = 0;
    if(start < end) direction = 1;
    else direction = -1;

    for(size_t layer = start; (start < end) ? layer < end : layer > end; layer += direction){
        for(size_t neuron = 0; neuron < layers[layer].neurons.size(); neuron++){
            action(layer, neuron);
        }
    }
}

double NeuralNetwork::computeMSE(const vector<double>& errors) {
    double errorAverage = 0.0;
    for(double err : errors){
        errorAverage += err;
    }
    return errorAverage / errors.size();
}

void NeuralNetwork::setTrainFile(const string &path) {
    File file(path);
    file.getTrainData(trainingData);
}
