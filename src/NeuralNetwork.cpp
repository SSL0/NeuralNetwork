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
        //result += layers[currentLayer - 1].biasNeuron.weights[currentLayer];
        layers[currentLayer].neurons[currentNeuron].result = Neuron::sigmoidFunc(result);
    });
    return layers.back().neurons;
}

void NeuralNetwork::TrainBP(int numOfEpoch, double learningRate) {
    // Array for MSE
    vector<double> errors;
    for(int epoch = 0; epoch <= numOfEpoch; epoch++) {
        for(const TrainType& set : trainingData){
            // Make error for last neuron
            vector<Neuron>& neurons = Predict(set.first);
            for(size_t i = 0; i < neurons.size(); i++){
                layers.back().neurons[i].error = layers.back().neurons[i].result - set.second[i];
            }

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
        printf("\rTrain progress: [%d%%] | MSE: %f", epoch * 100 / numOfEpoch, ComputeMSE(errors));
    }

    cout << endl;
    for(auto & set : trainingData){
        vector<Neuron> predict = Predict(set.first);
        string first, second;
        if(predict[0].result > .8) first = "setosa";
        else if(predict[1].result > .8) first = "versicolor";
        else if(predict[2].result > .8) first = "virginica";

        if(set.second[0] > .8) second = "setosa";
        else if(set.second[1] > .8) second = "versicolor";
        else if(set.second[2] > .8) second = "virginica";

        printf("Expected: [%s] \n Output: [%s] ", first.c_str(), second.c_str());
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

void NeuralNetwork::GetTrainFile(const string& path) {
    ifstream trainFile(path);
    string input, expected;
    while(!trainFile.eof()){
        getline(trainFile, input, '|');
        getline(trainFile, expected);

        trainingData.emplace_back(GetValuesFromStr(input, ','), GetValuesFromStr(expected, ','));
    }
}

vector<double> NeuralNetwork::GetValuesFromStr(const string& str, const char &delim) {
    vector<double> output;
    stringstream ss(str);
    for(string value; getline(ss, value, delim);){
        output.emplace_back(stod(value));
    }
    return output;
}
