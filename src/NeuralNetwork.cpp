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
    double expSum = 0.0;
    // Calculate other LAYERS
    goThoughtLayers(1, layers.size(), [this, &expSum](size_t currentLayer, size_t currentNeuron) {
        double result = 0;
        for (Neuron &prevLayerNeuron : layers[currentLayer - 1].neurons) {
            result += prevLayerNeuron.weights[currentNeuron] * prevLayerNeuron.result;
        }
        if(_withBias){
            result += layers[currentLayer - 1].biasNeuron.weights[currentLayer];
        }
        // For softmax activation func
        if(currentLayer == layers.size() - 1 && layers.back().neurons.size() > 2){
            layers[currentLayer].neurons[currentNeuron].result = exp(result);
            expSum += layers[currentLayer].neurons[currentNeuron].result;
        }else{
            layers[currentLayer].neurons[currentNeuron].result = Neuron::sigmoidFunc(result);
        }
    });

    // Softmax func
    if(layers.back().neurons.size() > 2){
        for(Neuron& neuron : layers.back().neurons){
            neuron.result /= expSum;
        }
    }

    results.clear();

    for(Neuron& neuron : layers.back().neurons){
        results.emplace_back(neuron.result);
    }

    return results;
}

void NeuralNetwork::trainBP(int numOfEpoch, double learningRate, int batch, bool withBias) {
    _withBias = withBias;

    File out("errors.txt");

    for(int epoch = 1; epoch < numOfEpoch; epoch++) {
        int correctAnswers = 0;
        for (int setCount = 0, setIndex; setCount < batch; setCount++) {
            setIndex = (int) (getRandVal() * trainingData.size());

            predict(trainingData[setIndex].first);

            // Make error for neurons on LAST LAYER
            for (size_t i = 0; i < layers.back().neurons.size(); i++) {
                layers.back().neurons[i].error = trainingData[setIndex].second[i] - layers.back().neurons[i].result;
            }

            // Back Propagation Error
            goThoughtLayers(layers.size() - 1, 0, [this, &learningRate](size_t currentLayer, size_t currentNeuron) {
                double error = layers[currentLayer].neurons[currentNeuron].error;
                double actual = layers[currentLayer].neurons[currentNeuron].result;

                double gradient = (actual * (1 - actual)) * error * learningRate;

                for (Neuron &prevLayerNeuron : layers[currentLayer - 1].neurons) {
                    double deltaWeights = gradient * prevLayerNeuron.result;
                    prevLayerNeuron.error += prevLayerNeuron.weights[currentNeuron] * error;
                    prevLayerNeuron.weights[currentNeuron] += deltaWeights;

                    if (_withBias) {
                        layers[currentLayer - 1].biasNeuron.weights[currentNeuron] += learningRate * gradient;
                    }
                }
                if(currentLayer < layers.size() - 1){
                    layers[currentLayer].neurons[currentNeuron].error = 0.0;
                }
            });

            if(getAnswer(trainingData[setIndex].second) == getAnswer(results)){
                correctAnswers++;
            }
        }
        vector<double> errors;
        for (Neuron &neuron : layers.back().neurons) {
            errors.emplace_back(neuron.error * neuron.error);
        }
        double MSE = computeMSE(errors);
        out.write(to_string(MSE));
        printf("\rTrain progress: [%d%%] | Epoch: %d | MSE: %f | Correct: %d", epoch * 100 / numOfEpoch, epoch, MSE, correctAnswers);
    }

    cout << endl << "Results after train: " << endl;
    for(auto & set : trainingData){
        predict(set.first);

        // Print string type:
        // Expected: [ %f...%f ] Output: [ %f...%f ]

        cout << "Output: [ ";
        cout << getAnswer(results);
        cout << "] Expected: [ ";
        cout << getAnswer(set.second);
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

double NeuralNetwork::getAnswer(vector<double>& array){
    double maximum = 0.0, answer = 0.0;
    for (size_t i = 0; i < array.size(); i++) {
        if(maximum < array[i]){
            maximum = array[i];
            answer = i;
        }
    }
    return answer;
}

double NeuralNetwork::computeMSE(vector<double>& errors) {
    errorsCount += errors.size();
    for(double err : errors){
        errorsAverage += err;
    }
    return errorsAverage / errorsCount;
}

void NeuralNetwork::setTrainFile(const string &path) {
    File file(path);
    file.getInputData(trainingData, layers.back().neurons.size());
}

vector<double> NeuralNetwork::setPredictFile(const string &path) {
    File file(path);
    vector<InputType> predicting;
    file.getInputData(predicting, layers.back().neurons.size());
    for(size_t i = 0; i < predicting.size(); i++){
        predict(predicting[i].first);

        cout << "Output: [ ";
        cout << getAnswer(results);
        cout << "] Expected: [ ";
        cout << getAnswer(predicting[i].second);
        cout << ']' << endl;
    }
}

double NeuralNetwork::getRandVal(double min, double max) {
    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<double> randGenerator(min, max);

    return randGenerator(mt);
}
