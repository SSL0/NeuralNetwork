//
// Created by ssl0 on 4/2/20.
//

#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <fstream>
#include <sstream>

#include <Layer.h>

using namespace std;

// IData, Expected
typedef pair<vector<double>, vector<double>> TrainType;

class NeuralNetwork {
public:
    /**
     * @param numOfNeuronOnEachLayer It's vector that contain number of neuron on each layer
     */
    NeuralNetwork(const vector<int>& neuronsForEach);

    /**
     * @brief Show result of NN
     * @param input data
     * @return Neurons on last layer
     */
    vector<Neuron>& Predict(const vector<double>& input);

    /**
     * @brief Train with "Backpropagation" method
     * @param trainingData Pair<I data, Expect result>
     * @param numOfEpoch Number of epoch
     * @param learningRate Num between 0...1 that show how fast will learning be
     */
    void TrainBP(int numOfEpoch, double learningRate);


    void GetTrainFile(const string& path);

private:
    vector<Layer> layers;
    vector<TrainType> trainingData;
    static double ComputeMSE(const vector<double>& errors);
    void GoThoughtLayers(size_t start, size_t end, const function<void(size_t, size_t)>& action);
    static vector<double> GetValuesFromStr(const string& str, const char& delim);
};
