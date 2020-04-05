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
#include <File.h>
using namespace std;

// Input data, Expected
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
    vector<double>& predict(const vector<double>& input);

    /**
     * @brief Train with "Backpropagation" method
     * @param trainingData Pair<I data, Expect result>
     * @param numOfEpoch Number of epoch
     * @param learningRate Num between 0...1 that show how fast will learning be
     */
    void trainBP(int numOfEpoch, double learningRate);


    void setTrainFile(const string& path);

private:
    vector<Layer> layers;
    vector<double> results;

    vector<TrainType> trainingData;
    static double computeMSE(const vector<double>& errors);
    void goThoughtLayers(size_t start, size_t end, const function<void(size_t, size_t)>& action);
};
