//
// Created by ssl0 on 4/2/20.
//

#pragma once

#include <iostream>
#include <vector>
#include <functional>

// For print double value in cout
#include <iomanip>

#include <Layer.h>
#include <File.h>
using namespace std;

// Input data, Expected
typedef pair<vector<double>, vector<double>> InputType;

class NeuralNetwork {
public:
    /**
     * @param numOfNeuronOnEachLayer Vector that contain number of neuron on each layer
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
    void trainBP(int numOfEpoch, double learningRate, int batch, bool withBias = false);

    vector<double> setPredictFile(const string& path);
    /**
     * @brief Get train data from file
     * @param path Set path for train file
     */
    void setTrainFile(const string& path);

private:
    /**
     * @brief Get output in value from array
     * @param array
     */
    static double getAnswer(vector<double>& array);
    double computeMSE(vector<double>& errors);

    /**
     * @brief Go for layers and all include neurons
     * @param start Layer From
     * @param end Layer To
     * @param action Action for every neuron
     */
    void goThoughtLayers(size_t start, size_t end, const function<void(size_t, size_t)>& action);

    static double getRandVal(double min = 0.0, double max = 1.0);

    bool _withBias;

    vector<Layer> layers;
    vector<double> results;

    vector<InputType> trainingData;
    size_t errorsCount = 0;
    double errorsAverage = 0.0;
};
