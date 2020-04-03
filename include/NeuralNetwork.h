//
// Created by ssl0 on 4/2/20.
//

#pragma once

#include <iostream>
#include <vector>

#include <Layer.h>

using namespace std;

// InputData, Expected
typedef pair<vector<double>, double> TrainingType;

class NeuralNetwork {
public:
    /**
     * @param numOfNeuronOnEachLayer It's vector that contain number of neuron on each layer
     */
    NeuralNetwork(vector<int> neuronsForEach);

    /**
     * @brief Show result of NN
     * @param input Input data
     * @return Neurons on last layer
     */
    double Predict(vector<double> input);

    /**
     * @brief Train with "Back Propagation Error" method
     * @param trainingData Pair<Input data, Expect result>
     * @param numOfEpoch Number of epoch
     * @param learningRate Num between 0...1 that show how fast will learning be
     */
    void TrainBPE(vector<TrainingType> trainingData, int numOfEpoch, double learningRate);

private:
    vector<Layer> layers;
};
