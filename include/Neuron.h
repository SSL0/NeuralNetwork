//
// Created by ssl0 on 4/2/20.
//

#pragma once

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class Neuron {
public:
    Neuron();

    void CreateRandWeights(int num);

    void CreateWeights(vector<double> weightsArr);

    static double SigmoidFunc(double x);

    vector<double> weights;

    double result = 0;

    double error;

private:
};