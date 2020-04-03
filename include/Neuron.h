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

    double Result = 0;

    void CreateRandWeights(int num);

    void CreateWeights(vector<double> weightsArr);

    vector<double> weights;

    static double SigmoidFunc(double x);
private:
};