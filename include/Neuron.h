//
// Created by ssl0 on 4/2/20.
//

#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace std;

class Neuron {
public:
    Neuron();

    void createRandWeights(int num);

    void createWeights(vector<double> weightsArr);

    static double sigmoidFunc(double x);

    vector<double> weights;

    double result = 0.0;

    double error = 0.0;

private:

    double getRandVal();
    default_random_engine eng;
    uniform_real_distribution<double> randGenerator;

};