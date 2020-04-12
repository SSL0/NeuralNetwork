#include <iostream>

using namespace std;

#include <NeuralNetwork.h>

int main() {
    NeuralNetwork nw({784, 512, 128, 32, 10});

    nw.setTrainFile("../Assets/dataset_digits.csv");

    nw.trainBP(385, 0.001, 100, true);

    return 0;
}