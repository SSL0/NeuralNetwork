#include <iostream>

using namespace std;

#include <NeuralNetwork.h>

int main() {
    NeuralNetwork nw({784, 256, 256, 10});

    nw.setTrainFile("../Assets/dataset_digits.csv");

    nw.trainBP(1000, 0.001, 100, true);
    return 0;
}