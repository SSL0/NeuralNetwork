#include <iostream>

using namespace std;

#include <NeuralNetwork.h>

int main() {
    NeuralNetwork nw({3, 2, 1});

    nw.setTrainFile("../Assets/dataset1.csv");

    nw.trainBP(5000, 0.15);

    return 0;
}