#include <iostream>

using namespace std;

#include <NeuralNetwork.h>

int main() {
    NeuralNetwork nw({4, 25, 3});

    nw.setTrainFile("../Assets/dataset_iris.csv");

    nw.trainBP(5000, 0.2, 75, true);

    return 0;
}