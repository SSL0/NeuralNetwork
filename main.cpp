#include <iostream>

using namespace std;

#include <NeuralNetwork.h>

int main() {
    NeuralNetwork nw({4, 25, 3});

    nw.GetTrainFile("../Assets/dataset.csv");

    nw.TrainBP(2000, 0.2);

    return 0;
}