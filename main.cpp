#include <iostream>

using namespace std;

#include <NeuralNetwork.h>

int main() {
    NeuralNetwork nw({3, 2, 1});
    vector<TrainingType> train = {
            TrainingType({0, 0, 0}, 0),
            TrainingType({0, 0, 1}, 0),
            TrainingType({0, 1, 0}, 0),
            TrainingType({0, 1, 1}, 0),
            TrainingType({1, 0, 0}, 0),
            TrainingType({1, 0, 1}, 0),
            TrainingType({1, 1, 0}, 0),
            TrainingType({1, 1, 1}, 1),

    };
    nw.TrainBPE(train, 5000, 12);
    cout << nw.Predict({1, 1, 1}) << endl;
    return 0;
}