#include <iostream>

using namespace std;

#include <NeuralNetwork.h>

int main() {
    NeuralNetwork nw({3, 2, 1});
    vector<TrainingType> train = {
            TrainingType({0, 0, 0}, 0),
            TrainingType({0, 0, 1}, 1),
            TrainingType({0, 1, 0}, 0),
            TrainingType({0, 1, 1}, 0),
            TrainingType({1, 0, 0}, 1),
            TrainingType({1, 0, 1}, 1),
            TrainingType({1, 1, 0}, 0),
            TrainingType({1, 1, 1}, 0),

    };
    nw.TrainBPE(train, 5000, 0.2);
    for(auto & trainingSet : train){
        cout << "Input: [" << trainingSet.first[0] << ", " << trainingSet.first[1] << ", " << trainingSet.first[2] << "] " << "Output: " << nw.Predict(trainingSet.first) << " Expected: " << trainingSet.second << endl;
    }
    return 0;
}