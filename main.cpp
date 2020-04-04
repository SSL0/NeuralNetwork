#include <iostream>

using namespace std;

#include <NeuralNetwork.h>

int main() {
    NeuralNetwork nw({3, 2, 1});
    vector<TrainType> train = {
            TrainType({0, 0, 0}, 0),
            TrainType({0, 0, 1}, 1),
            TrainType({0, 1, 0}, 0),
            TrainType({0, 1, 1}, 0),
            TrainType({1, 0, 0}, 1),
            TrainType({1, 0, 1}, 1),
            TrainType({1, 1, 0}, 0),
            TrainType({1, 1, 1}, 1),
    };
    nw.TrainBPE(train, 5000, 0.2);
    cout << endl;
    for(auto & trainingSet : train){
        cout << "Input: [" << trainingSet.first[0] << ", " << trainingSet.first[1] << ", " << trainingSet.first[2] << "] " << "Output: " << (nw.Predict(trainingSet.first).back().result > 0.8) << " Expected: " << trainingSet.second << endl;
    }
    return 0;
}