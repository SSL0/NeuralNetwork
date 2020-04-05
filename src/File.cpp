//
// Created by ssl0 on 4/5/20.
//

#include "File.h"

// Public

File::File(const string& path) {
    fileStream.open(path);
}

void File::getTrainData(vector<TrainType>& trainData) {
    string input, expected;
    while(!fileStream.eof()){
        getline(fileStream, input, '|');
        getline(fileStream, expected);

        trainData.emplace_back(getValuesFromStr(input, ','), getValuesFromStr(expected, ','));
    }
}

void File::write(const string &str) {
    fileStream << str << endl;
}

// Private

vector<double> File::getValuesFromStr(const string &str, const char &delim)  {
    vector<double> output;
    stringstream ss(str);
    for(string value; getline(ss, value, delim);){
        output.emplace_back(stod(value));
    }
    return output;
}
