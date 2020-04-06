//
// Created by ssl0 on 4/5/20.
//

#include "File.h"

// Public

File::File(const string& path) {
    this->path = path;
}

void File::getTrainData(vector<TrainType>& trainData) {
    inputStream.open(path);
    string input, expected;
    while(!inputStream.eof()){
        getline(inputStream, input, '|');
        getline(inputStream, expected);

        trainData.emplace_back(getValuesFromStr(input, ','), getValuesFromStr(expected, ','));
    }
    inputStream.close();

}

void File::write(const string &str) {
    if(!outputStream.is_open()) outputStream.open(path, ios::app);
    outputStream << str << endl;
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
