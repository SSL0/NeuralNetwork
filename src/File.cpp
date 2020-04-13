//
// Created by ssl0 on 4/5/20.
//

#include "File.h"

// Public

File::File(const string& path) {
    this->path = path;
}

void File::getInputData(vector<InputType>& trainData, int numOfOutputs) {
    inputStream.open(path);
    if(!inputStream.is_open()) return;
    string expectValue, input;
    while(!inputStream.eof()){
        getline(inputStream, expectValue, ',');
        getline(inputStream, input);

        vector<double> expecting;
        for(size_t i = 0; i < numOfOutputs; i++){
            if(!expectValue.empty() && i == stod(expectValue)){
                expecting.emplace_back(1);
            }else{
                expecting.emplace_back(0);
            }
        }

        trainData.emplace_back(getValuesFromStr(input, ','), expecting);
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
        output.emplace_back(stod(value) / 255);

    }
    return output;
}
