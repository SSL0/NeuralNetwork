//
// Created by ssl0 on 4/5/20.
//

#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

// Input data, Expected
typedef pair<vector<double>, vector<double>> InputType;

class File {
public:
    File(const string& path);

    void write(const string& str);

    void getInputData(vector<InputType>& trainData, int numOfOutputs);

private:
    string path;

    ifstream inputStream;
    ofstream outputStream;
    static vector<double> getValuesFromStr(const string& str, const char &delim);
};
