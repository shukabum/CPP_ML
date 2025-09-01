#include "core/Utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>

using namespace std;

vector<vector<double>> Utils:: readCSV(const string & filename){
    vector<vector<double>> data;
    ifstream file(filename);
    if(!file.is_open()){
        cerr<<"Error: cannot open the file: "<<filename<<endl;
        return data;
    }

    string line;
    while(getline(file,line)){
        stringstream ss(line);
        string value;
        vector<double> row;

        while(getline(ss, value,',')){
            row.push_back(stod(value));
        }
        data.push_back(row);
    }
    file.close();
    return data;
}

double Utils::randomDouble(double min, double max){
    static random_device rd;
    static mt19937 gen(rd());
    uniform_real_distribution<> dis(min,max);
    return dis(gen);

}