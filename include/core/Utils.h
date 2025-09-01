#ifndef UTILS_H
#define UTILS_H

#include<vector>
#include<string>
using namespace std;
namespace Utils{
    vector<vector<double>> readCSV(const string& filename);
    double randomDouble(double min, double max);
}


#endif