#ifndef DATASET_H
#define DATASET_H

#include "Matrix.h"
#include <vector>
#include <string>
using namespace std;


class Dataset {
private:
    Matrix X;
    vector<double> y;

public:
    Dataset(const Matrix& features, const vector<double>& labels);
    static Dataset fromCSV(const string& filename, bool hasHeader = false);

    Matrix getFeatures() const;
    vector<double> getLabels() const;
    size_t size() const;
};

#endif