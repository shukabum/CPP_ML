#include "core/Dataset.h"
#include "core/Utils.h"
#include <iostream>
using namespace std;

Dataset::Dataset(const Matrix& features, const vector<double>& labels)
    : X(features), y(labels) {}

Dataset Dataset::fromCSV(const string& filename, bool hasHeader) {
    auto raw = Utils::readCSV(filename);

    if (raw.empty()) {
        cerr << "Error: dataset is empty\n";
        return Dataset(Matrix(0, 0), {});
    }

    if (hasHeader) {
        raw.erase(raw.begin());
    }

    size_t n_samples = raw.size();
    size_t n_features = raw[0].size() - 1;

    vector<vector<double>> features(n_samples, vector<double>(n_features));
    vector<double> labels(n_samples);

    for (size_t i = 0; i < n_samples; i++) {
        for (size_t j = 0; j < n_features; j++) {
            features[i][j] = raw[i][j];
        }
        labels[i] = raw[i][n_features];
    }

    return Dataset(Matrix(features), labels);
}

Matrix Dataset::getFeatures() const { return X; }
vector<double> Dataset::getLabels() const { return y; }
size_t Dataset::size() const { return y.size(); }
