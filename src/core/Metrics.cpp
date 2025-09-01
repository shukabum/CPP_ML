#include "core/Metrics.h"
#include <cmath>
using namespace std;


double Metrics::meanSquaredError(const vector<double>& y_true, const vector<double>& y_pred) {
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); i++) {
        sum += pow(y_true[i] - y_pred[i], 2);
    }
    return sum / y_true.size();
}

double Metrics::accuracy(const vector<int>& y_true, const vector<int>& y_pred) {
    int correct = 0;
    for (size_t i = 0; i < y_true.size(); i++) {
        if (y_true[i] == y_pred[i]) correct++;
    }
    return static_cast<double>(correct) / y_true.size();
}