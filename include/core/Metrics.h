#ifndef METRICS_H
#define METRICS_H

#include <vector>
using namespace std;
 
namespace Metrics {
    double meanSquaredError(const vector<double>& y_true, const vector<double>& y_pred);
    double accuracy(const  vector<int>& y_true, const vector<int>& y_pred);
}

#endif