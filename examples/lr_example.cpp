#include <iostream>
#include <vector>
#include "models/LR.h"
#include "core/Matrix.h"
#include "core/Metrics.h"
using namespace std;


int main() {
    // Simple synthetic: y = 2x + 1
    vector<vector<double>> xraw = {{1},{2},{3},{4},{5},{6}};
    vector<double> y = {3,5,7,9,11,13};

    Matrix X(xraw);

    LinearRegression lr(0.01, 2000, 0.0); // lr, epochs, L2
    lr.fit(X, y);

    auto preds = lr.predict(X);

    cout << "Weights: ";
    for (auto w : lr.weights()) cout << w << " ";
    cout << "\nBias: " << lr.bias() << "\n";

    // MSE (using your Metrics core)
    double mse = 0.0;
    for (size_t i = 0; i < y.size(); ++i) {
        double e = preds[i] - y[i];
        mse += e * e;
    }
    mse /= y.size();
    cout << "MSE: " << mse << "\n";

    // Show predictions
    for (size_t i = 0; i < xraw.size(); ++i) {
        cout << "x=" << xraw[i][0] << " -> pred=" << preds[i] << " true=" << y[i] << "\n";
    }
    return 0;
}
