#include "models/LR.h"
#include <stdexcept>
#include <cmath>

using namespace std;


LinearRegression::LinearRegression(double learningRate, size_t epochs, double l2)
    : lr_(learningRate), epochs_(epochs), l2_(l2) {}

void LinearRegression::fit(const Matrix& X, const vector<double>& y) {
    const size_t n = X.rowCount();
    const size_t d = X.colCount();
    if (y.size() != n) throw invalid_argument("y.size() must equal X.rowCount()");

    // init params
    w_.assign(d, 0.0);
    b_ = 0.0;

    for (size_t epoch = 0; epoch < epochs_; ++epoch) {
        // compute predictions
        vector<double> yhat(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            double s = b_;
            for (size_t j = 0; j < d; ++j) s += w_[j] * X(i, j);
            yhat[i] = s;
        }

        // gradients
        vector<double> gradW(d, 0.0);
        double gradB = 0.0;

        for (size_t i = 0; i < n; ++i) {
            const double err = (yhat[i] - y[i]);   // d/dyhat MSE = 2*(yhat - y) / (2) -> average below
            gradB += err;
            for (size_t j = 0; j < d; ++j) {
                gradW[j] += err * X(i, j);
            }
        }

        // average and add L2
        const double invN = 1.0 / static_cast<double>(n);
        for (size_t j = 0; j < d; ++j) {
            gradW[j] = gradW[j] * invN + l2_ * w_[j];
        }
        gradB *= invN; // no L2 on bias

        // update
        for (size_t j = 0; j < d; ++j) w_[j] -= lr_ * gradW[j];
        b_ -= lr_ * gradB;
    }
}

vector<double> LinearRegression::predict(const Matrix& X) const {
    const size_t n = X.rowCount();
    const size_t d = X.colCount();
    if (w_.size() != d) throw invalid_argument("Model not fitted or feature mismatch");

    vector<double> out(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double s = b_;
        for (size_t j = 0; j < d; ++j) s += w_[j] * X(i, j);
        out[i] = s;
    }
    return out;
}

Matrix LinearRegression::predictMatrix(const Matrix& X) const {
    auto v = predict(X);
    Matrix m(v.size(), 1, 0.0);
    for (size_t i = 0; i < v.size(); ++i) m(i, 0) = v[i];
    return m;
}
