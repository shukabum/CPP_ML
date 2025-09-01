#ifndef LR_H
#define LR_H

#include <vector>
#include <cstddef>
#include "../core/Matrix.h"
using namespace std;

class LinearRegression {
public:
    // L2 = ridge penalty (lambda). Set to 0.0 to disable.
    explicit LinearRegression(double learningRate = 0.01,
                              size_t epochs = 1000,
                              double l2 = 0.0);

    // Fit on X (n x d) and y (n)
    void fit(const Matrix& X, const vector<double>& y);

    // Predict continuous values (n)
    vector<double> predict(const Matrix& X) const;

    // Convenience: predict as an (n x 1) matrix
    Matrix predictMatrix(const Matrix& X) const;

    // Getters
    const vector<double>& weights() const { return w_; }
    double bias() const { return b_; }

private:
    vector<double> w_;  // d
    double b_{0.0};          // scalar bias
    double lr_{0.01};
    size_t epochs_{1000};
    double l2_{0.0};         // ridge regularization strength
};

#endif 
