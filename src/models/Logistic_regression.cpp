#include "models/Logistic_regression.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>  // ✅ for iota
#include <iostream>

using namespace std;

// ------------------------
// Constructor
// ------------------------
LogisticRegression::LogisticRegression(int features, int classes, double lr, double reg, 
                                       int ep, int batch)
    : n_features(features), n_classes(classes), learning_rate(lr), 
      reg_lambda(reg), epochs(ep), batch_size(batch) {
    initializeWeights();
}

// ------------------------
// Xavier Initialization
// ------------------------
void LogisticRegression::initializeWeights() {
    mt19937 gen(random_device{}());
    double limit = sqrt(6.0 / (n_features + n_classes));
    uniform_real_distribution<> dis(-limit, limit);

    weights.assign(n_classes, vector<double>(n_features));
    bias.assign(n_classes, 0.0);

    for (int c = 0; c < n_classes; ++c) {
        for (int f = 0; f < n_features; ++f) {
            weights[c][f] = dis(gen);
        }
    }
}

// ------------------------
// Numerically stable softmax
// ------------------------
vector<double> LogisticRegression::softmax(const vector<double>& z) const {
    double maxVal = *max_element(z.begin(), z.end());
    vector<double> expVals(z.size());

    for (size_t i = 0; i < z.size(); ++i)
        expVals[i] = exp(z[i] - maxVal);

    double sum = accumulate(expVals.begin(), expVals.end(), 0.0);
    for (auto& val : expVals) val /= sum;

    return expVals;
}

// ------------------------
// Training with mini-batch GD + L2 reg + decay
// ------------------------
void LogisticRegression::fit(const vector<vector<double>>& X, const vector<int>& y) {
    int n_samples = X.size();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle indices for mini-batch
        vector<int> indices(n_samples);
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), mt19937(random_device{}()));

        for (int start = 0; start < n_samples; start += batch_size) {
            int end = min(start + batch_size, n_samples);
            int bsize = end - start;

            // Gradient accumulators
            vector<vector<double>> gradW(n_classes, vector<double>(n_features, 0.0));
            vector<double> gradB(n_classes, 0.0);

            for (int idx = start; idx < end; ++idx) {
                int i = indices[idx];

                // Linear logits
                vector<double> logits(n_classes, 0.0);
                for (int c = 0; c < n_classes; ++c) {
                    logits[c] = bias[c];
                    for (int f = 0; f < n_features; ++f)
                        logits[c] += weights[c][f] * X[i][f];
                }

                vector<double> probs = softmax(logits);

                // Gradient update
                for (int c = 0; c < n_classes; ++c) {
                    double error = probs[c] - (y[i] == c ? 1.0 : 0.0);
                    for (int f = 0; f < n_features; ++f)
                        gradW[c][f] += error * X[i][f];
                    gradB[c] += error;
                }
            }

            // Update weights with regularization
            for (int c = 0; c < n_classes; ++c) {
                for (int f = 0; f < n_features; ++f) {
                    gradW[c][f] /= bsize;
                    gradW[c][f] += reg_lambda * weights[c][f]; // L2 penalty
                    weights[c][f] -= learning_rate * gradW[c][f];
                }
                gradB[c] /= bsize;
                bias[c] -= learning_rate * gradB[c];
            }
        }

        // Learning rate decay
        learning_rate *= 0.99;
    }
}

// ------------------------
// Prediction
// ------------------------
vector<int> LogisticRegression::predict(const vector<vector<double>>& X) const {
    vector<int> predictions;
    for (const auto& sample : X) {
        vector<double> logits(n_classes, 0.0);
        for (int c = 0; c < n_classes; ++c) {
            logits[c] = bias[c];
            for (int f = 0; f < n_features; ++f)
                logits[c] += weights[c][f] * sample[f];
        }
        vector<double> probs = softmax(logits);
        predictions.push_back(max_element(probs.begin(), probs.end()) - probs.begin());
    }
    return predictions;
}

// ------------------------
// Accuracy Score
// ------------------------
double LogisticRegression::score(const vector<vector<double>>& X, const vector<int>& y) const {
    vector<int> preds = predict(X);
    int correct = 0;
    for (size_t i = 0; i < y.size(); ++i)
        if (preds[i] == y[i]) ++correct;
    return static_cast<double>(correct) / y.size();
}
