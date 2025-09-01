#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <vector>
#include <string>
using namespace std;

class LogisticRegression {
private:
    int n_features;
    int n_classes;
    double learning_rate;
    double reg_lambda;
    int epochs;
    int batch_size;
    vector<vector<double>> weights; // shape: [n_classes][n_features]
    vector<double> bias;                 // shape: [n_classes]

    vector<double> softmax(const vector<double>& z) const;
    void initializeWeights();

public:
    LogisticRegression(int features, int classes, double lr = 0.1, double reg = 0.01, 
                       int ep = 1000, int batch = 32);

    void fit(const vector<vector<double>>& X, const vector<int>& y);
    vector<int> predict(const vector<vector<double>>& X) const;
    double score(const vector<vector<double>>& X, const vector<int>& y) const;
};

#endif
