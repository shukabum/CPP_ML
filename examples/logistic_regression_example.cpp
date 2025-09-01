#include "models/Logistic_regression.h"
#include <iostream>
#include <vector>
#include <random>
using namespace std;

// Generate synthetic dataset
void generateDataset(vector<vector<double>>& X, vector<int>& y, int n_samples, int n_features, int n_classes) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist(0, 1);

    for (int i = 0; i < n_samples; i++) {
        vector<double> features(n_features);
        for (int j = 0; j < n_features; j++) {
            features[j] = dist(gen) + (i % n_classes) * 2; // separate classes
        }
        X.push_back(features);
        y.push_back(i % n_classes);
    }
}

// Print first n samples
void printDataset(const vector<vector<double>>& X, const vector<int>& y, int n) {
    for (int i = 0; i < n && i < (int)X.size(); i++) {
        cout << "x: [ ";
        for (double val : X[i]) cout << val << " ";
        cout << "] y: " << y[i] << "\n";
    }
}

int main() {
    int n_features = 4;
    int n_classes = 3;

    vector<vector<double>> X_train, X_test;
    vector<int> y_train, y_test;

    // Generate datasets
    generateDataset(X_train, y_train, 10000, n_features, n_classes);
    generateDataset(X_test, y_test,500, n_features, n_classes);

    // Print samples
    cout << "First 10 Training samples:\n";
    printDataset(X_train, y_train, 10);

    cout << "\nFirst 10 Test samples:\n";
    printDataset(X_test, y_test, 10);

    // Train model
    LogisticRegression model(n_features, n_classes, 0.1, 0.01, 500, 16);
    model.fit(X_train, y_train);

    // Evaluate
    double acc = model.score(X_test, y_test);
    cout << "\nTest Accuracy: " << acc * 100 << "%\n";

    return 0;
}
