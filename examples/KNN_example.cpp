#include <iostream>
#include <vector>
#include <random>
#include "models/KNN.h"

using namespace std;

// Function to generate Gaussian cluster
static void gen_gaussian_cluster(vector<vector<double>>& X, vector<int>& y,
                                 int n, int label, double cx, double cy, double sigma = 0.8) {
    mt19937 rng(42 + label);
    normal_distribution<double> nx(cx, sigma), ny(cy, sigma);
    for (int i = 0; i < n; ++i) {
        X.push_back({nx(rng), ny(rng)});
        y.push_back(label);
    }
}

int main() {
    // Create 3 clusters => 3 classes
    vector<vector<double>> X_train, X_test;
    vector<int> y_train, y_test;

    gen_gaussian_cluster(X_train, y_train, 70, 0, 0.0, 0.0);
    gen_gaussian_cluster(X_train, y_train, 70, 1, 4.0, 4.0);
    gen_gaussian_cluster(X_train, y_train, 60, 2, -4.0, 4.0);

    gen_gaussian_cluster(X_test, y_test, 10, 0, 0.0, 0.0);
    gen_gaussian_cluster(X_test, y_test, 5,  1, 4.0, 4.0);
    gen_gaussian_cluster(X_test, y_test, 5,  2, -4.0, 4.0);

    cout << "Train size: " << X_train.size() << " samples\n";
    cout << "Test size:  " << X_test.size() << " samples\n\n";

    // Print first 10 training samples
    cout << "First 10 train:\n";
    for (int i = 0; i < 10 && i < (int)X_train.size(); ++i) {
        cout << "x=[" << X_train[i][0] << ", " << X_train[i][1] << "] y=" << y_train[i] << "\n";
    }

    // Create KNN model with auto backend (KD-tree or brute force chosen automatically)
    AutoKNN knn(/*k=*/5, /*weighted=*/true);
    knn.fit(X_train, y_train);

    // Compute accuracy
    double acc = knn.score(X_test, y_test);
    cout << "\nTest accuracy: " << acc * 100.0 << "%\n";

    // Show predictions for first 10 test points
    auto preds = knn.predictBatch(X_test);
    cout << "\nFirst 10 predictions:\n";
    for (int i = 0; i < 10 && i < (int)X_test.size(); ++i) {
        cout << "x=[" << X_test[i][0] << ", " << X_test[i][1]
             << "] y_true=" << y_test[i] << " y_pred=" << preds[i] << "\n";
    }

    return 0;
}
