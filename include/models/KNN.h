#ifndef KNN_H
#define KNN_H

#include <vector>
#include <queue>
#include <unordered_map>
#include <limits>

using namespace std;

class AutoKNN {
private:
    struct Node {
        int pointIndex;
        int splitDim;
        double splitVal;
        int left, right;
    };

    // Params
    int k_;
    bool weighted_;
    int dims_;
    bool useKDTree_;   // <-- NEW: auto-selection flag

    // Data
    vector<vector<double>> X_;
    vector<int> y_;

    // KD-Tree
    vector<Node> nodes_;
    int root_;

    // Build KD-Tree
    int build(vector<int>& idxs, int depth);

    // Utils
    static double dist2(const vector<double>& a, const vector<double>& b);
    static int argmax_by_value(const unordered_map<int, double>& wmap);

    void knnQuery(int nodeId,
                  const vector<double>& q,
                  priority_queue<pair<double,int>>& heap,
                  int kEff) const;

    // Brute-force fallback
    int bruteForcePredict(const vector<double>& x) const;

public:
    AutoKNN(int k = 3, bool weighted = false);

    void fit(const vector<vector<double>>& X, const vector<int>& y);
    int predict(const vector<double>& x) const;
    vector<int> predictBatch(const vector<vector<double>>& Xq) const;
    double score(const vector<vector<double>>& Xq, const vector<int>& yq) const;
};

#endif
