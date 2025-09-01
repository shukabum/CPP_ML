#include "models/KNN.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <functional>
using namespace std;

AutoKNN::AutoKNN(int k, bool weighted)
    : k_(k), weighted_(weighted), root_(-1), useKDTree_(true) {}

void AutoKNN::fit(const vector<vector<double>>& X, const vector<int>& y) {
    if (X.empty()) throw invalid_argument("X is empty");
    if (X.size() != y.size()) throw invalid_argument("X and y size mismatch");

    dims_ = static_cast<int>(X[0].size());
    for (const auto& row : X) {
        if ((int)row.size() != dims_)
            throw invalid_argument("Inconsistent feature dimensions");
    }

    X_ = X;
    y_ = y;
    nodes_.clear();
    root_ = -1;

    // 🚀 Auto-select mode
    if (X_.size() > 40 && dims_ <= 20) {
        useKDTree_ = true;  // large dataset, low dimensions
    } else {
        useKDTree_ = false; // brute force fallback
        return;
    }

    // Build KD-Tree
    vector<int> idxs(X_.size());
    for (size_t i = 0; i < idxs.size(); ++i) idxs[i] = (int)i;

    root_ = build(idxs, 0);
}

int AutoKNN::build(vector<int>& idxs, int depth) {
    if (idxs.empty()) return -1;
    int splitDim = depth % dims_;
    size_t mid = idxs.size() / 2;

    nth_element(idxs.begin(), idxs.begin() + mid, idxs.end(),
        [&](int a, int b) { return X_[a][splitDim] < X_[b][splitDim]; });

    int pointIndex = idxs[mid];
    Node node {pointIndex, splitDim, X_[pointIndex][splitDim], -1, -1};
    int nodeId = nodes_.size();
    nodes_.push_back(node);

    if (mid > 0) {
        vector<int> L(idxs.begin(), idxs.begin() + mid);
        nodes_[nodeId].left = build(L, depth + 1);
    }
    if (mid + 1 < idxs.size()) {
        vector<int> R(idxs.begin() + mid + 1, idxs.end());
        nodes_[nodeId].right = build(R, depth + 1);
    }
    return nodeId;
}

double AutoKNN::dist2(const vector<double>& a, const vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

void AutoKNN::knnQuery(int nodeId,
                       const vector<double>& q,
                       priority_queue<pair<double,int>>& heap,
                       int kEff) const {
    if (nodeId == -1) return;
    const Node& node = nodes_[nodeId];

    int idx = node.pointIndex;
    double d2 = dist2(q, X_[idx]);
    if ((int)heap.size() < kEff) heap.emplace(d2, y_[idx]);
    else if (d2 < heap.top().first) { heap.pop(); heap.emplace(d2, y_[idx]); }

    double diff = q[node.splitDim] - node.splitVal;
    int nearChild = (diff <= 0.0) ? node.left : node.right;
    int farChild  = (diff <= 0.0) ? node.right : node.left;

    if (nearChild != -1) knnQuery(nearChild, q, heap, kEff);
    if ((int)heap.size() < kEff || (diff * diff) < heap.top().first) {
        if (farChild != -1) knnQuery(farChild, q, heap, kEff);
    }
}

int AutoKNN::argmax_by_value(const unordered_map<int, double>& wmap) {
    int best = -1;
    double bestv = -numeric_limits<double>::infinity();
    for (auto& kv : wmap) {
        if (kv.second > bestv || (kv.second == bestv && kv.first < best)) {
            best = kv.first;
            bestv = kv.second;
        }
    }
    return best;
}

int AutoKNN::bruteForcePredict(const vector<double>& x) const {
    int kEff = min(k_, (int)X_.size());
    priority_queue<pair<double,int>> heap;
    for (size_t i = 0; i < X_.size(); i++) {
        double d2 = dist2(x, X_[i]);
        if ((int)heap.size() < kEff) heap.emplace(d2, y_[i]);
        else if (d2 < heap.top().first) { heap.pop(); heap.emplace(d2, y_[i]); }
    }

    unordered_map<int,double> votes;
    constexpr double eps = 1e-12;
    while (!heap.empty()) {
        auto [d2,lbl] = heap.top(); heap.pop();
        double w = weighted_ ? 1.0 / (sqrt(d2) + eps) : 1.0;
        votes[lbl] += w;
    }
    return argmax_by_value(votes);
}

int AutoKNN::predict(const vector<double>& x) const {
    if (x.size() != (size_t)dims_) throw invalid_argument("Feature dimension mismatch");

    if (!useKDTree_) return bruteForcePredict(x);

    int kEff = min(k_, (int)X_.size());
    priority_queue<pair<double,int>> heap;
    knnQuery(root_, x, heap, kEff);

    unordered_map<int,double> votes;
    constexpr double eps = 1e-12;
    while (!heap.empty()) {
        auto [d2,lbl] = heap.top(); heap.pop();
        double w = weighted_ ? 1.0 / (sqrt(d2) + eps) : 1.0;
        votes[lbl] += w;
    }
    return argmax_by_value(votes);
}

vector<int> AutoKNN::predictBatch(const vector<vector<double>>& Xq) const {
    vector<int> out;
    out.reserve(Xq.size());
    for (const auto& q : Xq) out.push_back(predict(q));
    return out;
}

double AutoKNN::score(const vector<vector<double>>& Xq, const vector<int>& yq) const {
    auto preds = predictBatch(Xq);
    int correct = 0;
    for (size_t i = 0; i < preds.size(); ++i) if (preds[i] == yq[i]) ++correct;
    return preds.empty() ? 0.0 : (double)correct / preds.size();
}
