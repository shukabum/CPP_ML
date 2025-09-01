#ifndef MATRIX_H
#define MATRIX_H

#include<vector>
#include<iostream>
#include<stdexcept>
using namespace std;

class Matrix{
    private:
    vector<vector<double>>data;
    size_t rows, cols;

    public:
    Matrix(size_t r, size_t c, double val = 0.0);
    Matrix(const vector<vector<double>>& d);


    size_t rowCount() const;
    size_t colCount() const;

    double & operator()(size_t r, size_t c);
    const double&  operator()(size_t r,size_t c) const;

    Matrix transpose() const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scaler) const;
    Matrix operator/(double scaler) const;

    vector<double> dot(const vector<double>& vec) const;
    void print() const;

};

#endif