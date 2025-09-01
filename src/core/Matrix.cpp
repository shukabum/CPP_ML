#include "core/Matrix.h"
using namespace std;


Matrix::Matrix(size_t r,size_t c, double val): rows(r),cols(c){
    data.assign(r,vector<double>(c, val));
}

Matrix::Matrix(const vector<vector<double>> & d){
    rows = d.size();
    cols = d[0].size();
    data = d;
}


size_t  Matrix::rowCount() const{ return rows;}
size_t  Matrix::colCount() const{ return cols;}

double& Matrix::operator()(size_t r, size_t c){ return data[r][c];}
const double & Matrix::operator()(size_t r, size_t c)const {return data[r][c];}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for(size_t i=0;i<rows;i++){
        for(size_t j =0;j<cols;j++){
            result(j,i) = data[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator+(const Matrix&  other)const{
    if(rows!=other.rows||cols!= other.cols)
        throw invalid_argument("Matrix dimensions must match for addition");
    Matrix result(rows,cols);
    for(size_t i=0;i<rows;i++){
        for(size_t j=0;j<cols;j++){
            result(i,j)= data[i][j] + other(i,j);
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix&  other)const{
    if(rows!=other.rows||cols!= other.cols)
        throw invalid_argument("Matrix dimensions must match for subtraction");
    Matrix result(rows,cols);
    for(size_t i=0;i<rows;i++){
        for(size_t j=0;j<cols;j++){
            result(i,j)= data[i][j] - other(i,j);
        }
    }
    return result;
}
Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows)
        throw invalid_argument("Invalid dimensions for matrix multiplication");
    Matrix result(rows, other.cols, 0.0);
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < other.cols; j++)
            for (size_t k = 0; k < cols; k++)
                result(i, j) += data[i][k] * other(k, j);
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            result(i, j) = data[i][j] * scalar;
    return result;
}
Matrix Matrix::operator/(double scalar) const {
    if (scalar == 0)
        throw invalid_argument("Division by zero");
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            result(i, j) = data[i][j] / scalar;
    return result;
}

vector<double> Matrix::dot(const vector<double>& vec) const {
    if (cols != vec.size())
        throw invalid_argument("Matrix-vector dimension mismatch");
    std::vector<double> result(rows, 0.0);
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            result[i] += data[i][j] * vec[j];
    return result;
}

void Matrix::print() const {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++)
            std::cout << data[i][j] << " ";
        std::cout << "\n";
    }
}