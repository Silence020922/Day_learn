#ifndef _MATRIX_H
#define _MATRIX_H
//定义matrix类
class Matrix{
    public:
    Matrix(int size = 2);
    ~Matrix();
    void setMatrix(const double* values);
    int getSize() const {return size;}
    void printMatrix() const;
    double &element(int i,int j){return elements[i*size+j];}
    double element(int i,int j ) const {return elements[i*size +j];}
    private:
    int size;
    double* elements;
};
#endif