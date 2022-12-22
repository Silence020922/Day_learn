# include "Matrix.h"
# include <iostream>
using namespace std;

    // Matrix(int size = 2);
    // ~Matrix();
    // void setMatrix(const double* values);
    // int getSize() const {return size;}
    // void printMatrix() const;
    // double &element(int i,int j){return elements[i*size+j];}
    // double element(int i,int j ) const {return elements[i*size +j];}

void Matrix::setMatrix(const double* values){
    for (int i=0;i<size * size;i++){
        elements[i] =   values[i];
    }
}
Matrix::Matrix(int size /* = 2*/):size(size){
    elements = new double [size*size];  //动态内存分配
}
Matrix::~Matrix(){
    delete[] elements;
}
void Matrix::printMatrix() const {
    cout<<"the Matrix is"<<endl;
    for (int i =0;i<size;i++){
        for (int j=0;j<size;j++){
            cout<<elements[i,j]<<" ";
        }
        cout<<endl;
    }

}