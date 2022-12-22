# include "LinerEqu.h"
# include <iostream>
# include <cmath>
using namespace std;

    // public:
    // LinerEqu(int size=2);
    // ~LinerEqu();
    // void setLinerEqu(const double* a,const double* b);
    // bool solve();
    // void printlinerEqu() const;
    // void printSolution() const;
    // private:
    // double* sums;
    // double* solution;

LinerEqu::LinerEqu(int size /*= 2*/):Matrix(size){
    sums  = new double [size];
    solution = new double [size];
}
LinerEqu::~LinerEqu(){
    delete[] sums;
    delete[] solution;
}
void LinerEqu::setLinerEqu(const double* a,const double* b){ //a为线性矩阵b为右端向量
    setMatrix(a);
    for (int i = 0;i<getSize();i++){
        sums[i] = b[i];
    }
}
void LinerEqu::printlinerEqu() const {
    cout<<"The line Equ is :"<<endl;
    for (int i = 0;i < getSize();i++){
        for (int j=0;j<getSize();j++)
        cout<<element(i,j)<<" ";
        cout<<" "<<sums[i]<<endl;
    }
}
inline void swap (double &v1,double &v2){
    double temp = v1;
    v1 = v2;
    v2 = temp;
} //交换两个数

bool LinerEqu::solve(){
    int *js = new int[getSize()]; //js存储主元素所在列号的数组
    
    //选择主元素
    for (int k = 0;k<getSize()-1;k++){
        int is ; //主元素所在行号
        double max = 0;
        for (int i = k;i<getSize();i++){
            for(int j = k;j<getSize();j++){
                double t = fabs(element(i,j)); //fabs返回绝对值
                if (t > max){
                    max = t;
                    is  = i;
                    js[k] = j;

                }
            }
        }
        if (max == 0){
        delete[] js;
        return false;}else{
            //通过行列变换将主元素交换到k行k列
            if (js[k] != k)
            for (int i = 0;i<getSize();i++)
                swap(element(i,k),element(i,js[k]));
            if (is != k)
            for (int j = 0;j<getSize();j++)
                swap(element(k,j),element(is,j));
            swap(sums[k],sums[is]);
        }
        double major = element(k,k);
        for (int j = k+1;j<getSize();j++)
            element(k,j) /= major; //对改行元素归一
        sums[k] /= major;
        for (int i=k+1;i<getSize();i++){
            for(int j=k+1;j<getSize();j++)
                element(i,j) -= element(i,k)*element(k,j);
        sums[i] -= element(i,k) * sums[k];}
    }
// 判断剩下元素是否为0
double d  = element(getSize()-1,getSize()-1);
if (fabs(d)< 1e-15){
    delete[] js;
    return false;
}
solution[getSize() -1] = sums[getSize() - 1]/d;
for (int i = getSize() - 2;i>=0;i--){
    double t = 0.0;
    for (int j = i+1;j<= getSize()-1;j++){
        t += element(i,j)*solution[j];
    }
    solution[i] = sums[i] - t;
}
js[getSize() -1] = getSize()-1;
for (int k = getSize() - 1 ;k>=0;k--)
    if (js[k] != k) swap(solution[k],solution[js[k]]);
delete[] js;
return true;
}
void LinerEqu::printSolution() const {
    cout <<"The result is"<<endl;
    for (int i =0;i<getSize();i++)
        cout<<"x["<<i<<"]="<<solution[i]<<endl;
}