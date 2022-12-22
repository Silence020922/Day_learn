#ifndef _LINEREQU_H
#define _LINEREQU_H
#include "Matrix.h"
class LinerEqu:public Matrix{
    public:
    LinerEqu(int size=2);
    ~LinerEqu();
    void setLinerEqu(const double* a,const double* b);
    bool solve();
    void printlinerEqu() const;
    void printSolution() const;
    private:
    double* sums;
    double* solution;
};
#endif