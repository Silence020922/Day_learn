#ifndef __ACCUMULATE_H__
#define __ACCUMULATE_H__
# include "date.h"
class Accumulator{
    private:
    Date lastDate;
    double value; //数值的当前值
    double sum; //数值按日累加总和
    public:
    Accumulator(const Date &date,double value):lastDate(date),value(value),sum(0){};
    double getSum(const Date &date) const {return sum+value*date.distance(lastDate);}
    void change(const Date &date,double value){sum = getSum(date);lastDate = date;this->value = value;}
    void reset(const Date &date,double value){lastDate = date;this -> value = value;sum =0;} //将日期变成现在日期，数值变为value累加器清零

}; 


#endif