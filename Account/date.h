#ifndef __DATE_H__
#define __DATE_H__
class Date{ //闰年判断
    private:
    int year,month,day,totalDays;
    public:
    Date(int year,int month,int day);
    int getYear() const {return year;}
    int getMonth() const {return month;}
    int getDay() const {return day;}
    int getMaxday() const; //获得当月有多少天
    bool isLeapyear() const{
        return year%4==0 && year%100!=0 || year%400 == 0;
    }
    void show() const; //输出当前日期
    int distance(const Date &date) const{
        return totalDays -date.totalDays;
    }
};
#endif