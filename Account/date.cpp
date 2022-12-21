# include "date.h"
# include <iostream>
# include <cstdlib>
using namespace std;
namespace{// namespace 使得下面定义只在该文件内有效
    const int DAY_BEFORE_MONTH[] = {0,31,59,90,120,151,181,212,243,273,304,334,365};
}

Date::Date(int year,int month, int day):year(year),month(month),day(day){
    if (day <= 0 || day>getMaxday()){
        cout <<"Invalid date:";
        show();
        cout<<endl;
        exit(1);
    }
    int years = year -1;
    totalDays = years*365 + years/4 - years/100 + years/400 + DAY_BEFORE_MONTH[month -1]  + day;
    if (isLeapyear() && month > 2){totalDays ++;}
}

int Date::getMaxday() const{
    if (isLeapyear() && month==2){return 29;}
    else{return DAY_BEFORE_MONTH[month] - DAY_BEFORE_MONTH[month -1];}
}

void Date::show() const{
    cout<<getYear()<<"-"<<getMonth()<<"-"<<getDay();
}
