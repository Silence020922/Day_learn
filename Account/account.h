#ifndef _ACCOUNT_H_
#define _ACCOUNT_H_
#include "date.h"
# include <string>

class  Savingaccount
{
private:
    std::string id;
    double balance; //账户余额
    double rate;
    Date lastDate;
    double accumulation;
    static double total; //类静态成员,银行总金额
    void record(const Date &date,double amount,const std::string &desc);
    void error(const std::string &msg) const; //报告错误信息
    double accumulate(const Date &date) const{
    return accumulation + balance*(date.distance(lastDate));
    };
public:
    Savingaccount(const Date &date,const std::string &id,double rate); //构造函数
    double getBalance() const {return balance;}
    const std::string &getId() const {return id;}
    double getRate() const {return rate;}
    void deposit(const Date &date,double amount,const std::string &desc);
    void withdraw(const Date &date, double amount,const std::string &desc);
    void settle(const Date &date); //结算利息函数，每年调用一次
    static double getTotle () {return total;}
    void show() const;
};
#endif //_ACCOUNT_H_
