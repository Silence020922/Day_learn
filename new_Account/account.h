#ifndef _ACCOUNT_H_
#define _ACCOUNT_H_
# include "date.h"
# include "accumulate.h"
# include <string>

class  Account
{
private:
    std::string id;
    double balance; //账户余额
    static double total; //类静态成员,银行总金额
protected:
    Account(const Date &date,const std::string &id); //构造函数
    void record(const Date &date,double amount,const std::string &desc); //记一笔账
    void error(const std::string &msg) const; //报告错误信息
public:
    const std::string &getId() const {return id;} //三成员输出
    double getBalance()const{return balance;}
    static double getTotle () {return total;}
    void show() const;
};

class SavingAccount: public Account{
private: 
    Accumulator acc; //辅助计算利息的累加器
    double rate;
public:
    SavingAccount(const Date &date,const std::string &id,double rate);
    double getRate(){return rate;}
    void deposit(const Date &date,double amount,const std::string &desc);
    void withdraw(const Date &date,double amount,const std::string &desc);
    void settle(const Date &date); //利息结算
};

class CreditAccount:public Account{
private:
    Accumulator acc; //辅助计算利息的累加器
    double rate;
    double credit; //可透支额度
    double fee; //信用卡年费
    double getDebt() const {
        double balance = getBalance();
        return (balance<0?balance:0);
    }
public:
    CreditAccount(const Date &date,const std::string &id,double credit,double rate,double fee);
    double getCredit() const {return credit;}
    double getRate() const {return rate;}
    double getFee()const {return fee;}
    double getAvailableCredit() const {
        if (getBalance() < 0)
        return credit + getBalance();
        else 
        return credit;}
    void deposit(const Date &date,double amount,const std::string &desc);
    void withdraw(const Date &date,double amount,const std::string &desc);
    void settle(const Date &date);
    void show() const;
};
#endif //_ACCOUNT_H_
