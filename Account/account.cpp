#include "account.h"
#include <iostream>
# include <cmath>
using namespace std;

double Savingaccount::total = 0;
Savingaccount::Savingaccount(const Date &date,const string &id,double rate):lastDate(date),id(id),rate(rate),balance(0),accumulation(0){
    date.show();
    cout<<"\t#"<<id<<"created"<<endl;
}
void Savingaccount::record(const Date &date,double amount,const string &desc){
    accumulation = accumulate(date);
    lastDate = date;
    amount = floor(100*amount+0.5)/100; //实现四舍五入
    balance += amount;
    total += amount;
    date.show();
    cout<<"\t#"<<id<<'\t'<<amount<<'\t'<<balance<<'\t'<<desc<<endl;
}
void Savingaccount::error(const string &msg) const{
    cout<<"Error{#"<<id<<"):"<<msg<<endl;
}
void Savingaccount::withdraw(const Date &date,double amount,const string &desc){
    if (amount > getBalance())
    {cout<<"sorry,you don't have enough money"<<endl;}
    else
    record(date,-amount,desc);
}
void Savingaccount::deposit(const Date &date,double amount,const string &desc){
    record(date,amount,desc);
}
void Savingaccount::settle(const Date &date){
    double interest = accumulate(date)*rate/date.distance(Date(date.getYear()-1,1,1));
    if (interest != 0){
        record(date,interest,"interest");
        accumulation = 0;}
}
void Savingaccount::show() const{
    cout<<id<<"\tBanlance"<<balance<<endl;
}
