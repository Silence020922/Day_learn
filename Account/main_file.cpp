# include "account.h"
# include <iostream>
using namespace std;
int main() {
    Date date(2018,11,1);
    Savingaccount accounts[] = {
        Savingaccount(date,"03755217",0.015),
        Savingaccount(date,"02342342",0.015)
    };
    const int n = sizeof(accounts)/sizeof(Savingaccount);
    accounts[0].deposit(Date(2008,11,5),5000,"salary");
    accounts[1].deposit(Date(2008,11,25),10000,"sell stock 0323");
    accounts[0].deposit(Date(2008,12,5),5500,"salary");
    accounts[1].withdraw(Date(2008,12,20),4000,"buy a game");
    cout << endl;
    for (int i =0;i<n;i++){
        accounts[i].settle(Date(2009,1,1));
        accounts[i].show();
        cout<<endl;
    }
    cout<<"TOTAL\t"<<Savingaccount::getTotle()<<endl;
    return 0 ;
}