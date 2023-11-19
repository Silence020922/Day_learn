class Solution {
public:
    bool isValid(string s) {
        int n = s.length();
        if (n%2 != 0) return false;
        stack <char>stk;
        map<char,char> M = {
            {')','('},
            {']','['},
            {'}','{'}
        };
        for (char ch : s){
            if (ch=='(' || ch == '[' || ch == '{')
            {stk.push(ch);}else{
                if(!stk.empty() && stk.top() == M[ch]){
                    stk.pop();
        }else{return false;}
        }
        }
        return stk.empty();
    }
};
