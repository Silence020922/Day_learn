class Solution {
public:
    int uniquePaths(int m, int n) {
        if (m ==1 && n==1){return 1;}
        long long ans=1;
        for (int i=1;i<n;i++) {
            ans = ans*(m-1+i)/(i);
        }
        return ans;
       }
};

