class Solution {
public:
    const int mod = 1e9 + 7;

    int sumSubarrayMins(vector<int>& arr) {
        int n = arr.size();
        stack<int> st;
        st.push(-1);
        vector<int> idx(n, 0);
        // 单调栈
        for (int i = 0; i < n; i++) {
            while (st.size() > 1 && arr[st.top()] > arr[i]) st.pop();
            idx[i] = st.top();
            st.push(i);
        }
        
        // 动态规划 + 贡献法
        long long ans = 0;
        vector<long long> f(n, 0);
        for (int i = 0; i < n; i++) {
            f[i] = arr[i] * (i - idx[i]);
            if (idx[i] != -1) f[i] = (f[i] + f[idx[i]]) ;
            ans = (f[i] + ans);
        }

        return ans%mod;
    }
};


