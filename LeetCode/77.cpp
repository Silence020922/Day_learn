class Solution {
void dfs(vector<int>& path, vector<vector<int>>& ans, int idx, int k){
    if(path.size() == k) {ans.push_back(path);return;}
    for (int j = idx-1;j>0;j--){
        path.push_back(j);
        dfs(path,ans,j,k);
        path.pop_back();
    }
    }
public:
    vector<vector<int>> combine(int n, int k) {
        vector<int> path;
        vector<vector<int>> ans;
        if (n==k) {
            for (int i =0;i<n;i++){path.push_back(i+1);}
            ans.push_back(path);
            return ans;
            }
        dfs(path,ans,n+1,k);
        return ans;
    }
};

