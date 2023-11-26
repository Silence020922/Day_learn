class Solution {
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
    vector<vector<int>> ans;
    int l=candidates.size();
    sort(candidates.begin(),candidates.end());
    vector<int> path;
    dfs(ans,candidates,path,0,target,l,0);
    return ans;
    }
        
void dfs(vector<vector<int>>& ans,vector<int>& candidates,vector<int>& path,int idx,int target, int l,int      sum){
        if (sum==target){
            ans.push_back(path);
            return;
        }
        if (sum>target) return;
        for (int j = idx;j<l;j++){
            if (j-1>=idx && candidates[j]==candidates[j-1]){continue;}
            path.push_back(candidates[j]);
 dfs(ans,candidates,path,j+1,target,l,sum+candidates[j]);
            path.pop_back();
        }}
};

