class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int>> ans;
    vector<int> path;
    dfs(path,ans,nums);
    return ans;
    }
    void dfs(vector<int>& path,vector<vector<int>>& ans,vector<int>& nums){
    if (nums.empty()) {
        ans.push_back(path);
        return;
    } 
    for (int i = 0;i<nums.size();i++){
        int tmp = nums[0];
        nums.erase(nums.begin());
        path.push_back(tmp);
        dfs(path,ans,nums);
        nums.push_back(tmp);
        path.pop_back();
    }
    }
};
