class Solution {
public:
    bool carPooling(vector<vector<int>>& trips, int capacity) {
        vector<int> diff(1002);
        for (auto trip:trips) {
            diff[trip[1]] += trip[0];
            diff[trip[2]] -= trip[0];
        }
        int num = 0;
        for (int i:diff){
            num += i;
            if (num > capacity) return false;
        }
        return true;
    }
};

