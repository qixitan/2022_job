# 编程

## 数组相邻元素和为奇数

输入一个整数 `n`，请输出构造一个 `n*n`的数组，使得数组相邻元素和为奇数。

```c++
#include<iostream>
using namespace std;

int main(){
    int n;
    cin >> n;
    // 奇数行正序输出、偶数行分情况讨论
    if(n%2==1){
        // 输入n为奇数、则奇数行、偶数行正序输出
        for(int i = 1; i<=n*n; i++){
            cout << i << " ";
            if(i%n==0) cout << endl;
        }
    }else{
        // 输入n为偶数，则奇数行正序输出、偶数行倒叙输出；
        for(int i = 1; i<=n; i++){
            for(int j = 1; j<=n; j++){
                if(i%2==1){
                    // 奇数行
                    cout << (i-1)*n + j<< " ";
                }else{
                    // 偶数行
                    cout << i*n - j + 1 << " ";
                }
            }
            cout << endl;
        }
    }
    return 0;
}
```

## 包含min函数的栈

设计一个支持push，pop，top等操作并且可以在O(1)时间内检索出最小元素的堆栈。

- push(x)–将元素x插入栈中
- pop()–移除栈顶元素
- top()–得到栈顶元素
- getMin()–得到栈中最小元素

**tips: ** getMin只是找到最小值是哪个，没有啥移动删除的操作，只在push和pop才有改变栈内容的操作。

```cpp
class Minstack{
private:
	stack<int> stkIn;
	stack<int> stkMin;
public:
    Minstack(){}
    void push(int x){
        stkIn.push(x);
        if(sthMin.empty || sktMin.top()>=x){
            sthMin.push(x);
        }
    }
    void pop(){
        if(!stkMin.empty() && stkIn.top()==stkMin.top()){
            stkMin.pop();
        }
        if(!stkIn.empty()){
            stkIn.pop();
        }
    }
    void top(){
        return stkIn.top();
    }
    int getMin(){
        return stkMin.top();
    }
}
```

## 国王排队汇报问题

第一行三个正数 $m、 n、 id$ 分别表示大臣数量、重要性由几个方面，来找帮忙的大臣 $id$序号，接下来 $n$行没行由 $m$个正整数，第 $i$行第 $j$个数字表示为 $a_{ij}$ 表示为第 $i$个大臣汇报的第 $j$件事情的重要性。

```bash
样例输入：
3 3 2
90 90 90
80 100 90  // 这个大臣第几
80 85 85
输出：
2
```

```cpp
#include <iostream>
#include<vector>
using namespace std;

int main()
{
	int n, m, id;
	cin>>n >> m>>id;
	vector<int> record(n, 0);
	for(int i = 0; i<n; i++){
		int sum = 0;
		int x;
		for(int j = 0; j<m; j++){
			cin >> x;
			sum += x;
        }
		record[i] = sum;}
	id = id- 1;
	int res  = 1;
	for(int i = 0; i<id; i++){
		if(record[i]>=record[id]) res++;
    }
	for(int i = id+1; i<record.size(); i++){
		if(record[i]>record[id]) res++;
	}
	cout << res << endl;
   return 0;
}
```

## 计算未被消除的方块数目

当一个方块被另一个方块完全覆盖，则删除此方块，输出剩下的方块数目

```txt
3  // 表示有多少块方块
1 8   // 方块左右坐标
3 10 
2 8
```

```c++
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

static bool cmp(pair<int, int>a, pair<int, int> b){
    return a.fist-a[0].second >=b[1].fist-b[0].second;
}

int main(){
    int n;
    cin << n;
    pair<int, int> p;
    vector<pair<int, int>> rec(n);
    for(int i = 0; i<n; i++){
        cin << p.fist<< p.second;
        rec[i] = p;
    }
    sort(rec.begin(), rec.end(), cmp);
	vector<bool> sum(n, false);
	for(int i = 0; i<n-1; i++){
        for(int j = i+1; j < n; j++){
			cout<<rec[i].first << " "<< rec[i].second <<endl;
			cout<<rec[j].first << " "<< rec[j].second <<endl;
            if(rec[i].first <= rec[j].first && rec[i].second >= rec[j].second) sum[j]=true;
        }
    }
    int res = 0;
    for(bool b: sum){
        if(b){
            res++;
        }
    }
	cout << n-res;
    return 0;
}
```

## 路径和

输入一棵二叉树和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。

从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。

```cpp
class Solution {
private:
    vector<vector<int>> res;
    vector<int> path;
    void dfs(TreeNode *root, int target){
        if(!root) return;
        path.push_back(root->val);
        target -= root->val;
        if(!root->left && !root->right && target==0) res.push_back(path);
        if(root->left) dfs(root->left, target);
        if(root->right) dfs(root->right, target);
        path.pop_back();
    }
        
public:
    vector<vector<int>> findPath(TreeNode* root, int sum) {
        if(root==NULL) return res;
        dfs(root, sum);
        return res;
        
    }
};
```



# 计算机网络

## 组播地址：[组播地址范围](https://blog.csdn.net/qq_42197548/article/details/117439115)

IANA将D类地址空间分配给IPv4组播使用，IPv4地址有32位，D类地址最高4位为1110，因此地址范围从224.0.0.0到239.255.255.255，具体分类如下：224.0.0.0~224.0.0.255 永久组播地址，用于标识一组特定的网络设备，供路由协议、拓扑查找等 使用，不用于组播转发；


| 地址范围                  | 注释                                   |
| ------------------------- | -------------------------------------- |
| 224.0.1.0~231.255.255.255 |                                        |
| 233.0.0.0~238.255.255.255 | ASM组播地址                            |
| 232.0.0.0~232.255.255.255 | SSM组播地址                            |
| 239.0.0.0~239.255.255.255 | 本地管理组播地址，仅在本地管理域内有效 |