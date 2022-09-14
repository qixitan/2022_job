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

## 识别有效IP地址和掩码并进行分类

### 描述

请解析IP地址和对应的掩码，进行分类识别。要求按照A/B/C/D/E类地址归类，不合法的地址和掩码单独归类。

所有的IP地址划分为 A,B,C,D,E五类

A类地址从1.0.0.0到126.255.255.255;

B类地址从128.0.0.0到191.255.255.255;

C类地址从192.0.0.0到223.255.255.255;

D类地址从224.0.0.0到239.255.255.255；

E类地址从240.0.0.0到255.255.255.255



私网IP范围是：

从10.0.0.0到10.255.255.255

从172.16.0.0到172.31.255.255

从192.168.0.0到192.168.255.255

子网掩码为二进制下前面是连续的1，然后全是0。（例如：255.255.255.32就是一个非法的掩码）

（注意二进制下全是1或者全是0均为非法子网掩码）

注意：

\1. 类似于【0.*.*.*】和【127.*.*.*】的IP地址不属于上述输入的任意一类，也不属于不合法ip地址，计数时请忽略

\2. 私有IP地址和A,B,C,D,E类地址是不冲突的



### 输入描述：

多行字符串。每行一个IP地址和掩码，用~隔开。

请参考帖子https://www.nowcoder.com/discuss/276处理循环输入的问题。

### 输出描述：

统计A、B、C、D、E、错误IP地址或错误掩码、私有IP的个数，之间以空格隔开。

### 示例1



```tex
// 输入
10.70.44.68~255.254.255.0
1.0.0.1~255.0.0.0
192.168.0.2~255.255.255.0
19..0.~255.255.255.0

// 输出
1 0 1 0 0 2 1

// 说明
10.70.44.68~255.254.255.0的子网掩码非法，19..0.~255.255.255.0的IP地址非法，所以错误IP地址或错误掩码的计数为2；
1.0.0.1~255.0.0.0是无误的A类地址；
192.168.0.2~255.255.255.0是无误的C类地址且是私有IP；
所以最终的结果为1 0 1 0 0 2 1        
```



```cpp
#include<bits/stdc++.h>
#include<string.h>
#include<sstream>
using namespace std;
vector<int> bf(string s1)
{
    vector<int> a;
    int i=0,j=0;
     while(i<s1.size())
        {
        while(i<s1.size() && s1[i]!='.')i++;
        string temp=s1.substr(j,i-j);
        if(temp.size()==0)a.push_back(-1);
        else{
        int t=stoi(temp,0);
            a.push_back(t);
        }
            i++;
            j=i;
        }
    if(a.size()==3)a.push_back(-1);
    return a;
}
bool checkym(vector<int> a)
{
    set<int> si{0,128,192,224,240,248,252,254};
    for(int i=0;i<a.size();i++)
    {
        if(a[i]<0)
        {
            return false;
        }
    }
        if(a[0]==0)return false;
        if(a[0]==255){
            if(a[1]==255)
            {
                if(a[2]==255){
                    if(si.find(a[3])!=si.end()&&a[3]!=255)return true;
                    else return false;
                }
                else
                {
                    if(si.find(a[2])!=si.end()&&a[3]==0)
                    {
                    return true;
                    }
                    else
                    return false;
                }
            }
            else
            {
                if(si.find(a[1])!=si.end()&&a[2]==0&&a[3]==0)
                    {
                    return true;
                    }
                    else 
                    return false;
            }
        }
        else
            {
                if(si.find(a[0])!=si.end()&&a[1]==0&&a[2]==0&&a[3]==0)
                return true;
                else 
                return false;
            }
}
  
  
bool checkip(vector<int> a)
{
    for(int i=0;i<a.size();i++)
    {
        if(a[i]<0)return false;
    }
    return true;
}
    
int main(){
    string s;
    int an[7];
    memset(an,0,sizeof(an));
    while(cin>>s){
        int m=0;
        string s1,s2;
        while(m<s.size() && s[m]!='~'){m++;}
        s1=s.substr(0,m);
        s2=s.substr(m+1,s.size()-m);
        vector<int> a1,a2;
        a1=bf(s1);
        a2=bf(s2);
        if(a1[0]==0||a1[0]==127)continue;
        else
        {
        //a1是处理好的ip容器a2为子网掩码
        //掩码处理
        bool fip=checkip(a1);
        bool fym=checkym(a2);
        if((a1[0]>=1&&a1[0]<=126)&&fym&&fip)  an[0]+=1;
        if((a1[0]>=128&&a1[0]<=191)&&fym&&fip)an[1]+=1;
        if((a1[0]>=192&&a1[0]<=223)&&fym&&fip)an[2]+=1;
        if((a1[0]>=224&&a1[0]<=239)&&fym&&fip)an[3]+=1;
        if((a1[0]>=240&&a1[0]<=255)&&fym&&fip){an[4]+=1;}

        if(((a1[0]==10)||(a1[0]==172&&(a1[1]>=16&&a1[1]<=31))||(a1[0]==192&&a1[1]==168))&&fym&&fip)an[6]+=1;

        if(fym+fip==1||fym+fip==0){an[5]+=1;}
        }
    }
        for(int i=0;i<7;i++){
            cout<<an[i]<<" ";
        }
}


```

## 判断掩码是否正确并返回

```cpp
#include<sstream>
vector<int> getMask(string input){
    stringstream iss(input);
    string temp;
    unsigned m = 0;
    vector<int> mask;
    while(getline(iss, temp, '.')){
        m = (m<<8) + stoi(temp);
        mask.emplace_back(stoi(temp));
    }
 
    if(m == 0 || (m^(m-1)) == 1 || ((m&(m-1)) != (m<<1)) ){//判断子网掩码网络号是否全1，主机号是否全0
        mask.clear();              //子网掩码全0或者全1都不行
    }
    if(mask.size() > 4) mask.clear();
    return mask;
}

```



## 判断是否为素数

```cpp
bool isprime(int num){
    for(int i=2; i*i<=nums; i++){
        if(num%i==0) return false;
    }
    return true;
}
```



# C++ ACM模式输入输出参考程序

```cpp
// 常用头文件
#include<iostream>
#include<string>
#include<vector>
#include<algorithm>
#include<cctype>  //isalpha() 头文件
#include<limits.h>  // INT_MIN和INT_MAX的头文件
#include<sstream>  // stringstream头文件
#include<bits/stdc++.h>  // 万能头文件 quan'bu
```



```cpp
#include<iostream>
#include<sstread>
#include<string>
#include<vector>
#include<algorithm>
#include<limits.h>  // INT_MIN和INT_MAX的头文件
#include<sstream>  // stringstream头文件
using namespace std;
struct stu{
    string name;
    int num;
};


// 1、直接输入一个数
int main(){
    int n = 0;
    while(cin >> n){
        cout <, n<<endl;
    }
    return -1;
}

// 2、直接输入一个字符串
int main(){
    string str;
    while(cin >> str){
        cout << str << endl;
    }
    return -1;
}

// 3、只读取一个字符
int main(){
    char ch;
    while(cin >. ch){
        cout << ch <<endl;
    }
    return -1;
}

// 3.1给定一个数，表示有多少组数（可能时数字和字符串的组合），然后读取
int main(){
    int n = 0;
    while(cin >> n){    // 每次读取n+1个数，即一个样例有n+1个数
        vector<int> num(n);
        for(int i = 0; i < n; i++){
            cin >> nums[i];
        }
        // 处理这n+1个字符串
        for(int i = 0; i<n;i++){
            cout << nums[i] <<endl;
        }
    }
}

//3.2 首先给一个数字，表示需读取n个字符，然后顺序读取n个字符
int main() {
	int n = 0;
	while (cin >> n) {  //输入数量
		vector<string> strs;
		for (int i = 0; i < n; i++) {
			string temp;
			cin >> temp;
			strs.push_back(temp);
		}
		//处理这组字符串
		sort(strs.begin(), strs.end());
		for (auto& str : strs) {
			cout << str << ' ';
		}
	}
	return 0;
}


// 4.为给定数据个数，但是每一行代表一组数据，每个数据之间用空格隔开
//4.1使用getchar() 或者 cin.get() 读取判断是否是换行符，若是的话，则表示该组数（样例）结束了，需进行处理
int main() {
	int ele;
	while (cin >> ele) {
		int sum = ele;
		// getchar()   //读取单个字符
		/*while (cin.get() != '\n') {*/   //判断换行符号
		while (getchar() != '\n') {  //如果不是换行符号的话，读到的是数字后面的空格或者table
			int num;
			cin >> num;
			sum += num;
		}
		cout << sum << endl;
	}
	return 0;
}


//4.2 给定一行字符串，每个字符串用空格间隔，一个样例为一行
int main() {
	string str;
	vector<string> strs;
	while (cin >> str) {
		strs.push_back(str);
		if (getchar() == '\n') {  //控制测试样例
			sort(strs.begin(), strs.end());
			for (auto& str : strs) {
				cout << str << " ";
			}
			cout << endl;
			strs.clear();
		}
	}
	return 0;
}

//4.3 使用getline 读取一整行数字到字符串input中，然后使用字符串流stringstream，读取单个数字或者字符。
int main() {
	string input;
	while (getline(cin, input)) {  //读取一行
		stringstream data(input);  //使用字符串流
		int num = 0, sum = 0;
		while (data >> num) {
			sum += num;
		}
		cout << sum << endl;
	}
	return 0;
}


//4.4 使用getline 读取一整行字符串到字符串input中，然后使用字符串流stringstream，读取单个数字或者字符。
int main() {
	string words;
	while (getline(cin, words)) {
		stringstream data(words);
		vector<string> strs;
		string str;
		while (data >> str) {
			strs.push_back(str);
		}
		sort(strs.begin(), strs.end());
		for (auto& str : strs) {
			cout << str << " ";
		}
	}
}

//4.5 使用getline 读取一整行字符串到字符串input中，然后使用字符串流stringstream，读取单个数字或者字符。每个字符中间用','间隔
int main() {
	string line;
	
	//while (cin >> line) {  //因为加了“，”所以可以看出一个字符串读取
	while(getline(cin, line)){
		vector<string> strs;
		stringstream ss(line);
		string str;
		while (getline(ss, str, ',')) {
			strs.push_back(str);
		}
		//
		sort(strs.begin(), strs.end());
		for (auto& str : strs) {
			cout << str << " ";
		}
		cout << endl;
	}
	return 0;
}



int main() {
	string str;

	
	//C语言读取字符、数字
	int a;
	char c;
	string s;

	scanf_s("%d", &a);
	scanf("%c", &c);
	scanf("%s", &s);
	printf("%d", a);


	//读取字符
	char ch;
	cin >> ch;
	ch = getchar();
	while (cin.get(ch)) { //获得单个字符
		;
	}
	
	//读取字符串
	cin >> str;  //遇到空白停止
	getline(cin, str);  //读入一行字符串

}
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