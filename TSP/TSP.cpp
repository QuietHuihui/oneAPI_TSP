#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#include<vector>
#include<CL/sycl.hpp>
#include<oneapi/dpl/random>
using namespace std;
#define GROUP_NUM 100    //种群规模
#define CITY_NUM 15     //城市数量
#define ITERATION_NUM 1000   //最大迭代次数
#define Pc 0.9      //交叉率
#define Pm 0.1     //变异率

std::int64_t city_num = 15;

class TSP {
public:

	TSP() {

	}

	//城市坐标
	vector<pair<int, int>>city;

	void initCity() {
			vector<pair<int, int>>cities(CITY_NUM);
			sycl::buffer<pair<int, int>>a{ cities };
			sycl::queue{}.submit([&](sycl::handler& h) {
				sycl::accessor out{ a,h };
				h.parallel_for(CITY_NUM, [=](sycl::item<1>idx) {
					oneapi::dpl::minstd_rand engine(777, idx.get_linear_id());
					oneapi::dpl::uniform_int_distribution<int>distr(0, 100);
					auto res1 = distr(engine);
					auto res2 = distr(engine);
					out[idx].first = res1;
					out[idx].second = res2;
					});
				});
			//不知为何要加上这一行才能够成功地赋值
			sycl::host_accessor result{ a };
			//测试输出
			//for (int i = 0; i < CITY_NUM; i++)
			//	cout << result[i].first << ' ' << result[i].second << endl;
			this->city = cities;
	}

	void showCity() {
		cout << "生成随机城市的坐标: " << endl;
		for (int i = 0; i < this->city.size(); i++) {
			cout << i<<' ' << '(' << city[i].first << ',' << city[i].second << ')' << endl;
		}

	}
};
int main() {
	TSP tsp;
	tsp.initCity();
	tsp.showCity();
}