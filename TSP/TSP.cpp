#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#include<vector>
#include<cmath>
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
		//初始化解决方案：到达各个城市的顺序
		this->solution = vector<int>{0,1,2,4,3,5,6,7,8,9,10,11,12,13,14};
	}

	//城市坐标
	vector<pair<int, int>>city;
	//城市的旅行顺序
	vector<int>solution;

	//初始化城市，随机生成坐标
	void initCity() {
			vector<pair<int, int>>cities(CITY_NUM);
			sycl::buffer<pair<int, int>>a{ cities };
			sycl::queue{}.submit([&](sycl::handler& h) {
				sycl::accessor out{ a,h };
				h.parallel_for(CITY_NUM, [=](sycl::item<1>idx) {

					//利用oneapi的联合分布方法生成随机数
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

			//把并行生成的随机数复制给类的成员变量city
			this->city = cities;
	}

	//展示随机生成的城市坐标
	void showCity() {
		cout << "生成随机城市的坐标: " << endl;
		for (int i = 0; i < this->city.size(); i++) {
			cout << i<<' ' << '(' << city[i].first << ',' << city[i].second << ')' << endl;
		}

	}

	//评估函数，计算当前解决方案的距离之和
	int calDistance(vector<int>solution) {

		sycl::queue q;

		//把solution后移一位得到用来相减的vector
		vector<int>r_solution(solution.begin()+1,solution.end());
		r_solution.push_back(solution[0]);

		//存放((x1-x2)^2+(y1-y2)^2)^0.5的vector
		vector<int>sub_vec(CITY_NUM);


		sycl::buffer a_buf(solution);
		sycl::buffer b_buf(r_solution);
		sycl::buffer city_buf(this->city);
		sycl::buffer sub_buf(sub_vec);

		//并行地计算距离之和
		for (size_t i = 0; i < CITY_NUM; i++) {
			q.submit([&](sycl::handler& h) {
				sycl::accessor a(a_buf, h, sycl::read_only);
				sycl::accessor b(b_buf, h, sycl::read_only);
				sycl::accessor cty(city_buf, h, sycl::read_only);
				sycl::accessor sub(sub_buf, h, sycl::write_only, sycl::no_init);

				h.parallel_for(CITY_NUM, [=](auto i) {
					sub[i] = pow(pow(cty[a[i]].first - cty[b[i]].first, 2) + pow(cty[a[i]].second - cty[b[i]].second, 2), 0.5);
					});

				});
		};
		q.wait();

		//输出查看每一个点之间的距离
		//cout << "计算执行成功" << endl;
		//for (int i = 0; i < CITY_NUM; i++)
		//	cout << sub_vec[i] << endl;

		//把并行计算得到的每一个点的距离加起来，得到当前solution的评估值
		int result = 0;
		for (int i = 0; i < CITY_NUM; i++)
			result += sub_vec[i];
		cout << "当前评估值是: " << result << endl;
		return result;
	}


};
int main() {
	TSP tsp;
	tsp.initCity();
	tsp.showCity();
	vector<int>ivec{0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
	tsp.calDistance(ivec);
}