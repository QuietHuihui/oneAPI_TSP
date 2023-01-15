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
		this->solution = vector<int>{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14};
	}

	//城市坐标
	vector<pair<int, int>>city;
	//城市序列
	vector<int>solution;

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

	void calDistance() {

		//基本思路
		//先并行地把x和y坐标各自相减
		//然后给x和y坐标相减的结果各自平方
		//平方的结果，对应下标的相加到一个新的vector里
		//把新vector里的数并行地开根号
		//把这些开根的结果全部加起来，返回

		sycl::queue q;
		int result;

		//把solution后移一位得到用来相减的vector
		vector<int>r_solution(this->solution.begin()+1,this->solution.end());
		r_solution.push_back(this->solution[0]);

		//第一步，先将solution和r_solution的内容相减
		//存放x坐标相减结果的vector
		vector<int>sub_parallel_x(CITY_NUM);
		//存放y坐标相减结果的vector
		vector<int>sub_parallel_y(CITY_NUM);
		//存放((x1-x2)^2+(y1-y2)^2)^0.5的vector
		vector<int>sub_vec(CITY_NUM);


		sycl::buffer a_buf(this->solution);
		sycl::buffer b_buf(r_solution);
		sycl::buffer city_buf(this->city);
		//sycl::buffer sub_buf_x(sub_parallel_x);
		//sycl::buffer sub_buf_y(sub_parallel_y);
		sycl::buffer sub_buf(sub_vec);

		//进行并行减
		for (size_t i = 0; i < CITY_NUM; i++) {
			q.submit([&](sycl::handler& h) {
				sycl::accessor a(a_buf, h, sycl::read_only);
				sycl::accessor b(b_buf, h, sycl::read_only);
				sycl::accessor cty(city_buf, h, sycl::read_only);
				//sycl::accessor sub_x(sub_buf_x, h, sycl::write_only, sycl::no_init);
				//sycl::accessor sub_y(sub_buf_y, h, sycl::write_only, sycl::no_init);
				sycl::accessor sub(sub_buf, h, sycl::write_only, sycl::no_init);

				h.parallel_for(CITY_NUM, [=](auto i) {
					//sub_x[i] = pow(cty[a[i]].first - cty[b[i]].first,2);
					//sub_y[i] = pow(cty[a[i]].second - cty[b[i]].second,2);
					sub[i] = pow(pow(cty[a[i]].first - cty[b[i]].first, 2) + pow(cty[a[i]].second - cty[b[i]].second, 2), 0.5);
					});

				});
		};
		q.wait();



		cout << "计算执行成功" << endl;
	}
};
int main() {
	TSP tsp;
	tsp.initCity();
	tsp.showCity();
	tsp.calDistance();
}