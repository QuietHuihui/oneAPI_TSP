#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#include<vector>
#include<CL/sycl.hpp>
#include<oneapi/dpl/random>
using namespace std;
#define GROUP_NUM 100    //��Ⱥ��ģ
#define CITY_NUM 15     //��������
#define ITERATION_NUM 1000   //����������
#define Pc 0.9      //������
#define Pm 0.1     //������

std::int64_t city_num = 15;

class TSP {
public:

	TSP() {

	}

	//��������
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
			//��֪Ϊ��Ҫ������һ�в��ܹ��ɹ��ظ�ֵ
			sycl::host_accessor result{ a };
			//�������
			//for (int i = 0; i < CITY_NUM; i++)
			//	cout << result[i].first << ' ' << result[i].second << endl;
			this->city = cities;
	}

	void showCity() {
		cout << "����������е�����: " << endl;
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