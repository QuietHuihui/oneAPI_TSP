#include<iostream>
#include <sycl/sycl.hpp>
#include<CL/sycl.hpp>
#include<oneapi/dpl/random>
#include<vector>
#include<ctime>
#include<cstdlib>
using namespace std;
int main() {
	int CITY_NUM = 200;
	//vector<int>a = { 0,3,3,2,2,4,5,6,7,1,2,3,4,11,11 };
	vector<int>a;
	vector<int>b;
	for (int i = 0; i < CITY_NUM; i++) {
		a.push_back(rand() % CITY_NUM);
		b.push_back(rand() % CITY_NUM);
	}
	cout << "test--------" << endl;
	cout << a[0] << endl;
	cout << "test-------" << endl;
	vector<int>avec(CITY_NUM, 0);
	vector<int>bvec(CITY_NUM, 0);
	sycl::buffer am_buf(avec);
	sycl::buffer bm_buf(bvec);
	sycl::buffer a_buf(a);
	sycl::buffer b_buf(b);

	sycl::queue{}.submit([&](sycl::handler& h) {
		sycl::accessor am(am_buf, h, sycl::read_write);
		sycl::accessor bm(bm_buf, h, sycl::read_write);
		sycl::accessor _a(a_buf, h, sycl::read_write);
		sycl::accessor _b(b_buf, h, sycl::read_write);

		h.parallel_for(CITY_NUM, [=](sycl::item<1>idx) {
			//利用oneapi的联合分布方法生成随机数
			oneapi::dpl::minstd_rand engine_1(idx*100, idx.get_linear_id());
			//范围在0到100之间
			oneapi::dpl::uniform_int_distribution<int>distr_1(0, CITY_NUM);
			//利用oneapi的联合分布方法生成随机数
			oneapi::dpl::minstd_rand engine_2(idx * 120, idx.get_linear_id());
			//范围在0到100之间
			oneapi::dpl::uniform_int_distribution<int>distr_2(0, CITY_NUM);

			if (am[_a[idx]] == 0)am[_a[idx]]++;
			else if (am[_a[idx]] != 0) {
;				auto rd_1 = distr_1(engine_1);

				while (am[rd_1] != 0) {
					rd_1 = distr_1(engine_1);
				}
				_a[idx] = rd_1;
				am[rd_1]++;
			}
			if (bm[_b[idx]] == 0)bm[_b[idx]]++;
			else if (bm[_b[idx]] != 0) {
				auto rd_2 = distr_2(engine_2);
				while (bm[rd_2] != 0) {
					rd_2 = distr_2(engine_2);
				}
				_b[idx] = rd_2;
				bm[rd_2]++;
			}
			});
		});

	sycl::host_accessor result{a_buf};
	sycl::host_accessor _result{ b_buf };

	for (int i = 0; i < CITY_NUM; i++) {
		cout << a[i] << ',';
	}
	cout << endl;
	for (int i = 0; i < CITY_NUM; i++) {
		cout << b[i] << ',';
	}
	cout << endl;
}