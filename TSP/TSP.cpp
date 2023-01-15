#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#include<vector>
#include<cmath>
#include<ctime>
#include<unordered_map>
#include<CL/sycl.hpp>
#include<oneapi/dpl/random>
using namespace std;
#define N 100    //��Ⱥ��ģ
#define CITY_NUM 15     //��������
#define GMAX 1000   //����������
#define PC 0.9      //������
#define PM 0.1     //������

std::int64_t city_num = 15;

class TSP {
public:

	TSP() {
		//��ʼ���������������������е�˳��
		this->solution = vector<int>{0,1,2,4,3,5,6,7,8,9,10,11,12,13,14};
	}

	//��������
	vector<pair<int, int>>city;

	//���е�����˳��,���յĽ������
	vector<int>solution;

	//��Ⱥ
	vector<vector<int>>population;


	//��ʼ�����У������������
	void initCity() {
			vector<pair<int, int>>cities(CITY_NUM);
			sycl::buffer<pair<int, int>>a{ cities };
			sycl::queue{}.submit([&](sycl::handler& h) {
				sycl::accessor out{ a,h };
				h.parallel_for(CITY_NUM, [=](sycl::item<1>idx) {

					//����oneapi�����Ϸֲ��������������
					oneapi::dpl::minstd_rand engine(777, idx.get_linear_id());
					//��Χ��0��100֮��
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

			//�Ѳ������ɵ���������Ƹ���ĳ�Ա����city
			this->city = cities;
	}

	//չʾ������ɵĳ�������
	void showCity() {
		cout << "����������е�����: " << endl;
		for (int i = 0; i < this->city.size(); i++) {
			cout << i<<' ' << '(' << city[i].first << ',' << city[i].second << ')' << endl;
		}

	}

	//չʾ��Ⱥ
	void showPopulation() {
		for (int i = 0; i < N; i++) {
			cout << "��" << i << "������" <<' ';
			cout << "[";
			for (int j = 0; j < CITY_NUM; j++) {
				if(j==CITY_NUM-1)cout << population[i][j] << ']';
				else cout << population[i][j] << ',';
			}
			cout << endl;
		}
	}

	//��ʼ����Ⱥ���������һЩ����˳��
	void initPopulation() {
		this->population = vector<vector<int>>(N);
		srand(time(0));
		for (int i = 0; i < N; i++) {
			
			//ʹ�ù�ϣ����ȷ�����ɵ������г��в��ظ�
			unordered_map<int, int>mp;
			for (int j = 0; j < CITY_NUM; j++) {
				int num = rand() % CITY_NUM;
				//���������ɵ��������ظ�������������ֱ�����ɲ��ظ�������
				while (mp[num] != 0) {
					num = rand() % CITY_NUM;
				}
				mp[num]++;
				population[i].push_back(num);
			}
		}
	}

	//�������������㵱ǰ��������ľ���֮��
	int calDistance(vector<int>solution) {

		sycl::queue q;

		//��solution����һλ�õ����������vector
		vector<int>r_solution(solution.begin()+1,solution.end());
		r_solution.push_back(solution[0]);

		//���((x1-x2)^2+(y1-y2)^2)^0.5��vector
		vector<int>sub_vec(CITY_NUM);


		sycl::buffer a_buf(solution);
		sycl::buffer b_buf(r_solution);
		sycl::buffer city_buf(this->city);
		sycl::buffer sub_buf(sub_vec);

		//���еؼ������֮��
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

		//����鿴ÿһ����֮��ľ���
		//cout << "����ִ�гɹ�" << endl;
		//for (int i = 0; i < CITY_NUM; i++)
		//	cout << sub_vec[i] << endl;

		//�Ѳ��м���õ���ÿһ����ľ�����������õ���ǰsolution������ֵ
		int result = 0;
		for (int i = 0; i < CITY_NUM; i++)
			result += sub_vec[i];
		cout << "��ǰ����ֵ��: " << result << endl;
		return result;
	}


};
int main() {
	TSP tsp;
	tsp.initCity();
	tsp.showCity();
	vector<int>ivec{0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
	tsp.calDistance(ivec);
	tsp.initPopulation();
	tsp.showPopulation();
}