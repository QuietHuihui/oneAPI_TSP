#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#include<vector>
#include<cmath>
#include<ctime>
#include<algorithm>
#include<unordered_map>
#include<CL/sycl.hpp>
#include<oneapi/dpl/random>
#include<omp.h>
#include<fstream>
using namespace std;
#define N 1450    //��Ⱥ��ģ
#define CITY_NUM 15     //��������
#define GMAX 100   //����������
#define PC 0.9      //������
#define PM 0.1     //������

class TSP {
public:

	TSP() {
		//��ȡ��ǰʱ������Ϊ�ļ���
		time_t nowtime = time(NULL);
		struct tm* p;
		p = gmtime(&nowtime);
		char tmp[64];
		sprintf(tmp, "output-%d-%d-%d-%d-%d-%d.csv", 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
		string _filename = tmp;
		this->filename = _filename;

		//�����ļ������
		this->ofs.open(filename, ios::out | ios::app);
		//��ʼ����������
		initCity();
		//��ʼ����Ⱥ
		initPopulation();
		//��ʼ�����Ž��
		this->best = 0.0;
		//��ʼ���������������������е�˳��
		this->solution = vector<int>(CITY_NUM);
		this->eval = vector<float>(N);


	}

	//��������
	vector<pair<int, int>>city;

	//���е�����˳��,���յĽ������
	vector<int>solution;

	//��ѽ��
	float best;

	//��Ⱥ
	vector<vector<int>>population;

	//ÿ�����������ֵ
	vector<float>eval;

	//ÿ�����屻ѡ��ĸ���
	vector<float>prob_select;

	//����ļ���
	ofstream ofs;

	//����ļ���
	string filename;
	//��ʼ�����У������������
	void initCity() {
			cout << "��ʼ��ʼ������ ��" << endl;
			vector<pair<int, int>>cities(CITY_NUM);
			sycl::buffer<pair<int, int>>a{ cities };
			sycl::queue{}.submit([&](sycl::handler& h) {
				sycl::accessor out{ a,h };
				h.parallel_for(CITY_NUM, [=](sycl::item<1>idx) {

					//����oneapi�����Ϸֲ��������������
					oneapi::dpl::minstd_rand engine_1(777, idx.get_linear_id());
					oneapi::dpl::minstd_rand engine_2(888, idx.get_linear_id());
					//��Χ��0��100֮��
					oneapi::dpl::uniform_int_distribution<int>distr_1(0, 100);
					oneapi::dpl::uniform_int_distribution<int>distr_2(0, 100);

					auto res1 = distr_1(engine_1);
					auto res2 = distr_2(engine_2);

					out[idx].first = res1;
					out[idx].second = res2;
					});
				});
			//������һ�в��ܹ��ɹ��ظ�ֵ
			sycl::host_accessor result{ a };

			//�Ѳ������ɵ���������Ƹ���ĳ�Ա����city
			this->city = cities;
			cout << "��ʼ�����гɹ���" << endl;



			//�������ɵĳ����Լ�������Ϣ���ļ�
			ofs << "N,CITY_NUM,GMAX,PC,PM" << endl;
			ofs << N << ',' << CITY_NUM << ',' << GMAX << ',' << PC << ',' << PM << endl;
			ofs << "x,y" << endl;
			for (int i = 0; i < CITY_NUM; i++) {
				ofs << city[i].first << ',' << city[i].second << endl;
			}
			ofs << "cost,round_duration" << endl;
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

	//չʾ����ֵ��ѡ�����
	void show_eval_sel() {
		//�����������ֵ�ͱ�ѡ�����
		for (int i = 0; i < N; i++) {
			cout << "����" << i << "������ֵ��" << eval[i]
				<< ",��ѡ�������" << prob_select[i] << endl;
		}
	}

	//��ʼ����Ⱥ���������һЩ����˳��
	void initPopulation() {
		cout << "��ʼ��ʼ����Ⱥ��" << endl;
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
		cout << "��ʼ����Ⱥ�ɹ���" << endl;
	}

	//�������������㵱ǰ��������ľ���֮�͵ĵ���(�Ŵ��㷨������ѡ�����ֵ)
	void evaluate() {

		//��populationչƽ����ŵ���ʱvector��
		vector<int>pop_flat(N * CITY_NUM);

		int idx = 0;

		//����Ⱥ�е�ÿһ������ֵ��չƽ��population��
#pragma omp parallel for collapse(2)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < CITY_NUM; j++)
				pop_flat[idx++]= population[i][j];

		//���еؼ������Ⱥ��ÿһ�����������ֵ
		sycl::buffer<int, 1>pop_buf(pop_flat.data(), N * CITY_NUM);
		sycl::buffer<float, 1>eval_buf(this->eval.data(), N);
		sycl::buffer<pair<int, int>>city_buf(this->city);

		sycl::queue{}.submit([&](sycl::handler& h) {
			auto pop_acc = pop_buf.get_access<sycl::access::mode::read>(h);
			auto eval_acc = eval_buf.get_access<sycl::access::mode::write>(h);
			auto city_acc = city_buf.get_access<sycl::access::mode::read>(h);

			h.parallel_for(sycl::range<1>{N}, [=](sycl::id<1>index) {
				int idx = index;
				auto sum = 0.0;
				for (int i = index * CITY_NUM + 1; i < (index + 1) * CITY_NUM; i++) {
					sum += pow(pow(city_acc[pop_acc[i - 1]].first - city_acc[pop_acc[i]].first, 2) + pow(city_acc[pop_acc[i - 1]].second - city_acc[pop_acc[i]].second, 2), 0.5);
				}
					
				sum += pow(pow(city_acc[pop_acc[index * CITY_NUM]].first - city_acc[pop_acc[(index + 1) * CITY_NUM - 1]].first, 2) + pow(city_acc[pop_acc[index * CITY_NUM]].second - city_acc[pop_acc[(index + 1) * CITY_NUM - 1]].second, 2), 0.5);
				eval_acc[idx] = 1.0 / sum;
				});
			});
	}

	//����ÿ�����������ֵ��ѡ�����,������
	void cal_eval_sel() {
		cout << "��ʼ������Ⱥ��" << endl;
		eval.clear();
		eval = vector<float>(N);
		//��ÿ�����������ֵ��ӵ�eval��
		evaluate();
		//����ÿ���������Ӧ����: ������Ӧ��/����Ӧ��
		float total = 0.0;
		#pragma omp parallel for
		for (int i = 0; i < N; i++)
			total += eval[i];

		//���еؼ���ÿ������ı�ѡ�����
		vector<float>total_vec(N, total);
		vector<float>prob(N, 0.0);
		sycl::buffer eval_buf(this->eval);
		sycl::buffer total_buf(total_vec);
		sycl::buffer prob_buf(prob);

		sycl::queue{}.submit([&](sycl::handler& h) {
			sycl::accessor eval_acc(eval_buf, h, sycl::read_only);
			sycl::accessor total_acc(total_buf, h, sycl::read_only);
			sycl::accessor prob_acc(prob_buf, h, sycl::write_only, sycl::no_init);
			h.parallel_for(N, [=](sycl::item<1>idx) {
				prob_acc[idx] = eval_acc[idx] / total_acc[idx];
				});
			});
		sycl::host_accessor result{prob_buf};
		this->prob_select = prob;
		cout << "������Ⱥ�ɹ���" << endl;
	}

	//����ѡ��
	void select() {
		cout << "��ʼ����ѡ��" << endl;
		//�����ۼƸ���
		vector<float>addup_prob(N);
		addup_prob[0] = this->prob_select[0];

#pragma omp parallel for
		for (int i = 1; i < N; i++) {
			addup_prob[i] = addup_prob[i - 1] + this->prob_select[i];
		}

		//��¼��ѡ��ĸ���
		//���̶�ѡ�񷨣�����0~1֮���������������ۼƸ���ѡ�����
		vector<vector<int>>sel_indiv(N);
		srand(time(0));

#pragma omp parallel for collapse(2)
		for (int i = 0; i < N; i++) {
			//����0~1֮��������,4λС��
			float random = rand() % (10000) / (float)(10000);
			for (int j = 0; j < N; j++) {
				if (random <= addup_prob[j]) {
					sel_indiv[i] = vector<int>(this->population[j]);
					break;
				}
			}
		}
		//��ѡ���������Ⱥ���ǵ���ʼ��Ⱥ
#pragma omp parallel for
		for (int i = 0; i < sel_indiv.size(); i++) {
			this->population[i] = vector<int>(sel_indiv[i]);
		}	
		cout << "ѡ��ɹ���" << endl;
	}

	//���н���
	void cross() {
		cout << "��ʼ���档" << endl;
		srand(time(0));

		for (int i = 0; i + 1 < N; i++) {
			//����0~1֮�����λ���С�������С�ڽ�����ʾͽ��н���
			float random = rand() % (1000) / (float)(1000);
			if (random < PC) {
				//ʹ�õ��㽻�棬�����Ϊ���һ����
				int point = rand() % (CITY_NUM);
				vector<int>a = vector<int>(population[i]);
				vector<int>b = vector<int>(population[i + 1]);
				//��i��������Ұ�ߺ͵�i+1����������߽���
				for (int i = 0; i <= point && (point + i < CITY_NUM); i++) {
					int temp = a[point + i];
					a[point + i] = b[i];
					b[i] = temp;
				}
				//ȥ�����ظ�Ԫ��
				unordered_map<int, int>mp_a;
				unordered_map<int, int>mp_b;

				//ȥ���ظ�Ԫ��

				for (int i = 0; i < CITY_NUM; i++) {
					if (mp_a[a[i]] == 0)mp_a[a[i]]++;
					else if (mp_a[a[i]] != 0) {
						int num = rand() % CITY_NUM;
						while (mp_a[num] != 0) {
							num = rand() % CITY_NUM;
						}
						a[i] = num;
						mp_a[num]++;
					}

					if (mp_b[b[i]] == 0)mp_b[b[i]]++;
					else if (mp_b[b[i]] != 0) {
						int num = rand() % CITY_NUM;
						while (mp_b[num] != 0) {
							num = rand() % CITY_NUM;
						}
						b[i] = num;
						mp_b[num]++;
					}
				}
				//�ѽ�����ĸ��屣�浽��Ⱥ��
				population[i] = vector<int>(a);
				population[i + 1] = vector<int>(b);
			}
		}
		cout << "����ɹ���" << endl;

	}

	//���б���
	void mutate() {
		cout << "��ʼ���졣" << endl;
		//����ÿ�������ÿ�������������0~1֮�������������С��PM��
		//�Ǿ�����ؽ�����һ���������һ������
		srand(time(0));

#pragma omp parallel for collapse(2)
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < CITY_NUM; j++) {
				float random = rand() % (10000) / (float)(10000);
				if (random < PM) {
					int index = rand() % (CITY_NUM-1);
					int temp = population[i][j];
					population[i][j] = population[i][index];
					population[i][index] = temp;
				}
			}
		}
		cout << "����ɹ���" << endl;

	}

	//���¼�������ֵ���������ŷ���
	void get_eval() {
		cout << "��ʼ��������ֵ�����Ž⡣" << endl;
		eval.clear();
		eval = vector<float>(N);
		evaluate();

		float cur_best = 0.0;
		vector<int>cur_sol(CITY_NUM);
		//��ÿ�����������ֵ��ӵ�eval��

#pragma omp parallel for
		for (int i = 0; i < N; i++) {
			if (eval[i] > cur_best) {
				cur_best = eval[i];
				cur_sol = population[i];
			}
		}
		if (cur_best > this->best) {
			this->best = cur_best;
			this->solution = cur_sol;
		}
		cout << "��������ֵ�����Ž�ɹ���" << endl;

		//���һ�ε�������С����
		cout << "���ֵõ�����С������" << (1.0)/cur_best << "��" << endl;

		ofs << (1.0) / cur_best << ',';
	}

	//������Ž�
	void show_best() {
		cout << "���Ž�Ϊ:" << endl;

		//��ӡ����·��
		cout << "[";
		for (int i = 0; i < CITY_NUM; i++) {
			if (i == CITY_NUM - 1) {
				cout << solution[i] << ']'<<endl;
			}
			else {
				cout << solution[i] << ',';
			}
		}

		//��ӡ����·�ߵ���Ӧֵ
		cout << "��Ӧֵ:" << endl;
		cout << best << endl;

		//��ӡ��С����
		cout << "·�ߴ���Ϊ:" << endl;
		cout << (1.0) / best << endl;

	}

	//�����㷨
	void run() {
		//չʾ��ʼ���ĳ���
		showCity();
		//��ʼ����
		
		//��ȡ��ʼʱ��
		long long start = clock();

		for (int i = 0; i < GMAX; i++) {
			long long round_start = clock();
			cout << "�������е�" << i << '/' << GMAX-1 << "����" << endl;
			cal_eval_sel();
			select();
			cross();
			mutate();
			get_eval();
			cout<<"��" << i << '/' << GMAX-1 << "�������Ž�Ϊ:" << endl;
			show_best();
			long long round_end = clock();
			int round_duration = (round_end - round_start) * 1000 / CLOCKS_PER_SEC;
			cout << "��" << i << '/' << GMAX - 1 << "���ĺ�ʱΪ:" << round_duration <<"ms" << endl;
			ofs << round_duration << ',' << endl;
		}
		//��ȡ����ʱ��
		long long end = clock();
		//�㷨ִ�е�ʱ�䣬��λ�Ǻ���
		int duration = (end - start) * 1000 / CLOCKS_PER_SEC;

		//�������ʱ�䣬��С�����Լ����·�����ļ�
		ofs << endl;
		ofs << "duration(ms)"<< endl<<duration<<endl;
		ofs << endl;
		ofs << "min cost" << endl;
		ofs <<(1.0)/ best << endl<<endl;
		ofs << "solution" << endl;
		for (int i = 0; i < CITY_NUM; i++) {
			ofs << solution[i] << endl;
		}
		ofs.close();
		cout << "ȫ�����Ž�Ϊ:" << endl;
		show_best();
	}
};
int main() {
	TSP tsp;
	tsp.run();
}