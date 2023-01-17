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
using namespace std;
#define N 100    //��Ⱥ��ģ
#define CITY_NUM 15     //��������
#define GMAX 5   //����������
#define PC 0.9      //������
#define PM 0.1     //������

std::int64_t city_num = 15;

class TSP {
public:

	TSP() {
		//��ʼ����������
		initCity();
		//��ʼ����Ⱥ
		initPopulation();
		//��ʼ�����Ž��
		this->best = 0.0;
		//��ʼ���������������������е�˳��
		this->solution = vector<int>{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14};
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
			//��֪Ϊ��Ҫ������һ�в��ܹ��ɹ��ظ�ֵ
			sycl::host_accessor result{ a };
			//�������
			//for (int i = 0; i < CITY_NUM; i++)
			//	cout << result[i].first << ' ' << result[i].second << endl;

			//�Ѳ������ɵ���������Ƹ���ĳ�Ա����city
			this->city = cities;
			cout << "��ʼ�����гɹ���" << endl;
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
	float evaluate(vector<int>solution) {

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
		/*cout << "��ǰ����ֵ��: " << 1.0 / (float)result << endl;*/
		return 1.0/(float)result;
	}

	//����ÿ�����������ֵ��ѡ�����,������
	void cal_eval_sel() {
		cout << "��ʼ������Ⱥ��" << endl;
		eval.clear();
		//��ÿ�����������ֵ��ӵ�eval��
		for (int i = 0; i < N; i++) {
			eval.push_back(evaluate(population[i]));
		}
		//����ÿ���������Ӧ����: ������Ӧ��/����Ӧ��
		float total = 0.0;
		for (int i = 0; i < N; i++)
			total += eval[i];

		//���еؼ���ÿ������ı�ѡ�����
		vector<float>total_vec(N, total);
		vector<float>prob(N, 0.0);
		sycl::buffer eval_buf(this->eval);
		sycl::buffer total_buf(total_vec);
		sycl::buffer prob_buf(prob);

		sycl::queue q;
		for (size_t i = 0; i < N; i++) {
			q.submit([&](sycl::handler& h) {
				sycl::accessor eval_acc(eval_buf, h, sycl::read_only);
				sycl::accessor total_acc(total_buf, h, sycl::read_only);
				sycl::accessor prob_acc(prob_buf, h, sycl::write_only, sycl::no_init);

				h.parallel_for(N, [=](auto i) {
					prob_acc[i] = eval_acc[i] / total_acc[i];
					});
				});
		}
		q.wait();
		this->prob_select = prob;
		cout << "������Ⱥ�ɹ���" << endl;
	}

	//����ѡ��
	void select() {
		cout << "��ʼ����ѡ��" << endl;
		//�����ۼƸ���
		vector<float>addup_prob(N);
		addup_prob[0] = this->prob_select[0];

		for (int i = 1; i < N; i++) {
			addup_prob[i] = addup_prob[i - 1] + this->prob_select[i];
		}

		//��¼��ѡ��ĸ���
		//���̶�ѡ�񷨣�����0~1֮���������������ۼƸ���ѡ�����
		vector<vector<int>>sel_indiv(N);
		srand(time(0));
		for (int i = 0; i < N; i++) {
			
			//�����õ����
			//cout << "ѡ���У���" << i << "/" << N << "��" << endl;

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
				//ʹ�����㽻�棬�����Ϊ�м��
				int point = CITY_NUM / 2;
				vector<int>a = vector<int>(population[i]);
				vector<int>b = vector<int>(population[i + 1]);
				//��i��������Ұ�ߺ͵�i+1����������߽���
				for (int i = 0; i <= point; i++) {
					int temp = a[point + i];
					a[point + i] = b[i];
					b[i] = temp;
				}
				//ȥ�����ظ�Ԫ��
				unordered_map<int, int>mp_a;
				unordered_map<int, int>mp_b;
				//ȥ��a�е��ظ�Ԫ��

				//�Ȱ�a�����ǰ������Ԫ����ӵ�unordered map����
				//ͬʱ��b�����������Ԫ����ӵ�unordered map����
				for (int i = 0; i < point; i++) {
					mp_a[a[i]]++;
					mp_b[b[CITY_NUM - (point)+i]]++;
				}

				//�滻��a��������ظ�Ԫ��
				for (int i = 0; i <= point; i++) {
					//�������a�Ľ��������ظ���Ԫ��
					if (mp_a[a[point + i]] != 0) {
						for (int j = 0; j < CITY_NUM; j++) {
							if (mp_a[j] == 0) {
								mp_a[j]++;
								a[point + i] = j;
							}
						}
					}
					//�������b�Ľ����ǰ���ظ���Ԫ��
					if (mp_b[b[i]] != 0) {
						for (int j = 0; j < CITY_NUM; j++) {
							if (mp_b[j] == 0) {
								mp_b[j]++;
								b[i] = j;
							}
						}
					}
				}
				//�ѽ�����ĸ��屣�浽��Ⱥ��
				population[i] = vector<int>(a);
				population[i + 1] = vector<int>(b);
			}
		}
		cout << "����ɹ���" << endl;

		//cout << "����֮�����Ⱥ:" << endl;
		//showPopulation();
	}

	//���б���
	void mutate() {
		cout << "��ʼ���졣" << endl;
		//����ÿ�������ÿ�������������0~1֮�������������С��PM��
		//�Ǿ�����ؽ�����һ���������һ������
		srand(time(0));
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
		//��ÿ�����������ֵ��ӵ�eval��

		for (int i = 0; i < N; i++) {
			eval.push_back(evaluate(population[i]));
			if (eval[i] > best) {
				this->best = eval[i];
				this->solution = population[i];
			}
		}

		cout << "��������ֵ�����Ž�ɹ���" << endl;
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
		for (int i = 0; i < GMAX; i++) {
			cout << "�������е�" << i << '/' << GMAX-1 << "����" << endl;
			cal_eval_sel();
			select();
			cross();
			mutate();
			get_eval();
			cout<<"��" << i << '/' << GMAX-1 << "�������Ž�Ϊ:" << endl;
			show_best();
		}
		cout << "ȫ�����Ž�Ϊ:" << endl;
		show_best();
	}
};
int main() {
	TSP tsp;
	tsp.run();
}