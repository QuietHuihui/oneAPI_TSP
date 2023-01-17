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
#define N 100    //种群规模
#define CITY_NUM 15     //城市数量
#define GMAX 5   //最大迭代次数
#define PC 0.9      //交叉率
#define PM 0.1     //变异率

std::int64_t city_num = 15;

class TSP {
public:

	TSP() {
		//初始化城市坐标
		initCity();
		//初始化种群
		initPopulation();
		//初始化最优结果
		this->best = 0.0;
		//初始化解决方案：到达各个城市的顺序
		this->solution = vector<int>{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14};
	}

	//城市坐标
	vector<pair<int, int>>city;

	//城市的旅行顺序,最终的解决方案
	vector<int>solution;

	//最佳结果
	float best;

	//种群
	vector<vector<int>>population;

	//每个个体的评估值
	vector<float>eval;

	//每个个体被选择的概率
	vector<float>prob_select;


	//初始化城市，随机生成坐标
	void initCity() {
			cout << "开始初始化城市 。" << endl;
			vector<pair<int, int>>cities(CITY_NUM);
			sycl::buffer<pair<int, int>>a{ cities };
			sycl::queue{}.submit([&](sycl::handler& h) {
				sycl::accessor out{ a,h };
				h.parallel_for(CITY_NUM, [=](sycl::item<1>idx) {

					//利用oneapi的联合分布方法生成随机数
					oneapi::dpl::minstd_rand engine_1(777, idx.get_linear_id());
					oneapi::dpl::minstd_rand engine_2(888, idx.get_linear_id());
					//范围在0到100之间
					oneapi::dpl::uniform_int_distribution<int>distr_1(0, 100);
					oneapi::dpl::uniform_int_distribution<int>distr_2(0, 100);

					auto res1 = distr_1(engine_1);
					auto res2 = distr_2(engine_2);

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
			cout << "初始化城市成功。" << endl;
	}

	//展示随机生成的城市坐标
	void showCity() {
		cout << "生成随机城市的坐标: " << endl;
		for (int i = 0; i < this->city.size(); i++) {
			cout << i<<' ' << '(' << city[i].first << ',' << city[i].second << ')' << endl;
		}

	}

	//展示种群
	void showPopulation() {
		for (int i = 0; i < N; i++) {
			cout << "第" << i << "个个体" <<' ';
			cout << "[";
			for (int j = 0; j < CITY_NUM; j++) {
				if(j==CITY_NUM-1)cout << population[i][j] << ']';
				else cout << population[i][j] << ',';
			}
			cout << endl;
		}
	}

	//展示评估值和选择概率
	void show_eval_sel() {
		//输出所有评估值和被选择概率
		for (int i = 0; i < N; i++) {
			cout << "个体" << i << "的评估值是" << eval[i]
				<< ",被选择概率是" << prob_select[i] << endl;
		}
	}

	//初始化种群，随机产生一些旅行顺序
	void initPopulation() {
		cout << "开始初始化种群。" << endl;
		this->population = vector<vector<int>>(N);
		srand(time(0));
		for (int i = 0; i < N; i++) {
			
			//使用哈希表，以确保生成的序列中城市不重复
			unordered_map<int, int>mp;
			for (int j = 0; j < CITY_NUM; j++) {
				int num = rand() % CITY_NUM;
				//如果随机生成的数字有重复，就重新生成直到生成不重复的数字
				while (mp[num] != 0) {
					num = rand() % CITY_NUM;
				}
				mp[num]++;
				population[i].push_back(num);
			}
		}
		cout << "初始化种群成功。" << endl;
	}

	//评估函数，计算当前解决方案的距离之和的倒数(遗传算法倾向于选择最大值)
	float evaluate(vector<int>solution) {

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
		/*cout << "当前评估值是: " << 1.0 / (float)result << endl;*/
		return 1.0/(float)result;
	}

	//计算每个个体的评估值和选择概率,并保存
	void cal_eval_sel() {
		cout << "开始评估种群。" << endl;
		eval.clear();
		//把每个个体的评估值添加到eval中
		for (int i = 0; i < N; i++) {
			eval.push_back(evaluate(population[i]));
		}
		//计算每个个体的适应概率: 个体适应度/总适应度
		float total = 0.0;
		for (int i = 0; i < N; i++)
			total += eval[i];

		//并行地计算每个个体的被选择概率
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
		cout << "评价种群成功。" << endl;
	}

	//进行选择
	void select() {
		cout << "开始进行选择。" << endl;
		//计算累计概率
		vector<float>addup_prob(N);
		addup_prob[0] = this->prob_select[0];

		for (int i = 1; i < N; i++) {
			addup_prob[i] = addup_prob[i - 1] + this->prob_select[i];
		}

		//记录被选择的个体
		//轮盘赌选择法，生成0~1之间的随机数，根据累计概率选择个体
		vector<vector<int>>sel_indiv(N);
		srand(time(0));
		for (int i = 0; i < N; i++) {
			
			//测试用的输出
			//cout << "选择中，第" << i << "/" << N << "个" << endl;

			//生成0~1之间的随机数,4位小数
			float random = rand() % (10000) / (float)(10000);
			for (int j = 0; j < N; j++) {
				if (random <= addup_prob[j]) {
					sel_indiv[i] = vector<int>(this->population[j]);
					break;
				}
			}
		}
		//把选择出来的种群覆盖掉初始种群
		for (int i = 0; i < sel_indiv.size(); i++) {
			this->population[i] = vector<int>(sel_indiv[i]);
		}	
		cout << "选择成功。" << endl;
	}

	//进行交叉
	void cross() {
		cout << "开始交叉。" << endl;
		srand(time(0));

		for (int i = 0; i + 1 < N; i++) {
			//生成0~1之间的三位随机小数，如果小于交配概率就进行交配
			float random = rand() % (1000) / (float)(1000);
			if (random < PC) {
				//使用两点交叉，交叉点为中间点
				int point = CITY_NUM / 2;
				vector<int>a = vector<int>(population[i]);
				vector<int>b = vector<int>(population[i + 1]);
				//第i个个体的右半边和第i+1个个体的左半边交换
				for (int i = 0; i <= point; i++) {
					int temp = a[point + i];
					a[point + i] = b[i];
					b[i] = temp;
				}
				//去除掉重复元素
				unordered_map<int, int>mp_a;
				unordered_map<int, int>mp_b;
				//去除a中的重复元素

				//先把a交叉点前的所有元素添加到unordered map里面
				//同时把b交叉点后的所有元素添加到unordered map里面
				for (int i = 0; i < point; i++) {
					mp_a[a[i]]++;
					mp_b[b[CITY_NUM - (point)+i]]++;
				}

				//替换掉a交叉点后的重复元素
				for (int i = 0; i <= point; i++) {
					//如果发现a的交叉点后有重复的元素
					if (mp_a[a[point + i]] != 0) {
						for (int j = 0; j < CITY_NUM; j++) {
							if (mp_a[j] == 0) {
								mp_a[j]++;
								a[point + i] = j;
							}
						}
					}
					//如果发现b的交叉点前有重复的元素
					if (mp_b[b[i]] != 0) {
						for (int j = 0; j < CITY_NUM; j++) {
							if (mp_b[j] == 0) {
								mp_b[j]++;
								b[i] = j;
							}
						}
					}
				}
				//把交配完的个体保存到种群里
				population[i] = vector<int>(a);
				population[i + 1] = vector<int>(b);
			}
		}
		cout << "交叉成功。" << endl;

		//cout << "交叉之后的种群:" << endl;
		//showPopulation();
	}

	//进行变异
	void mutate() {
		cout << "开始变异。" << endl;
		//对于每个个体的每个基因随机生成0~1之间的随机数，如果小于PM，
		//那就随机地交换这一个基因和另一个基因
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
		cout << "变异成功。" << endl;

	}

	//重新计算评估值并更新最优方案
	void get_eval() {
		cout << "开始更新评估值和最优解。" << endl;
		eval.clear();
		//把每个个体的评估值添加到eval中

		for (int i = 0; i < N; i++) {
			eval.push_back(evaluate(population[i]));
			if (eval[i] > best) {
				this->best = eval[i];
				this->solution = population[i];
			}
		}

		cout << "更新评估值和最优解成功。" << endl;
	}

	//输出最优解
	void show_best() {
		cout << "最优解为:" << endl;

		//打印最优路线
		cout << "[";
		for (int i = 0; i < CITY_NUM; i++) {
			if (i == CITY_NUM - 1) {
				cout << solution[i] << ']'<<endl;
			}
			else {
				cout << solution[i] << ',';
			}
		}

		//打印最优路线的适应值
		cout << "适应值:" << endl;
		cout << best << endl;

		//打印最小花费
		cout << "路线代价为:" << endl;
		cout << (1.0) / best << endl;

	}

	//运行算法
	void run() {
		//展示初始化的城市
		showCity();
		//开始运行
		for (int i = 0; i < GMAX; i++) {
			cout << "正在运行第" << i << '/' << GMAX-1 << "代。" << endl;
			cal_eval_sel();
			select();
			cross();
			mutate();
			get_eval();
			cout<<"第" << i << '/' << GMAX-1 << "代的最优解为:" << endl;
			show_best();
		}
		cout << "全局最优解为:" << endl;
		show_best();
	}
};
int main() {
	TSP tsp;
	tsp.run();
}