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
#define N 100000    //种群规模
#define CITY_NUM 300     //城市数量
#define GMAX 100   //最大迭代次数
#define PC 0.9      //交叉率
#define PM 0.1     //变异率

class TSP {
public:

	TSP() {
		//获取当前时间以作为文件名
		time_t nowtime = time(NULL);
		struct tm* p;
		p = gmtime(&nowtime);
		char tmp[64];
		sprintf(tmp, "output-%d-%d-%d-%d-%d-%d.csv", 1900 + p->tm_year, 1 + p->tm_mon, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
		string _filename = tmp;
		this->filename = _filename;

		//创建文件输出流
		this->ofs.open(filename, ios::out | ios::app);
		//初始化城市坐标
		initCity();
		//初始化种群
		initPopulation();
		//初始化最优结果
		this->best = 0.0;
		//初始化解决方案：到达各个城市的顺序
		this->solution = vector<int>(CITY_NUM);
		this->eval = vector<float>(N);


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

	//输出文件流
	ofstream ofs;

	//输出文件名
	string filename;
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
			//加上这一行才能够成功地赋值
			sycl::host_accessor result{ a };

			//把并行生成的随机数复制给类的成员变量city
			this->city = cities;
			cout << "初始化城市成功。" << endl;



			//保存生成的城市以及基本信息到文件
			ofs << "N,CITY_NUM,GMAX,PC,PM" << endl;
			ofs << N << ',' << CITY_NUM << ',' << GMAX << ',' << PC << ',' << PM << endl;
			ofs << "x,y" << endl;
			for (int i = 0; i < CITY_NUM; i++) {
				ofs << city[i].first << ',' << city[i].second << endl;
			}
			ofs << "cost" << endl;
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
	void evaluate() {

		//把population展平，存放到临时vector中
		vector<int>pop_flat(N * CITY_NUM);

		int idx = 0;

		//把种群中的每一个基因赋值给展平的population中
#pragma omp parallel for collapse(2)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < CITY_NUM; j++)
				pop_flat[idx++]= population[i][j];

		//并行地计算出种群中每一个个体的评估值
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

	//计算每个个体的评估值和选择概率,并保存
	void cal_eval_sel() {
		cout << "开始评估种群。" << endl;
		eval.clear();
		eval = vector<float>(N);
		//把每个个体的评估值添加到eval中
		evaluate();
		//计算每个个体的适应概率: 个体适应度/总适应度
		float total = 0.0;
		#pragma omp parallel for
		for (int i = 0; i < N; i++)
			total += eval[i];

		//并行地计算每个个体的被选择概率
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
		cout << "评价种群成功。" << endl;
	}

	//进行选择
	void select() {
		cout << "开始进行选择。" << endl;
		//计算累计概率
		vector<float>addup_prob(N);
		addup_prob[0] = this->prob_select[0];

#pragma omp parallel for
		for (int i = 1; i < N; i++) {
			addup_prob[i] = addup_prob[i - 1] + this->prob_select[i];
		}

		//记录被选择的个体
		//轮盘赌选择法，生成0~1之间的随机数，根据累计概率选择个体
		vector<vector<int>>sel_indiv(N);
		srand(time(0));

#pragma omp parallel for collapse(2)
		for (int i = 0; i < N; i++) {
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
#pragma omp parallel for
		for (int i = 0; i < sel_indiv.size(); i++) {
			this->population[i] = vector<int>(sel_indiv[i]);
		}	
		cout << "选择成功。" << endl;
	}

	//进行交叉
	void cross() {
		cout << "开始交叉。" << endl;
		srand(time(0));

		//应当先展开再来实现交叉，这里需要修改。循环中使用oneAPI会更慢
		//把population展平，存放到临时vector中
		vector<int>pop_flat(N * CITY_NUM);
		int idx = 0;
		//把种群中的每一个基因赋值给展平的population中
#pragma omp parallel for collapse(2)
		for (int i = 0; i < N; i++)
			for (int j = 0; j < CITY_NUM; j++)
				pop_flat[idx++] = population[i][j];


		vector<int>a_vec(CITY_NUM, 0);
		vector<int>b_vec(CITY_NUM, 0);
		vector<int>am_vec(CITY_NUM, 0);
		vector<int>bm_vec(CITY_NUM, 0);
		vector<int>clear_vec(CITY_NUM, 0);

		sycl::buffer<int, 1>a_buf(a_vec.data(), CITY_NUM);
		sycl::buffer<int, 1>b_buf(b_vec.data(), CITY_NUM);
		sycl::buffer<int, 1>pop_buf(pop_flat.data(), N * CITY_NUM);
		sycl::buffer<int, 1>am_buf(am_vec.data(), CITY_NUM);
		sycl::buffer<int, 1>bm_buf(bm_vec.data(), CITY_NUM);
		sycl::buffer<int, 1>clear_buf(clear_vec.data(),CITY_NUM);

		sycl::queue q;

		q.submit([&](sycl::handler& h) {
			auto popu = pop_buf.get_access<sycl::access::mode::read_write>(h);
			auto a = a_buf.get_access<sycl::access::mode::read_write>(h);
			auto b = b_buf.get_access<sycl::access::mode::read_write>(h);
			auto am = am_buf.get_access<sycl::access::mode::read_write>(h);
			auto bm = bm_buf.get_access<sycl::access::mode::read_write>(h);
			auto cleaner = clear_buf.get_access<sycl::access::mode::read>(h);

			h.parallel_for(sycl::range<1>{N}, [=](sycl::item<1>index) {
				// 生成0~1之间的随机小数，如果小于交配概率就进行交配
				//利用oneapi的联合分布方法生成随机浮点数
				oneapi::dpl::minstd_rand engine(index * 100, index.get_linear_id());
				//范围在0到1之间
				oneapi::dpl::uniform_real_distribution<float>distr(0, 1);

				//使用单点交叉，交叉点为随机一个点
				//利用oneapi的联合分布方法生成随机整数
				oneapi::dpl::minstd_rand iengine(index * 95, index.get_linear_id());
				//范围在0到1之间
				oneapi::dpl::uniform_int_distribution<int>cdistr(0, CITY_NUM);

				//判断能否交叉用的随机数
				auto random = distr(engine);

				if ((random < PC)&&(index+1<N)) {
					//先把将要交配的两个个体赋值给a向量和b向量
					//需要处理index+1超出N的问题
					for (int i = 0; i < CITY_NUM; i++) {
						a[i] = popu[index*CITY_NUM+i];
						b[i] = popu[(index * CITY_NUM+1) + i];
					}

					
					//进行交叉
					//随机交叉点位
					auto point = cdistr(iengine);
					//第i个个体的右半边和第i+1个个体的左半边交换
					for (int i = 0; i <= point && (point + i < CITY_NUM); i++) {
						int temp = a[point + i];
						a[point + i] = b[i];
						b[i] = temp;
					}

					//进行去重

					//首先初始化去重用的向量
					for (int i = 0; i < CITY_NUM; i++) {
						am[i] = cleaner[i];
						bm[i] = cleaner[i];
					}
					//判断有无重复元素并进行替换
					for (int i = 0; i < CITY_NUM; i++) {
						if (am[a[i]] == 0)am[a[i]]++;
						else if (am[a[i]] != 0) {
							for (int j = 0; j < CITY_NUM; j++) {
								if (am[j] == 0) {
									a[i] = j;
									am[j]++;
									break;
								}
							}
						}
						if (bm[b[i]] == 0)bm[b[i]]++;
						else if (bm[b[i]] != 0) {
							for (int j = 0; j < CITY_NUM; j++) {
								if (bm[j] == 0) {
									b[i] = j;
									bm[j]++;
									break;
								}
							}
						}
					}

					//将交叉完成后的个体替换掉种群中原来的个体
					for (int i = 0; i < CITY_NUM; i++) {
						popu[index * CITY_NUM + i]=a[i];
						popu[(index * CITY_NUM + 1) + i]=b[i];
					}

				}
				});

			});
		q.wait();


		//for (int i = 0; i + 1 < N; i++) {
		//	//生成0~1之间的三位随机小数，如果小于交配概率就进行交配
		//	float random = rand() % (1000) / (float)(1000);
		//	if (random < PC) {
		//		//使用单点交叉，交叉点为随机一个点
		//		int point = rand() % (CITY_NUM);
		//		vector<int>a = vector<int>(population[i]);
		//		vector<int>b = vector<int>(population[i + 1]);
		//		//第i个个体的右半边和第i+1个个体的左半边交换
		//		for (int i = 0; i <= point && (point + i < CITY_NUM); i++) {
		//			int temp = a[point + i];
		//			a[point + i] = b[i];
		//			b[i] = temp;
		//		}
		//		//去除掉重复元素
		//		vector<int>avec(CITY_NUM, 0);
		//		vector<int>bvec(CITY_NUM, 0);
		//		sycl::buffer am_buf(avec);
		//		sycl::buffer bm_buf(bvec);
		//		sycl::buffer a_buf(a);
		//		sycl::buffer b_buf(b);

		//		sycl::queue{}.submit([&](sycl::handler& h) {
		//			sycl::accessor am(am_buf, h, sycl::read_write);
		//			sycl::accessor bm(bm_buf, h, sycl::read_write);
		//			sycl::accessor _a(a_buf, h, sycl::read_write);
		//			sycl::accessor _b(b_buf, h, sycl::read_write);

		//			h.parallel_for(CITY_NUM, [=](sycl::item<1>idx) {
		//				//利用oneapi的联合分布方法生成随机数
		//				oneapi::dpl::minstd_rand engine_1(idx * 100, idx.get_linear_id());
		//				//范围在0到100之间
		//				oneapi::dpl::uniform_int_distribution<int>distr_1(0, CITY_NUM);
		//				//利用oneapi的联合分布方法生成随机数
		//				oneapi::dpl::minstd_rand engine_2(idx * 120, idx.get_linear_id());
		//				//范围在0到CITY_NUM之间
		//				oneapi::dpl::uniform_int_distribution<int>distr_2(0, CITY_NUM);
		//				auto rd_1 = distr_1(engine_1);
		//				auto rd_2 = distr_2(engine_2);
		//				if (am[_a[idx]] == 0)am[_a[idx]]++;
		//				else if (am[_a[idx]] != 0) {
		//					;
		//					while (am[rd_1] != 0) {
		//						rd_1 = distr_1(engine_1);
		//					}
		//					_a[idx] = rd_1;
		//					am[rd_1]++;
		//				}
		//				if (bm[_b[idx]] == 0)bm[_b[idx]]++;
		//				else if (bm[_b[idx]] != 0) {
		//					while (bm[rd_2] != 0) {
		//						rd_2 = distr_2(engine_2);
		//					}
		//					_b[idx] = rd_2;
		//					bm[rd_2]++;
		//				}
		//				});
		//			});

		//		sycl::host_accessor result{ a_buf };
		//		sycl::host_accessor _result{ b_buf };

		//		//把交配完的个体保存到种群里
		//		population[i] = vector<int>(a);
		//		population[i + 1] = vector<int>(b);
		/*	}
		}*/
		cout << "交叉成功。" << endl;

	}

	//进行变异
	void mutate() {
		cout << "开始变异。" << endl;
		//对于每个个体的每个基因随机生成0~1之间的随机数，如果小于PM，
		//那就随机地交换这一个基因和另一个基因
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
		cout << "变异成功。" << endl;

	}

	//重新计算评估值并更新最优方案
	void get_eval() {
		cout << "开始更新评估值和最优解。" << endl;
		eval.clear();
		eval.clear();
		eval = vector<float>(N);
		evaluate();

		float cur_best = 0.0;
		vector<int>cur_sol(CITY_NUM);
		//把每个个体的评估值添加到eval中

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
		cout << "更新评估值和最优解成功。" << endl;

		//输出一次迭代的最小花费
		cout << "本轮得到的最小花费是" << (1.0)/cur_best << "。" << endl;

		ofs << (1.0) / cur_best << ','<<endl;
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
		
		//获取开始时间
		long long start = clock();

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
		//获取结束时间
		long long end = clock();
		//算法执行的时间，单位是毫秒
		int duration = (end - start) * 1000 / CLOCKS_PER_SEC;

		//输出持续时间，最小花费以及最佳路径到文件
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
		cout << "全局最优解为:" << endl;
		show_best();
	}
};
int main() {
	TSP tsp;
	tsp.run();
}