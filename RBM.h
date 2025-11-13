#include <vector>
#include <random>
#include <cassert>
#include <cmath>

class RBM {
	const int n_visible; // 可視変数の数
	const int n_hidden;  // 隠れ変数の数
	std::vector<float> b, c; // パラメータ b, c
	std::vector<std::vector<float>> w; // パラメータ w
	std::vector<float> get_probability_hidden(const std::vector<int> &visible);
	std::vector<float> get_probability_visible(const std::vector<int> &hidden);
	std::vector<int> get_sample(const std::vector<float> &prob);
	
	public:
	RBM(int visible_size, int hidden_size);
	void train(const std::vector<int> &visible0, const float rate=0.01);
	std::vector<int> get_reconstruction(const std::vector<int> &visible);
};

RBM::RBM(int visible_size, int hidden_size) : n_visible(visible_size), n_hidden(hidden_size) {
	
	// bを0で初期化(range-based for文)
	b.resize(n_visible);
	for(auto &ref : b) ref = 0.0;
	
	// cを0で初期化
	c.resize(n_hidden);
	for(auto &ref : c) ref = 0.0;
	
	// 重みを平均0, 標準偏差0.01の正規分布に従う乱数で初期化
	std::random_devide rd;
	std::mt19937 mt(rd());
	std::normal_distribution<float> rand(0, 0.01);
	
	w.resize(n_visible);
	for(auto &ref : w) {
		ref.resize(n_hidden);
		for(auto &ref2 : ref) {
			ref2 = rand(mt);
		}
	}
}

std::vector<float> RBM::get_probability_hidden(const std::vector<int> &visible)
{
	// check input
	assert(visible.size() == static_cast<unsigned>(n_visible));
	
	// prepare variable
	std::vector<float> prob;
	prob.resize(n_hidden);
	
	// calculate probability
	for(int j = 0; j < n_hidden; j++ )
	{
		float lambda = c[j];
		for (int i = 0; i < n_visible; i++)
		{
			lambda += w[i][j] * visible[i];
		}
		prob[j] = exp(lambda) / (1.0 + exp(lambda));
	}
	return prob;
}

std::vector<float> RBM::get_probability_visible(const std::vector<int> &hidden)
{
	// check input
	assert(hidden.size() == static_cast<unsigned>(n_hidden));
	
	// prepare variable
	std::vector<float> prob;
	prob.resize(n_visible);
	
	for(int i = 0; i < n_visible; i++)
	{
		float lambda = b[i];
		for(int  j = 0; j < n_hidden; j++)
		{
			lambda += w[i][j]*hidden[j];
		}
		prob[i] = lambda;
	}
	return prob;
}

std::vector<int> RBM::get_sample(const std::vector<float> &prob)
{
	// prepare variables
	std::vector<int> sample(prob.size());
	std::random_devide rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> rand(0,1.0);
	
	// create sample
	for ( int i=0;i < prob.size(); i++)
	{
		if (rand(mt) <= prob[i]) 
		{
			sample[i] = 1;
		} 
		else
		{
			sample[i] = 0;
		}
	}
	
	return sample;
}

void RBM::train(const std::vector<int>& visible0, const float rate)
{
	// サンプルの計算
	auto prob0 = get_probability_hidden(visible0);
	auto hidden0 = get_sample(prob0);
	auto visible1 = get_sample(get_probability_visible(hidden0));
	auto prob1 = get_probability_hidden(visible1);
	
	// wの更新
	for(int i = 0; i < n_visible; i++)
	{
		for(int j = 0; j < n_visible; j++)
		{
			w[i][j] += rate * (visible0[i]*hidden0[j] - visible1[i]*prob1[j]);
		}
	}
	
	// bの更新
	for(int i = 0; i < n_visible; i++)
	{
		b[i] += rate*(visible0[i] - visible1[i]);
	}
	
	// cの更新
	for( int j = 0; j < n_hidden; j++)
	{
		c[j] += rate * (hidden0[j] - prob1[j]);
	}
}

std::vector<int> RBM::get_reconstruction(const std::vector<int> &visible)
{
	auto hidden = get_sample(get_probability_hidden(visible));
	return get_sample(get_probability_visible(hidden));
}






