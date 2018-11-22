#pragma once

#include "Initilizer.h"
using namespace Eigen;

template<typename T = float>
class RandomNormal : public Initializer<T> {
private:
	T * m_data;
	float m_mean;
	float m_stddev;
public:
	RandomNormal(float mean = 0.0, float stddev = 0.1) : m_mean(mean), m_stddev(stddev) {}
	~RandomNormal() { delete m_data; }
	T* data(int size);
};

template<typename T>
T* RandomNormal<T>::data(int size) {
	std::default_random_engine generator;
	std::normal_distribution<T> distribution(m_mean, m_stddev);
	m_data = new T[size];
	for (int i = 0; i < size; i++) {
		m_data[i] = distribution(generator);
	}
	return m_data;
}
