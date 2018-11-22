#pragma once

#include <Eigen/Dense>
#include "Initilizer.h"
#include "Activation.h"

template<int num_inputs, int num_outputs, typename T = float>
class FullyConnected {
private:
	typedef Eigen::Matrix<T, Dynamic, Dynamic> Matrix;
	typedef Eigen::Matrix<T, 1, Dynamic> Vector;
	Matrix m_w;
	Vector m_b;
	T (*m_activation_fn)(T);
	Activation<T>* m_activation;
public:
	FullyConnected(Initializer<T>* weight_init, Initializer<T>* bias_init, Activation<T>* activation) {
		m_w = Map<Matrix>(weight_init->data(num_inputs*num_outputs),num_inputs, num_outputs);
		m_b = Map<Vector>(bias_init->data(num_outputs), num_outputs);
		m_activation = activation;
	}
	~FullyConnected() {}
	Matrix getWeights() {
		return m_w;
	}
	Vector getBiases() {
		return m_b;
	}
	void applyActivation() {
		m_w = m_activation->apply(m_w);
	}
	Matrix feed(Vector input) {
		return m_activation->apply(input*m_w + m_b);
	}
};
