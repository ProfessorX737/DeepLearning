#pragma once
#include "Activation.h"

template<typename T = float>
class TanhActivation : public Activation<T> {
private:
	typedef Eigen::Matrix<T, Dynamic, Dynamic> Matrix;
public:
	TanhActivation() {}
	~TanhActivation() {}
	Matrix apply(const Matrix& m) {
		return m.array().tanh();
	}
};
