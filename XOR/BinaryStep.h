#pragma once
#include "Activation.h"

template<typename T = float>
class BinaryStep : public Activation<T> {
private:
	typedef Eigen::Matrix<T, Dynamic, Dynamic> Matrix;
	static T step(T val) {
		return (val > 0) ? 1.0 : 0.0;
	}
public:
	BinaryStep() {}
	~BinaryStep() {}
	Matrix apply(const Matrix& m) {
		return m.unaryExpr(std::ref(step));
	}
};
