#pragma once

#include <Eigen/Dense>

template<typename T>
class Activation {
private:
	//private T derivative(T) {}
	typedef Eigen::Matrix<T, Dynamic, Dynamic> Matrix;
public:
	//typedef T(Activation::*function)(T);
	//virtual function getActivationFn() = 0;
	// virtual function getDerivativeFn() {
	//	 return activation;
	//}
	virtual Matrix apply(const Matrix& m) = 0;
};
