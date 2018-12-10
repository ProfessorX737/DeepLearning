#pragma once
#include "Node.h"
#include "Tensor.h"
#include "Initializer.h"

class Variable : public Node {
public:
	Variable(Graph& graph, const Tensor::dim_init_list& shape, DataType dt);
	Variable(Graph& graph, const TensorShape& shape, DataType dt);
	template<typename T>
	void init(const Initializer<T>& i);
	template<typename T>
	void init(const std::initializer_list<T>& list) { t_.fill<T>(list); }
	void eval(Tensor& out) const { out = t_; };
	Tensor tensor();
private:
	Tensor t_;
};

template<typename T>
inline void Variable::init(const Initializer<T>& i)
{
	i.init(t_.data<T>(), t_.numElements());
}
