#pragma once
#include "Node.h"
#include "Tensor.h"

template<typename T>
class Initializer;

template<typename T>
class VariableOp : public Node {
public:
	VariableOp(Graph& graph, const Tensor::dim_init_list& shape, DataType dt) 
		: Node(graph, "Variable") { t_.init(shape, dt); }
	VariableOp(Graph& graph, const TensorShape& shape, DataType dt) 
		: Node(graph, "Variable") { t_.init(shape, dt); }
	void init(const Initializer<T>& i) { i.init(t_.data<T>(), t_.numElements()); }
	void init(const std::initializer_list<T>& list) { t_.fill<T>(list); }
	void eval(Tensor& out) const { out = t_; };
	DataType dataType() const override { return t_.dataType(); }
	Tensor tensor() {
		Tensor t;
		t = t_;
		return std::move(t);
	}
	bool deriv(Tensor& out) const {
		out = Tensor(t_.shape(), t_.dataType());
		memset(out.data<T>(),)
	}
private:
	Tensor t_;
};

template<typename T>
std::shared_ptr<VariableOp<T>> Variable(Graph& graph, const Tensor::dim_init_list& shape) {
	return std::make_shared<VariableOp<T>>(graph,shape, DataTypeToEnum<T>::v());
}

template<typename T>
std::shared_ptr<VariableOp<T>> Variable(Graph& graph, const TensorShape& shape) {
	return std::make_shared<VariableOp<T>>(graph,shape, DataTypeToEnum<T>::v());
}
