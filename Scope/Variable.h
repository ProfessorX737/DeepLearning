#pragma once
#include "Node.h"
#include "Tensor.h"

template<typename T>
class Initializer;
class VariableOp;

typedef std::shared_ptr<VariableOp> VariablePtr;

VariablePtr Variable(Graph& graph, const Tensor::dim_init_list& shape, DataType dt);
VariablePtr Variable(Graph& graph, const TensorShape& shape, DataType dt);

class VariableOp : public Node {
public:
	VariableOp(Graph& graph, const Tensor::dim_init_list& shape, DataType dt);
	VariableOp(Graph& graph, const TensorShape& shape, DataType dt);
	template<typename T>
	void init(const Initializer<T>& i);
	template<typename T>
	void init(const std::initializer_list<T>& list) { t_.fill<T>(list); }
    void eval(std::unordered_map<int,Tensor>& nodeTensorMap, Tensor& out) const override { out = t_; };
	void collectPaths(std::vector<int>& curr, std::vector<std::vector<int>>& paths) const override;
	//bool deriv(Tensor& out) const;
	DataType dataType() const override { return t_.dataType(); }
	Tensor tensor();
private:
	Tensor t_;
};

template<typename T>
inline void VariableOp::init(const Initializer<T>& i)
{
	i.init(t_.data<T>(), t_.numElements());
}

