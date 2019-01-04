#pragma once
#include "Node.h"
#include "logging.h"
#include "TensorShape.h"

class PlaceholderOp;

typedef std::shared_ptr<PlaceholderOp> PlaceholderPtr;
PlaceholderPtr Placeholder(Graph& graph, const TensorShape& shape, DataType dt);
	

class PlaceholderOp : public Node {
public:
	PlaceholderOp(Graph& graph, const TensorShape& shape, DataType dt = DT_FLOAT);
	void eval(std::unordered_map<int,Tensor>& nodeTensorMap, Tensor& out) override;
	DataType dataType() const override { return dt_; }
	TensorShape shape() { return shape_; }
private:
	TensorShape shape_;
	DataType dt_;
};
