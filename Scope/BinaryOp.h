#pragma once
#include "Node.h"
#include "logging.h"
#include "Tensor.h"


class BinaryOp : public Node {
public:
	BinaryOp(const Node& left, const Node& right, const std::string& class_name) : Node(class_name) {
		children_.push_back(&left);
		children_.push_back(&right);
	}
	void eval(Tensor& out) const override {
		Tensor a,b;
		children_[0]->eval(a);
		children_[1]->eval(b);
		CHECK_EQ(a.dataType(), b.dataType()) 
			<< "binary operands need to use the same data type: "
			<< a.dataType() << " vs " << b.dataType();
		binaryOp(a,b,out);
	}
private:
	virtual void binaryOp(const Tensor& left, const Tensor& right, Tensor& out) const = 0;
};
