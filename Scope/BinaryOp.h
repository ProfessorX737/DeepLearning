#pragma once
#include "Node.h"
#include "logging.h"
#include "Tensor.h"


class BinaryOp : public Node {
public:
	BinaryOp(Node* left, Node* right, const std::string& class_name) : Node(class_name) {
		children_.push_back(left);
		children_.push_back(right);
	}
	void eval(Tensor& out) override {
		Tensor a,b;
		children_[0]->eval(a);
		children_[1]->eval(b);
		CHECK_EQ(a.dataType(), b.dataType()) 
			<< "Tensor a & b need to use the same data type: "
			<< a.dataType() << " vs " << b.dataType();
		binaryOp(a,b,out);
	}
private:
	virtual void binaryOp(Tensor& left,Tensor& right, Tensor& out) = 0;
};
