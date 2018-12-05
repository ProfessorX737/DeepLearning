#pragma once
#include "Node.h"

class Tensor;

class BinaryOp : public Node {
public:
	BinaryOp(Node* left, Node* right, const std::string& class_name) : Node(class_name) {
		children_.push_back(left);
		children_.push_back(right);
	}
	Tensor eval() override {
		return binaryOp(children_[0]->eval(), children_[1]->eval());
	}
private:
	virtual Tensor binaryOp(const Tensor& left, const Tensor& right) = 0;
};
