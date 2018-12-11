#pragma once
#include "Op.h"

class BinaryOp : public Op<2> {
public:
	BinaryOp(const Node& left, const Node& right, const std::string& class_name, Graph& graph) : Op({ {&left, &right} }, class_name, graph) {}
private:
	virtual void binaryOp(const Tensor& left, const Tensor& right, Tensor& out) const = 0;
	void op(const std::array<Tensor, 2>& in, Tensor& out) const override {
		CHECK_EQ(in[0].dataType(), in[1].dataType()) 
			<< "binary operands need to use the same data type: "
			<< in[0].dataType() << " vs " << in[1].dataType();
		binaryOp(in[0], in[1], out);
	}
};


//#pragma once
//#include "Node.h"
//#include "logging.h"
//#include "Tensor.h"
//
//class BinaryOp : public Node {
//public:
//	BinaryOp(const Node& left, const Node& right, const std::string& class_name, Graph& graph) : Node(class_name, graph) {
//		children_.push_back(&left);
//		children_.push_back(&right);
//	}
//	void eval(Tensor& out) const override {
//		Tensor a,b;
//		children_[0]->eval(a);
//		children_[1]->eval(b);
//		CHECK_EQ(a.dataType(), b.dataType()) 
//			<< "binary operands need to use the same data type: "
//			<< a.dataType() << " vs " << b.dataType();
//		binaryOp(a,b,out);
//	}
//	void eval(std::unordered_map<int, Tensor>& nodeTensorMap, Tensor& out) const override {
//
//		Tensor in[2];
//		for (int i = 0; i < 2; i++) {
//			auto it = nodeTensorMap.find(children_[i]->getId());
//			if (it == nodeTensorMap.end()) {
//				children_[i]->eval(nodeTensorMap, in[i]);
//				nodeTensorMap[children_[i]->getId()] = in[i];
//			}
//			else {
//				in[i] = it->second;
//			}
//		}
//		CHECK_EQ(in[0].dataType(), in[1].dataType()) 
//			<< "binary operands need to use the same data type: "
//			<< in[0].dataType() << " vs " << in[1].dataType();
//		binaryOp(in[0],in[1],out);
//	}
//private:
//	virtual void binaryOp(const Tensor& left, const Tensor& right, Tensor& out) const = 0;
//};
