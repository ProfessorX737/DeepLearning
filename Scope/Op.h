#pragma once
#include "Node.h"
#include "logging.h"
#include "Tensor.h"
#include <iostream>

template<int NOPERANDS>
struct DerivContext {
    DerivContext(const std::array<Tensor,NOPERANDS>& operands, int wrtIdx,
                 const std::unordered_map<int,Tensor>& nodeTensorMap, int batchIndex)
    : operands(operands), wrtIdx(wrtIdx), nodeTensorMap(nodeTensorMap), batchIndex(batchIndex) {}

    const std::array<Tensor,NOPERANDS>& operands;
    int wrtIdx;
    const std::unordered_map<int,Tensor>& nodeTensorMap;
    int batchIndex;
};

template<typename T,int NOPERANDS>
class Op : public Node {
public:
	Op(Graph& graph, const std::array<NodePtr,NOPERANDS>& operands, const std::string& class_name) : Node(graph, class_name) {
		for (int i = 0; i < NOPERANDS; i++) {
			children_.push_back(std::move(operands[i]));
		}
	}
	void eval(Tensor& out) override {
		std::array<Tensor, NOPERANDS> in;
		for (int i = 0; i < NOPERANDS; i++) {
			children_[i]->eval(in[i]);
		}
		op(in,out);
	}
	void eval(std::unordered_map<int,Tensor>& nodeTensorMap, Tensor& out) override {

		std::array<Tensor, NOPERANDS> in;
		for (int i = 0; i < NOPERANDS; i++) {
//			if (children_[i]->numChildren() == 0) {
//				children_[i]->eval(nodeTensorMap, in[i]);
//			}
//			else {
				auto it = nodeTensorMap.find(children_[i]->getId());
				if (it == nodeTensorMap.end()) {
					children_[i]->eval(nodeTensorMap, in[i]);
					nodeTensorMap[children_[i]->getId()] = in[i];
				}
				else {
					in[i] = it->second;
				}
//			}
		}
		op(in,out);
	}
	DataType dataType() const override {
		return (NOPERANDS == 0) ? DT_INVALID : children_[0]->dataType();
	}

    void evalDeriv(Tensor& dx, std::unordered_map<int,Tensor>& nodeTensorMap,
                           const std::vector<int>& path, const int pathIndex, int batchIndex) const override {
        
        std::array<Tensor, NOPERANDS> in;
        std::array<typename TTypes<T,Tensor::MAX_DIMS>::Tensor, NOPERANDS> in2;
        
		for (int i = 0; i < NOPERANDS; i++) {
//			if (children_[i]->numChildren() == 0 /*&& children_[i]->getClassName().compare("Variable")*/) {
            if(children_[i]->getClassName().compare("Variable") == 0) {
				children_[i]->eval(nodeTensorMap, in[i]);
			}
			else {
				auto it = nodeTensorMap.find(children_[i]->getId());
				if (it == nodeTensorMap.end()) {
                    LOG(FATAL) << "operand for derivative cannot be found in nodeTensorMap";
				}
				else {
                    Tensor t;
                    t.sharedCopyInit<T>(it->second, batchIndex);
					in[i] = t;
				}
			}
		}
        int wrtIdx = path[pathIndex];
        DerivContext<NOPERANDS> ctx(in,wrtIdx,nodeTensorMap,batchIndex);
        //deriv(dx, in, wrtIdx, nodeTensorMap);
        deriv(dx,ctx);
        children_[wrtIdx]->evalDeriv(dx, nodeTensorMap, path, pathIndex+1, batchIndex);
    }

private:
	virtual void op(const std::array<Tensor,NOPERANDS>& in, Tensor& out) const = 0;
//    virtual void deriv(Tensor& dx, const std::array<Tensor, NOPERANDS>& in, int wrtIdx,
//                       const std::unordered_map<int,Tensor>& nodeTensorMap) const = 0;
    virtual void deriv(Tensor& dx, DerivContext<NOPERANDS>& ctx) const = 0;
};
