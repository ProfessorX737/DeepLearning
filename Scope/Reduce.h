//
//  Reduce.h
//  DeepLearning
//
//  Created by Xavier Poon on 8/01/2019.
//  Copyright © 2019 CreativityInk. All rights reserved.
//

#pragma once
#include "UnaryOp.h"

template<typename T, int...Reduce>
class ReduceOp : public UnaryOp {
public:
    static constexpr int numReduceDims = sizeof...(Reduce);
    static constexpr int numResultDims = Tensor::MAX_DIMS - numReduceDims;
    ReduceOp(Graph& graph, NodePtr operand) : UnaryOp(graph,operand,"Reduce") {
        reduceDims_ = { Reduce... };
    }
    
    void unaryOp(const Tensor& operand, Tensor& out) const override {
        typename TTypes<T,numResultDims>::Tensor tout;
        reduce(tout, operand.tensor<T,Tensor::MAX_DIMS>(),reduceDims_);
        TensorShape outShape;
        for(int i = 0; i < tout.NumDimensions; i++) {
            outShape.addDim(tout.dimension(i));
        }
        out.init(outShape,operand.dataType());
        out.tensor<T,numResultDims>() = tout;
    }
    
    virtual void reduce(typename TTypes<T,numResultDims>::Tensor& out, const typename TTypes<T,Tensor::MAX_DIMS>::Tensor& in, const Eigen::array<int,numReduceDims>& reduceDims) const = 0;
    
	void deriv(Tensor& dx, const std::array<Tensor, 1>& in, int wrtIdx,
               const std::unordered_map<int,Tensor>& nodeTensorMap) const override {
        
	}
private:
    Eigen::array<int,sizeof...(Reduce)> reduceDims_;
};

template<typename T, int...Reduce>
class ReduceSumOp : public ReduceOp<T,Reduce...> {
    static constexpr int numReduceDims = sizeof...(Reduce);
    static constexpr int numResultDims = ReduceOp<T,Reduce...>::numResultDims;
public:
    ReduceSumOp(Graph& graph, NodePtr operand) : ReduceOp<T,Reduce...>(graph,operand) {}
    void reduce(typename TTypes<T,numResultDims>::Tensor& out, const typename TTypes<T,Tensor::MAX_DIMS>::Tensor& in, const Eigen::array<int,numReduceDims>& reduceDims) const override {
        out = in.sum(reduceDims);
    }
};

template<int...Reduce>
NodePtr ReduceSum(Graph& graph, NodePtr operand) {
    NodePtr ret;
    NUMBER_TYPE_CASES(operand->dataType(), (ret = std::make_shared<ReduceSumOp<T,Reduce...>>(graph, operand)));
    return ret;
}

template<typename T, int...Reduce>
class ReduceMaxOp : public ReduceOp<T,Reduce...> {
    static constexpr int numReduceDims = sizeof...(Reduce);
    static constexpr int numResultDims = ReduceOp<T,Reduce...>::numResultDims;
public:
    ReduceMaxOp(Graph& graph, NodePtr operand) : ReduceOp<T,Reduce...>(graph,operand) {}
    void reduce(typename TTypes<T,numResultDims>::Tensor& out, const typename TTypes<T,Tensor::MAX_DIMS>::Tensor& in, const Eigen::array<int,numReduceDims>& reduceDims) const override {
        out = in.maximum(reduceDims);
    }
};

template<int...Reduce>
NodePtr ReduceMax(Graph& graph, NodePtr operand) {
    NodePtr ret;
    NUMBER_TYPE_CASES(operand->dataType(), (ret = std::make_shared<ReduceMaxOp<T,Reduce...>>(graph, operand)));
    return ret;
}
