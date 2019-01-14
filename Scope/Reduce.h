//
//  Reduce.h
//  DeepLearning
//
//  Created by Xavier Poon on 8/01/2019.
//  Copyright © 2019 CreativityInk. All rights reserved.
//

#pragma once
#include "UnaryOp.h"
#include "Broadcast.h"

template<typename T, typename Reducer, int...Reduce>
class ReduceOp : public UnaryOp {
public:
    static constexpr int numReduceDims = sizeof...(Reduce);
    static constexpr int numResultDims = Tensor::MAX_DIMS - numReduceDims;
    ReduceOp(Graph& graph, NodePtr operand) : UnaryOp(graph,operand,"Reduce") {
        reduceDims_ = { Reduce... };
    }
    
    void unaryOp(const Tensor& operand, Tensor& out) const override {
        typename TTypes<T,numResultDims>::Tensor tout;
        tout = operand.tensorPadRight<T,Tensor::MAX_DIMS>().reduce(reduceDims_,Reducer());
        TensorShape outShape;
        for(int i = 0; i < tout.NumDimensions; i++) {
            outShape.addDim(tout.dimension(i));
        }
        out.init(outShape,operand.dataType());
        out.tensor<T,numResultDims>() = tout;
    }
    
	void deriv(Tensor& dx, const std::array<Tensor, 1>& in, int wrtIdx,
               const std::unordered_map<int,Tensor>& nodeTensorMap) const override {
        if(!dx.hasSameShape(in[0])) {
            Eigen::array<int,Tensor::MAX_DIMS> multDims;
            CHECK(BCast::multDims(multDims, dx.shape(), in[0].shape())) << "derivative cannot be broadcasted to input shape for Reduce Op deriv: " << dx.dimString() << " vs " << in[0].dimString();
            auto oldDx = dx.tensor<T,Tensor::MAX_DIMS>();
            dx.init(in[0].shape(),in[0].dataType());
            dx.tensor<T,Tensor::MAX_DIMS>() = oldDx.broadcast(multDims);
        }
	}
private:
    Eigen::array<int,sizeof...(Reduce)> reduceDims_;
};

template<int...Reduce>
NodePtr ReduceSum(Graph& graph, NodePtr operand) {
    NodePtr ret;
    NUMBER_TYPE_CASES(operand->dataType(), (ret = std::make_shared<ReduceOp<T,Eigen::internal::SumReducer<T>,Reduce...>>(graph, operand)));
    return ret;
}

template<int...Reduce>
NodePtr ReduceMax(Graph& graph, NodePtr operand) {
    NodePtr ret;
    NUMBER_TYPE_CASES(operand->dataType(), (ret = std::make_shared<ReduceOp<T,Eigen::internal::MaxReducer<T>,Reduce...>>(graph, operand)));
    return ret;
}

template<int...Reduce>
NodePtr ReduceMean(Graph& graph, NodePtr operand) {
    NodePtr ret;
    NUMBER_TYPE_CASES(operand->dataType(), (ret = std::make_shared<ReduceOp<T,Eigen::internal::MeanReducer<T>,Reduce...>>(graph, operand)));
    return ret;
}
