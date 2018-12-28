//
//  Optimizer.h
//  DeepLearning
//
//  Created by Xavier Poon on 26/12/2018.
//  Copyright © 2018 CreativityInk. All rights reserved.
//

#pragma once
#include "Node.h"

template<typename T>
class OptimizerOp : public Node {
public:
    OptimizerOp(Graph& graph, NodePtr minimize) : Node(graph,"Optimizer"), minimize_(minimize) {
        dt_ = DataTypeToEnum<T>::v();
        minimize_->collectPaths(paths_,variables_);
    }
    ~OptimizerOp() {}
    
    // does not change 'out' arg
    void eval(std::unordered_map<int,Tensor>& nodeTensorMap, Tensor& out) const override {
        minimize_->eval(nodeTensorMap,out);
        std::vector<Tensor> gradients;
        //minimize_->evalGradients<T>(nodeTensorMap, paths_, gradients);
        
        // update all variables using the gradient dx ...
    }
    
    // for testing the derivative
    void evalDeriv(Tensor& dx) {
        dx.init<T>({1});
        std::unordered_map<int,Tensor> nodeTensorMap;
        minimize_->eval(nodeTensorMap);
        minimize_->evalDeriv(dx,nodeTensorMap,paths_[1],0);
    }
    
private:
    std::vector<std::vector<int>> paths_;
    std::vector<Tensor> variables_;
    NodePtr minimize_;
    DataType dt_;
};

inline NodePtr Optimizer(Graph& graph, NodePtr minimize) {
    NodePtr ret;
    NUMBER_TYPE_CASES(minimize->dataType(),ret = std::make_shared<OptimizerOp<T>>(graph,minimize));
    return ret;
}