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
        one.init({1,1},dt_);
        one.fill<T>({1});
        minimize_->collectPaths(paths_);
    }
    ~OptimizerOp() {}
    
    // does not change 'out' arg
    void eval(std::unordered_map<int,Tensor>& nodeTensorMap, Tensor& out) const override {
        minimize_->eval(nodeTensorMap,out);
        std::vector<Tensor> gradients;
        for(int i = 0; i < paths_.size(); i++) {
            Tensor dx;
            dx.init<T>({1});
            minimize_->evalDeriv(dx,nodeTensorMap,paths_[i],0);
            gradients.push_back(dx);
        }
        
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
    NodePtr minimize_;
    Tensor one;
    DataType dt_;
};

inline NodePtr Optimizer(Graph& graph, NodePtr minimize) {
    NodePtr ret;
    NUMBER_TYPE_CASES(minimize->dataType(),ret = std::make_shared<OptimizerOp<T>>(graph,minimize));
    return ret;
}