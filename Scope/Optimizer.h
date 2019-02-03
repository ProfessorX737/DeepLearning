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
    
	// out tensor will have same evaluated result as the minimize_ node
    void eval(std::unordered_map<int,Tensor>& nodeTensorMap, Tensor& out) override {
        minimize_->eval(nodeTensorMap,out);
        gradients_.clear();
        for(int i = 0; i < paths_.size(); i++) {
            Tensor dx({1,1},dt_);
            dx.fill<T>({1});
            gradients_.push_back(dx);
        }
        minimize_->evalGradients(nodeTensorMap, paths_, gradients_);
        
        // update all variables using the gradient dx ...
		updateVariables(variables_, gradients_);
    }

	virtual void updateVariables(std::vector<Tensor>& variables, std::vector<Tensor>& gradients) = 0;
    
    std::vector<Tensor>& getGradients() {
        return gradients_;
    }
    
private:
    std::vector<std::vector<int>> paths_;
    std::vector<Tensor> variables_;
    std::vector<Tensor> gradients_;
    NodePtr minimize_;
    DataType dt_;
};
