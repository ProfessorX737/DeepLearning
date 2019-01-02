//
//  Sigmoid.h
//  DeepLearning
//
//  Created by Xavier Poon on 2/01/2019.
//  Copyright © 2019 CreativityInk. All rights reserved.
//

#pragma once
#include "UnaryOp.h"

template<typename T>
class SigmoidOp : public UnaryOp {
public:
    SigmoidOp(Graph& graph, NodePtr operand) : UnaryOp(graph,operand,"Sigmoid") {}
};