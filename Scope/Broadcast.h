//
//  broadcast.h
//  DeepLearning
//
//  Created by Xavier Poon on 12/01/2019.
//  Copyright © 2019 CreativityInk. All rights reserved.
//

#ifndef BROADCAST_H
#define BROADCAST_H

#include "Tensor.h"

struct BCast {
    //
    // returns true if shape 'from' can be broadcasted to same shape as 'to'
    static bool compatible(const TensorShape& from, const TensorShape& to);
    
    // if MAX_DIMS is greater than num dims of 'from' and 'to' tensors, the multDims array has 1's padded to the right
    // returns true if shape 'from' can be broadcasted to same shape as 'to'
    static bool multDimsPadRight(Eigen::array<int,Tensor::MAX_DIMS>& multDims, const TensorShape& from, const TensorShape& to);
    
    // if MAX_DIMS is greater than num dims of 'from' and 'to' tensors, the multDims array has 1's padded to the left
    // returns true if shape 'from' can be broadcasted to same shape as 'to'
    static bool multDimsPadLeft(Eigen::array<int,Tensor::MAX_DIMS>& multDims, const TensorShape& from, const TensorShape& to);
    
    // same as multDimsPadLeft
    // returns true if shape 'from' can be broadcasted to same shape as 'to'
    static bool multDims(Eigen::array<int,Tensor::MAX_DIMS>& multDims, const TensorShape& from, const TensorShape& to);
    
    static bool reduceDims(std::vector<int>& reductionIndicies, const TensorShape& from, const TensorShape& to);
};

#endif /* BROADCAST_H */
