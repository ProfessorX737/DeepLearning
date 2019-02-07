#include "Node.h"
#include "Tensor.h"
#include "Graph.h"
#include <iostream>

Node::Node(Graph& graph, const std::string& class_name) {
	// set class_id
	auto it = classIdMap.find(class_name);
	if (it != classIdMap.end()) {
		class_id_ = it->second;
	}
	else {
		class_id_ = static_cast<int>(classIdMap.size());
		classIdMap[class_name] = class_id_;
	}
	class_name_ = class_name;
	// set scope_id
	id_ = graph.getUniqueId();
}

DataType Node::dataType() const {
    LOG(PROBLEM) << "dataType() function not implemented in node " << class_name_;
    return DT_INVALID;
}

void Node::evalGradients(std::unordered_map<int,Tensor>& nodeTensorMap, const std::vector<std::vector<int>>& paths,
                         std::vector<Tensor>& inOutGrads, int batchIndex) {
    DCHECK_EQ(inOutGrads.size(),paths.size()) << "outGrads vector must contain same number of tensors"
                                            << " as number of paths";
    for(int i = 0; i < paths.size(); i++) {
        evalDeriv(inOutGrads[i], nodeTensorMap, paths[i], 0, batchIndex);
    }
}

void Node::eval(Tensor& out) {
    std::unordered_map<int,Tensor> nodeTensorMap;
    eval(nodeTensorMap,out);
}

void Node::eval(std::unordered_map<int,Tensor>& nodeTensorMap) {
    Tensor out;
    eval(nodeTensorMap,out);
}


void Node::collectPaths(std::vector<std::vector<int>>& outPaths, std::vector<Tensor>& outVariables) const {
    std::vector<int> curr;
    collectPaths(curr, outPaths, outVariables);
}

void Node::collectPaths(std::vector<int>& curr, std::vector<std::vector<int>>& outPaths,
                        std::vector<Tensor>& outVariables) const {
	for (int i = 0; i < children_.size(); i++) {
		std::vector<int> pathToNext = curr;
		pathToNext.push_back(i);
		children_[i]->collectPaths(pathToNext,outPaths,outVariables);
	}
}
