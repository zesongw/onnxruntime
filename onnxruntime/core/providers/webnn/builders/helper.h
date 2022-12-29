// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/common/status.h>
#include <core/graph/basic_types.h>

namespace onnxruntime {

class GraphViewer;
class NodeArg;

namespace logging {
class Logger;
}

namespace webnn {

bool GetShape(const NodeArg& node_arg, std::vector<int64_t>& shape, const logging::Logger& logger);

bool IsInputSupported(const NodeArg& node_arg, const std::string& parent_name, const logging::Logger& logger);

// Get a list of groups of supported nodes, each group represents a subgraph supported by WebNN EP
std::vector<std::vector<NodeIndex>> GetSupportedNodes(const GraphViewer& graph_viewer,
                                                      const logging::Logger& logger);

inline std::unordered_map<std::string,std::string> op_map = {
    {"Add","add"},
    {"BatchNormalization","batchNormalization"},
    {"Clip","clamp"},
    {"Conv","conv2d"},
    {"ConvTranspose","convTranspose2d"},
    {"Concat","concat"},
    {"Gemm","gemm"},
    {"MatMul","matmul"},
    {"GRU","gru"},
    {"GlobalAveragePool","averagePool2d"},
    {"GlobalMaxPool","maxPool2d"},
    {"AveragePool","averagePool2d"},
    {"Reshape","reshape"},
    {"Resize","resample2d"},
    {"Transpose","transpose"}
};

}  // namespace webnn
}  // namespace onnxruntime
