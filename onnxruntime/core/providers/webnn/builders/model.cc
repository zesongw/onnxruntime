// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <vector>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/common.h"
#include "core/providers/webnn/builders/helper.h"
#include "model.h"

namespace onnxruntime {
namespace webnn {

Model::Model(const emscripten::val& context, const emscripten::val& graph, const logging::Logger& logger)
    : wnn_context_(context),
      wnn_graph_(graph),
      logger_(logger) {}

Model::~Model() {}

Status Model::Predict(const InlinedHashMap<std::string, OnnxTensorData>& inputs,
                      const InlinedHashMap<std::string, OnnxTensorData>& outputs) {
  for (const auto& input : inputs) {
    const std::string& name = input.first;
    const struct OnnxTensorData tensor = input.second;
    if (tensor.tensor_info.data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The input of graph has unsupported type, name: ",
                             name, " type: ", tensor.tensor_info.data_type);
    }
    auto num_elements = SafeInt<size_t>(Product(tensor.tensor_info.shape));
    emscripten::val view{emscripten::typed_memory_view(num_elements, static_cast<const float*>(tensor.buffer))};
    // WebNN API only accepts non-shared ArrayBufferView.
    // https://webmachinelearning.github.io/webnn/#typedefdef-mlnamedarraybufferviews
    emscripten::val SharedArrayBuffer = emscripten::val::global("SharedArrayBuffer");
    if (SharedArrayBuffer.as<bool>()) {
      emscripten::val non_shared_array = emscripten::val::global("Float32Array").new_(static_cast<uint32_t>(num_elements));
      non_shared_array.call<void>("set", view);
      wnn_inputs_.set(name, non_shared_array);
    } else {
      wnn_inputs_.set(name, view);
    }
  }

  emscripten::val SharedArrayBuffer = emscripten::val::global("SharedArrayBuffer");
  InlinedHashMap<std::string, emscripten::val> val_vec;
  for (const auto& output : outputs) {
    const std::string& name = output.first;
    const struct OnnxTensorData tensor = output.second;
    if (tensor.tensor_info.data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The input of graph has unsupported type, name: ",
                             name, " type: ", tensor.tensor_info.data_type);
    }
    auto num_elements = SafeInt<size_t>(Product(tensor.tensor_info.shape));
    emscripten::val view{emscripten::typed_memory_view(num_elements, static_cast<const float*>(tensor.buffer))};
    val_vec.insert({name, view});
    emscripten::val non_shared_array = emscripten::val::global("Float32Array").new_(static_cast<uint32_t>(num_elements));
    if (SharedArrayBuffer.as<bool>()) {
      wnn_outputs_.set(name, non_shared_array);
    } else {
      wnn_outputs_.set(name, view);
    }
  }

  // Set the JS buffer back to the output tensor.
  wnn_context_.call<emscripten::val>("computeSync", wnn_graph_, wnn_inputs_, wnn_outputs_);
  if (SharedArrayBuffer.as<bool>()) {
    for (const auto& output : outputs) {
      const std::string& name = output.first;
      emscripten::val view = val_vec.at(name);
      view.call<void>("set", wnn_outputs_[name]);
    }
  }
  return Status::OK();
}

bool Model::IsScalarOutput(const std::string& output_name) const {
  return Contains(scalar_outputs_, output_name);
}

const OnnxTensorInfo& Model::GetInputOutputInfo(const std::string& name) const {
  return input_output_info_.at(name);
}

void Model::SetInputMap(InlinedHashMap<std::string, size_t>&& input_map) {
  input_map_ = std::move(input_map);
}

void Model::SetOutputMap(InlinedHashMap<std::string, size_t>&& output_map) {
  output_map_ = std::move(output_map);
}

size_t Model::GetMappedInputIdx(const std::string& name) const {
  return input_map_.at(name);
}

size_t Model::GetMappedOutputIdx(const std::string& name) const {
  return output_map_.at(name);
}

}  // namespace webnn
}  // namespace onnxruntime
