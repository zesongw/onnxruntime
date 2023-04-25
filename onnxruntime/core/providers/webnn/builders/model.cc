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
#ifdef ENABLE_WEBASSEMBLY_THREADS
    wnn_inputs_[name].call<void>("set", view);
#else
    wnn_inputs_.set(name, view);
#endif
  }

#ifdef ENABLE_WEBASSEMBLY_THREADS
  // This vector uses for recording output buffers from WebNN graph compution when WebAssembly
  // multi-threads is enabled, since WebNN API only accepts non-shared ArrayBufferView,
  // https://www.w3.org/TR/webnn/#typedefdef-mlnamedarraybufferviews
  // and at this time the 'view' defined by Emscripten is shared ArrayBufferView, the memory
  // address is different from the non-shard one, additional memory copy is requred here.
  InlinedHashMap<std::string, emscripten::val> val_vec;
#endif
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
#ifdef ENABLE_WEBASSEMBLY_THREADS
    val_vec.insert({name, view});
#else
    wnn_outputs_.set(name, view);
#endif
  }
  wnn_context_.call<emscripten::val>("computeSync", wnn_graph_, wnn_inputs_, wnn_outputs_);

#ifdef ENABLE_WEBASSEMBLY_THREADS
  // Set the JS buffer back to the output tensor.
  for (const auto& output : outputs) {
    const std::string& name = output.first;
    emscripten::val view = val_vec.at(name);
    view.call<void>("set", wnn_outputs_[name]);
  }
#endif
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

// Pre-allocate the input and output tensors for the WebNN graph.
void Model::SetWnnInputOutput() {
  for (const auto& input : inputs_) {
    const auto& input_info = input_output_info_.at(input);
    const auto input_shape = input_info.shape;
    const auto num_elements = SafeInt<size_t>(Product(input_shape));
    wnn_inputs_.set(input,
                    emscripten::val::global("Float32Array").new_(static_cast<const int>(num_elements)));
  }
  for (const auto& output : outputs_) {
    const auto& output_info = input_output_info_.at(output);
    const auto output_shape = output_info.shape;
    const auto num_elements = SafeInt<size_t>(Product(output_shape));
    wnn_outputs_.set(output,
                     emscripten::val::global("Float32Array").new_(static_cast<const int>(num_elements)));
  }
}

size_t Model::GetMappedInputIdx(const std::string& name) const {
  return input_map_.at(name);
}

size_t Model::GetMappedOutputIdx(const std::string& name) const {
  return output_map_.at(name);
}

}  // namespace webnn
}  // namespace onnxruntime
