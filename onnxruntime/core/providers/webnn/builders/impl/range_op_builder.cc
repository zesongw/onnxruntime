// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class RangeOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

// Add operator related.

void RangeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // Skip all initializers.
  for (size_t i = 0; i < node.InputDefs().size(); i++) {
    model_builder.AddInitializerToSkip(node.InputDefs()[i]->Name());
  }
}

Status RangeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                             const Node& node,
                                             const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  // Copy the data from the start/limit/delta initializers.
  float start;
  float limit;
  float delta;
  const auto CopyInputData = [&input_defs, &model_builder](size_t input_idx, float& data) {
    std::string input_name;
    if (input_idx >= input_defs.size())
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid input index", input_idx, input_defs.size());
    input_name = input_defs[input_idx]->Name();
    const auto& initializers(model_builder.GetInitializerTensors());
    const auto& tensor = *initializers.at(input_name);
    std::vector<uint8_t> unpacked_tensor;
    auto status = onnxruntime::utils::UnpackInitializerData(tensor, unpacked_tensor);
    if (!status.IsOK())
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Error while unpacking");

    auto data_type = tensor.data_type();
    switch (data_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        data = *reinterpret_cast<const float*>(unpacked_tensor.data());
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
        data = static_cast<float>(*reinterpret_cast<const int32_t*>(unpacked_tensor.data()));
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:
        data = static_cast<float>(*reinterpret_cast<const int64_t*>(unpacked_tensor.data()));
        break;
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported data type: ", data_type);
        break;
    }
    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(CopyInputData(0, start));
  ORT_RETURN_IF_ERROR(CopyInputData(1, limit));
  ORT_RETURN_IF_ERROR(CopyInputData(2, delta));

  //  1-D tensor containing generated range of values.
  std::vector<int32_t> shape{std::max(0, static_cast<int32_t>(ceil((1.0 * (limit - start)) / delta)))};

  emscripten::val options = emscripten::val::object();
  options.set("start", start);
  options.set("delta", delta);

  // WebNN has an issue passing empty inputs in this single Op. But it shouldn't matter in the real models.
  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>(
      "fillSequence", emscripten::val("float32"), emscripten::val::array(shape), options);

  // Cast to the same type as the inputs.
  const auto& initializers(model_builder.GetInitializerTensors());
  const auto& tensor_type = initializers.at(input_defs[0]->Name())->data_type();
  std::string operand_type;
  ORT_RETURN_IF_ERROR(GetWebNNType(tensor_type, operand_type));
  output = model_builder.GetBuilder().call<emscripten::val>(
      "cast", output, emscripten::val(operand_type));
  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

bool RangeOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                       const Node& node,
                                       const WebnnDeviceType device_type,
                                       const logging::Logger& logger) const {
  const auto& name = node.Name();
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (input_defs.size() != 3) {
    LOGS(logger, VERBOSE) << op_type << " [" << name << "] need start limit and delta.";
    return false;
  }

  const auto& start_name = input_defs[0]->Name();
  const auto& limit_name = input_defs[1]->Name();
  const auto& delta_name = input_defs[2]->Name();
  if (!Contains(initializers, start_name) ||
      !Contains(initializers, limit_name) ||
      !Contains(initializers, delta_name)) {
    LOGS(logger, VERBOSE) << op_type << " [" << name << "] need inputs as initializer.";
    return false;
  }

  const auto& tensor_type = initializers.at(input_defs[0]->Name())->data_type();
  // WebNN need to cast the output to the same type as the inputs.
  if (!IsSupportedDataType(tensor_type, device_type)) {
    LOGS(logger, VERBOSE) << op_type << " [" << name << "] data type [" << tensor_type << "] is not supported.";
  }
  return true;
}  // namespace webnn

void CreateRangeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<RangeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
