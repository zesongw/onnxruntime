// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/common.h>

#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"
#include "core/providers/webnn/builders/helper.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class BinaryOpBuilder : public BaseOpBuilder {
  // Add operator related
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related
 private:
  int GetMinSupportedOpSet(const Node& node) const override;
};

// Add operator related

Status BinaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                              const logging::Logger& /* logger */) const {
  const auto& op_type(node.OpType());

  ::wnn::Operand input0 = model_builder.GetOperand(node.InputDefs()[0]->Name());
  ::wnn::Operand input1 = model_builder.GetOperand(node.InputDefs()[1]->Name());
  ::wnn::Operand output;
  if (op_type == "Add") {
    output = model_builder.GetBuilder().Add(input0, input1);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "BinaryOpBuilder::AddToModelBuilderImpl, unknown op: ", op_type);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related

int BinaryOpBuilder::GetMinSupportedOpSet(const Node& /* node */) const {
  // Add/Sub/Mul/Div opset 6- has broadcast attributes we do not support now
  return 7;
}

void CreateBinaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<BinaryOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
