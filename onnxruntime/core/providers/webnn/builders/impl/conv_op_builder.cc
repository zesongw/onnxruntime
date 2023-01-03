// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"
#include "builder_utils.h"

namespace onnxruntime {
namespace webnn {

class ConvOpBuilder : public BaseOpBuilder {
  // Add operator related
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& /* node */,
                         const logging::Logger& /* logger */) const override;
};

// Helper functions
common::Status SetConvBaseOptions(ModelBuilder& model_builder,
                        const Node& node, emscripten::val& options,
                        const std::vector<int32_t>& strides,
                        const std::vector<int32_t>& dilations,
                        const std::vector<int32_t>& pads,
                        const logging::Logger& logger) {
  NodeAttrHelper helper(node);
  const auto group = helper.Get("group", static_cast<int32_t>(1));
  const auto& input_defs = node.InputDefs();
  const auto& weight_tensor = *model_builder.GetInitializerTensors().at(input_defs[1]->Name());
  const auto& weight_shape = weight_tensor.dims();

  options.set("strides", emscripten::val::array(strides));
  options.set("dilations", emscripten::val::array(dilations));
  options.set("inputLayout", emscripten::val("nchw"));
  options.set("groups", group);
  // Add Padding
  // Usually using autopadding is more efficient than using explicit padding
  // Try to see if we can map explicit padding to auto padding
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
  AutoPadType auto_pad_type;
  ORT_RETURN_IF_ERROR(HandleAutoPad(input_shape, weight_shape[2], weight_shape[3],
                                    helper.Get("pads", std::vector<int64_t>{0, 0, 0, 0}),
                                    helper.Get("strides", std::vector<int64_t>{1, 1}),
                                    helper.Get("dilations", std::vector<int64_t>{1, 1}),
                                    StringToAutoPadType(helper.Get("auto_pad", "NOTSET")),
                                    auto_pad_type));
  if (AutoPadType::SAME_UPPER == auto_pad_type || AutoPadType::SAME_LOWER == auto_pad_type) {
    if (AutoPadType::SAME_LOWER == auto_pad_type) {  // default is SAME_UPPER
      options.set("autoPad", emscripten::val("same-lower"));
    } else {
      options.set("autoPad", emscripten::val("same-upper"));
    }
  } else {
    options.set("padding", emscripten::val::array(pads));
  }

  // Add bias if present
  if (input_defs.size() > 2) {
    options.set("bias", model_builder.GetOperand(input_defs[2]->Name()));
  }
  emscripten::val activation = model_builder.FindActivation(node, *node.OutputDefs()[0]);
  if(emscripten::val::null() != activation) {
    options.set("activation", activation);
  }
  
  return Status::OK();
}

// Add operator related

Status ConvOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                            const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val filter = model_builder.GetOperand(input_defs[1]->Name());
  emscripten::val output = emscripten::val::object();

  NodeAttrHelper helper(node);
  const auto strides = helper.Get("strides", std::vector<int32_t>{1, 1});
  const auto dilations = helper.Get("dilations", std::vector<int32_t>{1, 1});
  const auto pads = helper.Get("pads", std::vector<int32_t>{0, 0, 0, 0});

  if (op_type == "Conv") {
    emscripten::val options = emscripten::val::object();
    ORT_RETURN_IF_ERROR(SetConvBaseOptions(model_builder, node, options, strides, dilations, pads, logger));
    options.set("filterLayout", emscripten::val("oihw"));
    output = model_builder.GetBuilder().call<emscripten::val>("conv2d", input, filter, options);
  } else {
    emscripten::val options = emscripten::val::object();
    ORT_RETURN_IF_ERROR(SetConvBaseOptions(model_builder, node, options, strides, dilations, pads, logger));
    options.set("filterLayout", emscripten::val("iohw"));
    // When the 'output_shape' is specificed, the 'output_padding' values
    // in options.outputPadding are ignored.
    std::vector<int32_t> output_shape;
    std::vector<int32_t> output_padding;
    if (helper.HasAttr("output_shape")) {
      // Default value of 'output_shape' will be ignore as we already check if
      // it's existed
      output_shape = helper.Get("output_shape", std::vector<int32_t>{-1, -1});
      options.set("outputSizes", emscripten::val::array(output_shape));
    } else {
      output_padding = helper.Get("output_padding", std::vector<int32_t>{0, 0});
      options.set("outputPadding", emscripten::val::array(output_padding));
    }

    output = model_builder.GetBuilder().call<emscripten::val>("convTranspose2d", input, filter, options);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related

bool ConvOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                      const logging::Logger& logger) const {
  const auto& name = node.Name();
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();

  const auto& weight_name = input_defs[1]->Name();
  if (Contains(initializers, weight_name)) {
    const auto& tensor = *initializers.at(weight_name);
    if (tensor.dims().size() != 4) {
      LOGS(logger, VERBOSE) << op_type << " [" << name << "] dimension: " << tensor.dims().size()
                            << " Only conv 2d is supported.";
      return false;
    }
  } else {
    LOGS(logger, VERBOSE) << "The weight of " << op_type << " [" << name << "] must be known";
    return false;
  }

  return true;
}

void CreateConvOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.find(op_type) != op_registrations.op_builder_map.cend())
    return;

  static std::vector<std::string> op_types =
      {
          "Conv",
          "ConvTranspose",
      };

  op_registrations.builders.push_back(std::make_unique<ConvOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
