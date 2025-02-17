// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Graph} from '../../../graph';
import {OperatorImplementation, OperatorInitialization} from '../../../operators';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';

export const unsqueeze: OperatorImplementation<number[]> =
    (inferenceHandler: WebGLInferenceHandler, inputs: Tensor[], axes: number[]): Tensor[] => {
      validateInputs(inputs);
      const outputShape = ShapeUtil.unsqueezeShape(inputs[0].dims, axes);
      const output = inferenceHandler.reshapeUnpacked(inputs[0], outputShape);
      return [output];
    };

export const parseUnsqueezeAttributes: OperatorInitialization<number[]> = (node: Graph.Node): number[] =>
    node.attributes.getInts('axes');

const validateInputs = (inputs: Tensor[]): void => {
  if (!inputs || inputs.length !== 1) {
    throw new Error('Unsqueeze requires 1 input.');
  }

  if (inputs[0].type === 'string') {
    throw new Error('invalid input tensor types.');
  }
};