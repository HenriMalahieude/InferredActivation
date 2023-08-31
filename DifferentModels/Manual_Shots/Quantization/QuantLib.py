import tensorflow_model_optimization as tfmot

#Copied from https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/quantization/keras/QuantizeConfig
"""class CustomQuantConf(tfmot.QuantizeConfig):
    def __init__(self, number_of_bits):
        self.n_bits = number_of_bits

    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, tfmot.quantizers.LastValueQuantizer(self.n_bits))]
    
    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, tfmot.quantizers.MovingAverageQuantizer(self.n_bits))]
    
    def set_quantize_weights(self, layer, quantize_weights):
        layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        layer.activation = quantize_activations[0]

    def get_output_quantizers(self, layer):
        return []
    
    def get_config(self):
        return {}"""
    
#Copied from tensorflow_model_optimization/python/core/quantization/keras/default_8bit/default_8bit_quantize_registry.py from the github
class DefaultCustomQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
  """QuantizeConfig for non recurrent Keras layers."""

  def __init__(self, weight_attrs, activation_attrs, quantize_output, n_bits=8):
    self.weight_attrs = weight_attrs
    self.activation_attrs = activation_attrs
    self.quantize_output = quantize_output

    # TODO(pulkitb): For some layers such as Conv2D, per_axis should be True.
    # Add mapping for which layers support per_axis.
    self.weight_quantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer(
        num_bits=n_bits, per_axis=False, symmetric=True, narrow_range=True)
    self.activation_quantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
        num_bits=n_bits, per_axis=False, symmetric=False, narrow_range=False)

  def get_weights_and_quantizers(self, layer):
    return [(getattr(layer, weight_attr), self.weight_quantizer)
            for weight_attr in self.weight_attrs]

  def get_activations_and_quantizers(self, layer):
    return [(getattr(layer, activation_attr), self.activation_quantizer)
            for activation_attr in self.activation_attrs]

  def set_quantize_weights(self, layer, quantize_weights):
    if len(self.weight_attrs) != len(quantize_weights):
      raise ValueError(
          '`set_quantize_weights` called on layer {} with {} '
          'weight parameters, but layer expects {} values.'.format(
              layer.name, len(quantize_weights), len(self.weight_attrs)))

    for weight_attr, weight in zip(self.weight_attrs, quantize_weights):
      current_weight = getattr(layer, weight_attr)
      if current_weight.shape != weight.shape:
        raise ValueError('Existing layer weight shape {} is incompatible with'
                         'provided weight shape {}'.format(
                             current_weight.shape, weight.shape))

      setattr(layer, weight_attr, weight)

  def set_quantize_activations(self, layer, quantize_activations):
    if len(self.activation_attrs) != len(quantize_activations):
      raise ValueError(
          '`set_quantize_activations` called on layer {} with {} '
          'activation parameters, but layer expects {} values.'.format(
              layer.name, len(quantize_activations),
              len(self.activation_attrs)))

    for activation_attr, activation in \
        zip(self.activation_attrs, quantize_activations):
      setattr(layer, activation_attr, activation)

  def get_output_quantizers(self, layer):
    if self.quantize_output:
      return [self.activation_quantizer]
    return []

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    return {
        'weight_attrs': self.weight_attrs,
        'activation_attrs': self.activation_attrs,
        'quantize_output': self.quantize_output
    }

  def __eq__(self, other):
    if not isinstance(other, DefaultCustomQuantizeConfig):
      return False

    return (self.weight_attrs == other.weight_attrs and
            self.activation_attrs == self.activation_attrs and
            self.weight_quantizer == other.weight_quantizer and
            self.activation_quantizer == other.activation_quantizer and
            self.quantize_output == other.quantize_output)

  def __ne__(self, other):
    return not self.__eq__(other)