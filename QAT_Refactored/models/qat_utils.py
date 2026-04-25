import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf_keras

# [MODIFIED] Use the stable public API for QuantizeConfig
class RepVGGQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    """
    Custom QuantizeConfig for RepVGGBlock.
    Directs tfmot to quantize the internal branches (3x3 and 1x1) separately.
    
    [CRITICAL FIX]: We DO NOT quantize 'activations' explicitly here because
    RepVGGBlock.activation is a Layer object (e.g. LeakyReLU), not a standard 
    callable/string that TFMOT supports. We rely on 'get_output_quantizers' 
    to quantize the final output of the block (which is post-activation).
    """

    def get_weights_and_quantizers(self, layer):
        """Define which weights to quantize."""
        # We target the kernels of the internal Conv2D layers.
        weights_quantizers = []
        
        # 1. Dense Branch (3x3)
        if hasattr(layer, 'rbr_dense_conv') and layer.rbr_dense_conv is not None:
            weights_quantizers.append(
                (layer.rbr_dense_conv.kernel, tfmot.quantization.keras.quantizers.LastValueQuantizer(
                    num_bits=8, per_axis=False, symmetric=True, narrow_range=True
                ))
            )
            
        # 2. 1x1 Branch
        if hasattr(layer, 'rbr_1x1_conv') and layer.rbr_1x1_conv is not None:
            weights_quantizers.append(
                (layer.rbr_1x1_conv.kernel, tfmot.quantization.keras.quantizers.LastValueQuantizer(
                    num_bits=8, per_axis=False, symmetric=True, narrow_range=True
                ))
            )

        # 3. Deploy Branch (if QAT is run after re-param, though rare)
        if hasattr(layer, 'rbr_reparam') and layer.rbr_reparam is not None:
             weights_quantizers.append(
                (layer.rbr_reparam.kernel, tfmot.quantization.keras.quantizers.LastValueQuantizer(
                    num_bits=8, per_axis=False, symmetric=True, narrow_range=True
                ))
            )
            
        return weights_quantizers

    def get_activations_and_quantizers(self, layer):
        """
        Define which activations to quantize.
        Returns empty list to prevent TFMOT from trying to wrap 
        Layer objects (like L.LeakyReLU) as activation functions.
        """
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        """Apply the quantized weights back to the layer logic."""
        iterator = iter(quantize_weights)
        
        if hasattr(layer, 'rbr_dense_conv') and layer.rbr_dense_conv is not None:
            layer.rbr_dense_conv.kernel = next(iterator)
            
        if hasattr(layer, 'rbr_1x1_conv') and layer.rbr_1x1_conv is not None:
            layer.rbr_1x1_conv.kernel = next(iterator)
            
        if hasattr(layer, 'rbr_reparam') and layer.rbr_reparam is not None:
            layer.rbr_reparam.kernel = next(iterator)

    def set_quantize_activations(self, layer, quantize_activations):
        """
        Apply the quantized activation.
        No-op since get_activations_and_quantizers returns empty.
        """
        pass 

    def get_output_quantizers(self, layer):
        """
        Quantize the final output of the layer.
        This captures the post-activation output of the RepVGGBlock.
        """
        return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=8, per_axis=False, symmetric=False, narrow_range=False
        )]

    def get_config(self):
        return {}


class DeformableDepthwiseQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    """
    QuantizeConfig for DeformableDepthwiseConv2D.
    Quantizes:
    - depthwise kernel (main conv weight)
    - offset adaptor kernel (offset/mask predictor)
    """

    def get_weights_and_quantizers(self, layer):
        q = tfmot.quantization.keras.quantizers.LastValueQuantizer(
            num_bits=8, per_axis=False, symmetric=True, narrow_range=True
        )
        out = []
        if hasattr(layer, "depthwise_kernel") and layer.depthwise_kernel is not None:
            out.append((layer.depthwise_kernel, q))
        if hasattr(layer, "offset_conv") and layer.offset_conv is not None:
            if hasattr(layer.offset_conv, "kernel") and layer.offset_conv.kernel is not None:
                out.append((layer.offset_conv.kernel, q))
        return out

    def get_activations_and_quantizers(self, layer):
        del layer
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        iterator = iter(quantize_weights)
        if hasattr(layer, "depthwise_kernel") and layer.depthwise_kernel is not None:
            layer.depthwise_kernel = next(iterator)
        if hasattr(layer, "offset_conv") and layer.offset_conv is not None:
            if hasattr(layer.offset_conv, "kernel") and layer.offset_conv.kernel is not None:
                layer.offset_conv.kernel = next(iterator)

    def set_quantize_activations(self, layer, quantize_activations):
        del layer, quantize_activations

    def get_output_quantizers(self, layer):
        del layer
        return [
            tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
                num_bits=8, per_axis=False, symmetric=False, narrow_range=False
            )
        ]

    def get_config(self):
        return {}
