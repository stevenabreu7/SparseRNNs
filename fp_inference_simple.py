import jax
import os
import pickle
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from functools import partial

from sparseRNNs.utils.logging_np import logger
from sparseRNNs.fxparray_np import FxpArray
from sparseRNNs.fxparray_np import fxp_from_fp
from sparseRNNs.fxparray_np import RoundingMode
from sparseRNNs.fxpmodel_np import FxpRegressionModel, FxpSSM, FxpS5Config
from sparseRNNs.fxputils import create_fxp_qconfig, add_target_bits_exp
from sparseRNNs.model.ssm_np import discretize_zoh
from sparseRNNs.model.ssm import QuantizationConfig


def load_data(data_folder="data/model_1x"):
    # Create temporary alias for the old module name to fix pickle loading
    import sparseRNNs.fxparray_np as fxparray
    sys.modules['fxparray'] = fxparray

    with open(f"{data_folder}/fxpmodel_fxp_qconfig.pkl", "rb") as f:
        fxp_qconfig = pickle.load(f)

    with open(f"{data_folder}/fxpmodel_io.pkl", "rb") as f:
        io = pickle.load(f)

    with open(f"{data_folder}/fxpmodel.pkl", "rb") as f:
        fxpmodel = pickle.load(f)
        params = fxpmodel['params']
        qconfig_verbose = fxpmodel['qconfig']

    with open(f"{data_folder}/fxpmodel_modeldict.pkl", "rb") as f:
        modeldict = pickle.load(f)

    # Clean up the temporary alias
    del sys.modules['fxparray']

    return io, params, modeldict, fxp_qconfig


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simple_fxp_sigmoid(x, x_exp=6, y_exp=8):
    """Simple fixed-point sigmoid implementation"""
    # Convert to int if float
    if isinstance(x, float) or (hasattr(x, "dtype") and np.issubdtype(x.dtype, np.floating)):
        xx = (x * (1 << x_exp)).astype(int)
    else:
        xx = x.astype(int)
    
    # Simple sigmoid approximation
    sign = 2 * (xx > 0) - 1
    abs_x = np.abs(xx)
    
    # Clamp to avoid overflow
    abs_x = np.minimum(abs_x, 6 * (1 << x_exp))
    
    # Simple lookup table approach
    sigmoid_val = sigmoid(abs_x / (1 << x_exp))
    yy = np.round(sigmoid_val * (1 << y_exp)).astype(int)
    
    return yy


def simple_recurrent_loop(Bu_real, Bu_imag, A_real, A_imag, 
                         Bu_real_exp, Bu_imag_exp, A_real_exp, A_imag_exp,
                         x_real_exp, x_imag_exp):
    """Simple recurrent loop implementation"""
    seq_len, d_ssm = Bu_real.shape
    
    # Initialize state
    x_real = np.zeros(d_ssm, dtype=np.int32)
    x_imag = np.zeros(d_ssm, dtype=np.int32)
    
    outputs_real = []
    outputs_imag = []
    
    for t in range(seq_len):
        # Get current inputs
        But_real = Bu_real[t]
        But_imag = Bu_imag[t]
        
        # Complex multiplication: A * x
        Ax_real = ((A_real * x_real) >> A_real_exp) - ((A_imag * x_imag) >> A_real_exp)
        Ax_imag = ((A_real * x_imag) >> A_imag_exp) + ((A_imag * x_real) >> A_imag_exp)
        
        # Adjust Bu based on exponent differences
        Bu_real_adj = (
            (But_real >> (Bu_real_exp - x_real_exp))
            if Bu_real_exp > x_real_exp
            else (But_real << (x_real_exp - Bu_real_exp))
        )
        Bu_imag_adj = (
            (But_imag >> (Bu_imag_exp - x_imag_exp))
            if Bu_imag_exp > x_imag_exp
            else (But_imag << (x_imag_exp - Bu_imag_exp))
        )
        
        # Update state: x = Ax + Bu
        x_real = Ax_real + Bu_real_adj
        x_imag = Ax_imag + Bu_imag_adj
        
        outputs_real.append(x_real.copy())
        outputs_imag.append(x_imag.copy())
    
    return np.stack(outputs_real), np.stack(outputs_imag)


def simple_first_encoder_layer_forward(encoder_output, modeldict, fxp_qconfig, layer0_qconfig):
    """Complete forward pass through the first encoder layer"""
    print("\n=== Simple First Encoder Layer Forward Pass ===")
    
    # Get layer 0 data
    layer0_data = modeldict['encoder']['layers_0']
    
    # Get input (after encoder and relu)
    x = encoder_output.data.astype(np.float32) / (1 << encoder_output.exp)
    print(f"Input shape: {x.shape}")
    print(f"Input (first 5 values): {x[:5, 0]}")
    
    # Step 1: Batch Normalization
    print("\n--- Step 1: Batch Normalization ---")
    bn_mean = layer0_data['norm']['mean']
    bn_var = layer0_data['norm']['var']
    bn_scale = layer0_data['norm'].get('scale', np.ones_like(bn_mean))
    bn_bias = layer0_data['norm'].get('bias', np.zeros_like(bn_mean))
    bn_eps = 1e-5
    
    # Apply batch norm: (x - mean) / sqrt(var + eps) * scale + bias
    x_norm = (x - bn_mean) / np.sqrt(bn_var + bn_eps)
    if bn_scale is not None:
        x_norm = x_norm * bn_scale
    if bn_bias is not None:
        x_norm = x_norm + bn_bias
    
    print(f"After batch norm (first 5 values): {x_norm[:5, 0]}")
    
    # Step 2: SSM Forward Pass
    print("\n--- Step 2: SSM Forward Pass ---")
    ssm_data = layer0_data['mixer']
    
    # Get SSM parameters
    B_complex = ssm_data['B'][..., 0] + 1j * ssm_data['B'][..., 1]
    Lambda_re = ssm_data['Lambda_re']
    Lambda_im = ssm_data['Lambda_im']
    Lambda_complex = Lambda_re + 1j * Lambda_im
    C_complex = ssm_data['C'][..., 0] + 1j * ssm_data['C'][..., 1]
    D = ssm_data['D']
    log_step = ssm_data['log_step']
    step = np.exp(log_step[:, 0])
    
    print(f"B_complex shape: {B_complex.shape}, max magnitude: {np.abs(B_complex).max():.6f}")
    print(f"Lambda_complex shape: {Lambda_complex.shape}, max magnitude: {np.abs(Lambda_complex).max():.6f}")
    print(f"C_complex shape: {C_complex.shape}, max magnitude: {np.abs(C_complex).max():.6f}")
    print(f"D shape: {D.shape}, max magnitude: {np.abs(D).max():.6f}")
    print(f"step values (first 5): {step[:5]}")
    
    # Discretize
    Lambda_bar, B_bar = discretize_zoh(Lambda_complex, B_complex, step)
    
    print(f"Lambda_bar magnitude (first 5): {np.abs(Lambda_bar[:5])}")
    print(f"B_bar magnitude max: {np.abs(B_bar).max():.6f}")
    
    # Compute Bu = B @ u (using batch-normalized input for recurrence)
    Bu = x_norm @ B_bar.T  # Shape: (seq_len, d_ssm)
    
    print(f"Bu shape: {Bu.shape}")
    print(f"Bu magnitude (first 5, first dim): {np.abs(Bu[:5, 0])}")
    print(f"Bu real (first 5, first dim): {Bu[:5, 0].real}")
    print(f"Bu imag (first 5, first dim): {Bu[:5, 0].imag}")
    
    # Recurrent computation: x_t = Lambda_bar * x_{t-1} + Bu_t
    seq_len, d_ssm = Bu.shape
    x_ssm = np.zeros((seq_len, d_ssm), dtype=np.complex128)
    x_state = np.zeros(d_ssm, dtype=np.complex128)
    
    for t in range(seq_len):
        x_state = Lambda_bar * x_state + Bu[t]
        x_ssm[t] = x_state
    
    print(f"x_ssm after recurrence (first 5, first dim real): {x_ssm[:5, 0].real}")
    print(f"x_ssm after recurrence (first 5, first dim imag): {x_ssm[:5, 0].imag}")
    
    # Apply ReLU (only to real part for complex numbers)
    x_ssm_relu = np.where(x_ssm.real > 0, x_ssm, 0)
    
    print(f"x_ssm after ReLU (first 5, first dim real): {x_ssm_relu[:5, 0].real}")
    print(f"x_ssm after ReLU (first 5, first dim imag): {x_ssm_relu[:5, 0].imag}")
    
    # Output projection: y = C @ x + D @ u
    y_c_projection = np.real(x_ssm_relu @ C_complex.T)
    print(f"C projection (first 5 values): {y_c_projection[:5, 0]}")
    
    # Check conjugate symmetry
    conj_sym = True  # Based on the SSM setup in the full model
    if conj_sym:
        y_c_projection *= 2  # Double for conjugate symmetry
        print(f"After conjugate symmetry doubling (first 5 values): {y_c_projection[:5, 0]}")
    
    # KEY FIX: Add feedthrough using RAW input (before batch norm), not normalized input
    # This matches what the full model is actually doing based on our debugging
    y_d_projection = x * D  # Use x (raw input), not x_norm (batch normalized input)
    print(f"D projection using RAW input (first 5 values): {y_d_projection[:5, 0]}")
    
    y_ssm = y_c_projection + y_d_projection
    
    print(f"SSM output shape: {y_ssm.shape}")
    print(f"SSM output (first 5 values): {y_ssm[:5, 0]}")
    
    # Step 3: GLU (Gated Linear Unit) 
    print("\n--- Step 3: GLU ---")
    out2_weight = layer0_data['out2']['kernel']
    out2_bias = layer0_data['out2']['bias']
    
    print(f"out2_weight shape: {out2_weight.shape}, max: {out2_weight.max():.6f}")
    print(f"out2_bias shape: {out2_bias.shape}, max: {out2_bias.max():.6f}")
    
    # Apply ReLU to SSM output
    x_glu = np.maximum(y_ssm, 0)
    print(f"After ReLU for GLU (first 5 values): {x_glu[:5, 0]}")
    
    # Gate computation: sigmoid(out2(x))
    gate_logits = x_glu @ out2_weight.T + out2_bias
    print(f"Gate logits (first 5 values): {gate_logits[:5, 0]}")
    
    gate = sigmoid(gate_logits)
    print(f"Gate values (first 5 values): {gate[:5, 0]}")
    
    # Multiply: x * gate
    x_gated = x_glu * gate
    
    print(f"GLU output shape: {x_gated.shape}")
    print(f"GLU output (first 5 values): {x_gated[:5, 0]}")
    
    # Step 4: Residual Connection
    print("\n--- Step 4: Residual Connection ---")
    print(f"Input for residual (first 5 values): {x[:5, 0]}")
    print(f"GLU output for residual (first 5 values): {x_gated[:5, 0]}")
    
    x_residual = x_gated + x  # Add skip connection
    print(f"After residual (first 5 values): {x_residual[:5, 0]}")
    
    # Step 5: Final ReLU
    x_final = np.maximum(x_residual, 0)
    
    print(f"Final output shape: {x_final.shape}")
    print(f"Final output (first 5 values): {x_final[:5, 0]}")
    
    return {
        'input': x,
        'after_batch_norm': x_norm,
        'ssm_output': y_ssm,
        'glu_output': x_gated,
        'residual_output': x_residual,
        'final_output': x_final,
        'ssm_states': x_ssm,
        'Bu': Bu,
        'C_projection': y_c_projection,
        'D_projection': y_d_projection,  # Now using raw input
        'gate_logits': gate_logits,
        'gate_values': gate,
    }


def simple_first_encoder_layer(inputs, modeldict, fxp_qconfig, recipe):
    """Simple numpy implementation of the first encoder layer"""
    print("\n=== Simple First Encoder Layer Implementation ===")
    
    # Get layer 0 data
    layer0_data = modeldict['encoder']['layers_0']
    layer0_qconfig = fxp_qconfig['blocks']
    
    # Check if using shared exponents
    if 'layers_0' in layer0_qconfig:
        layer0_qconfig = layer0_qconfig['layers_0']
    
    # 1. Linear encoder (already done in main, use encoder output)
    # We'll get this from the full model run
    
    # 2. First sequence layer components
    # Batch norm parameters
    bn_mean = layer0_data['norm']['mean']
    bn_var = layer0_data['norm']['var'] 
    bn_scale = layer0_data.get('norm', {}).get('scale', np.ones_like(bn_mean))
    bn_bias = layer0_data.get('norm', {}).get('bias', np.zeros_like(bn_mean))
    bn_eps = 1e-5
    
    # SSM parameters
    ssm_data = layer0_data['mixer']
    B_complex = ssm_data['B'][..., 0] + 1j * ssm_data['B'][..., 1]
    Lambda_re = ssm_data['Lambda_re']
    Lambda_im = ssm_data['Lambda_im']
    Lambda_complex = Lambda_re + 1j * Lambda_im
    C_complex = ssm_data['C'][..., 0] + 1j * ssm_data['C'][..., 1]
    D = ssm_data['D']
    log_step = ssm_data['log_step']
    step = np.exp(log_step[:, 0])
    
    # Discretize
    Lambda_bar, B_bar = discretize_zoh(Lambda_complex, B_complex, step)
    
    # GLU parameters
    out2_weight = layer0_data['out2']['kernel']
    out2_bias = layer0_data['out2']['bias']
    
    print(f"Input shape: {inputs.shape}")
    print(f"Batch norm mean shape: {bn_mean.shape}")
    print(f"Lambda_bar shape: {Lambda_bar.shape}")
    print(f"B_bar shape: {B_bar.shape}")
    print(f"C_complex shape: {C_complex.shape}")
    print(f"D shape: {D.shape}")
    print(f"out2_weight shape: {out2_weight.shape}")
    
    return {
        'bn_mean': bn_mean,
        'bn_var': bn_var,
        'bn_scale': bn_scale,
        'bn_bias': bn_bias,
        'Lambda_bar': Lambda_bar,
        'B_bar': B_bar,
        'C_complex': C_complex,
        'D': D,
        'out2_weight': out2_weight,
        'out2_bias': out2_bias,
        'layer0_qconfig': layer0_qconfig,
    }


def run_full_model_and_extract_intermediates(fxp_inputs, model_cls):
    """Run the full model and extract intermediate values"""
    print("\n=== Running Full Model ===")
    
    # Set up model
    model = model_cls(store_intermediates=True)
    
    # Do forward pass
    logger.info("Starting forward pass...")
    tstart = time.time()
    y = model(fxp_inputs, integration_timesteps=10)
    tend = time.time()
    logger.info(f"Forward pass took {tend - tstart:.3f} seconds")
    
    # Extract intermediates
    intermediates = model.last_intermediates()
    
    # Get encoder intermediates
    encoder_intermediates = model.encoder.last_intermediates()
    
    # Get first layer intermediates  
    layer0_intermediates = model.encoder.seq_layers[0].last_intermediates()
    
    # Get SSM intermediates
    ssm_intermediates = model.encoder.seq_layers[0].mixer.last_intermediates()
    
    # Get batch norm intermediates
    bn_intermediates = model.encoder.seq_layers[0].norm.last_intermediates()
    
    return {
        'output': y,
        'model_intermediates': intermediates,
        'encoder_intermediates': encoder_intermediates,
        'layer0_intermediates': layer0_intermediates,
        'ssm_intermediates': ssm_intermediates,
        'bn_intermediates': bn_intermediates,
    }


def plot_comparisons(simple_results, full_results):
    """Plot comparisons between simple and full implementations"""
    print("\n=== Plotting Comparisons ===")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Comparison: Simple vs Full Implementation', fontsize=16)
    
    # Get encoder output for simple implementation input
    encoder_output_relu = full_results['encoder_intermediates']['encoder_output_relu']
    encoder_data = encoder_output_relu.data.astype(np.float32) / (1 << encoder_output_relu.exp)
    
    # Get full model outputs
    layer0_pre_s5 = full_results['layer0_intermediates']['pre_s5']
    layer0_pre_s5_data = layer0_pre_s5.data.astype(np.float32) / (1 << layer0_pre_s5.exp)
    
    layer0_output = full_results['layer0_intermediates']['output']
    layer0_output_data = layer0_output.data.astype(np.float32) / (1 << layer0_output.exp)
    
    # Plot 1: Input comparison
    axes[0, 0].plot(encoder_data[:100, 0], 'b-', label='Simple Input', alpha=0.7)
    axes[0, 0].plot(encoder_data[:100, 0], 'r--', label='Full Input', alpha=0.7)
    axes[0, 0].set_title('Input (Encoder Output)')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: After Batch Norm
    axes[0, 1].plot(simple_results['after_batch_norm'][:100, 0], 'b-', label='Simple BN', alpha=0.7)
    axes[0, 1].plot(layer0_pre_s5_data[:100, 0], 'r--', label='Full BN', alpha=0.7)
    axes[0, 1].set_title('After Batch Normalization')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: SSM Output
    ssm_output = full_results['ssm_intermediates'].get('ys')
    if ssm_output is not None:
        ssm_output_data = ssm_output.data.astype(np.float32) / (1 << ssm_output.exp)
        axes[0, 2].plot(simple_results['ssm_output'][:100, 0], 'b-', label='Simple SSM', alpha=0.7)
        axes[0, 2].plot(ssm_output_data[:100, 0], 'r--', label='Full SSM', alpha=0.7)
    else:
        axes[0, 2].plot(simple_results['ssm_output'][:100, 0], 'b-', label='Simple SSM', alpha=0.7)
    axes[0, 2].set_title('SSM Output')
    axes[0, 2].set_xlabel('Time Step')
    axes[0, 2].set_ylabel('Value')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Plot 4: GLU Output
    glu_output = full_results['layer0_intermediates'].get('post_GLU')
    if glu_output is not None:
        glu_output_data = glu_output.data.astype(np.float32) / (1 << glu_output.exp)
        axes[1, 0].plot(simple_results['glu_output'][:100, 0], 'b-', label='Simple GLU', alpha=0.7)
        axes[1, 0].plot(glu_output_data[:100, 0], 'r--', label='Full GLU', alpha=0.7)
    else:
        axes[1, 0].plot(simple_results['glu_output'][:100, 0], 'b-', label='Simple GLU', alpha=0.7)
    axes[1, 0].set_title('GLU Output')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 5: Final Output
    axes[1, 1].plot(simple_results['final_output'][:100, 0], 'b-', label='Simple Final', alpha=0.7)
    axes[1, 1].plot(layer0_output_data[:100, 0], 'r--', label='Full Final', alpha=0.7)
    axes[1, 1].set_title('Final Output')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Plot 6: Error Analysis
    error = simple_results['final_output'][:100, 0] - layer0_output_data[:100, 0]
    axes[1, 2].plot(error, 'g-', label='Difference', alpha=0.7)
    axes[1, 2].set_title('Error (Simple - Full)')
    axes[1, 2].set_xlabel('Time Step')
    axes[1, 2].set_ylabel('Error')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('fp_layer0_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"Mean absolute error: {np.mean(np.abs(error)):.6f}")
    print(f"Max absolute error: {np.max(np.abs(error)):.6f}")
    print(f"RMS error: {np.sqrt(np.mean(error**2)):.6f}")


def compare_intermediates(simple_results, full_results):
    """Compare simple implementation results with full model intermediates"""
    print("\n=== Detailed Comparison of Intermediates ===")
    
    # Helper function to convert FxpArray to float
    def fxp_to_float(fxp_arr):
        if fxp_arr is None:
            return None
        return fxp_arr.data.astype(np.float32) / (1 << fxp_arr.exp)
    
    # Get key intermediate values from full model
    encoder_output = full_results['encoder_intermediates']['encoder_output']
    encoder_output_relu = full_results['encoder_intermediates']['encoder_output_relu']
    
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Encoder output (first 5 values): {encoder_output[:5, 0]}")
    print(f"Encoder output ReLU (first 5 values): {encoder_output_relu[:5, 0]}")
    
    # Convert to float for comparison
    encoder_float = fxp_to_float(encoder_output_relu)
    print(f"Encoder output ReLU as float (first 5 values): {encoder_float[:5, 0]}")
    
    # Layer 0 intermediates
    layer0_input = full_results['layer0_intermediates']['ssm_input']
    layer0_pre_s5 = full_results['layer0_intermediates']['pre_s5']
    layer0_output = full_results['layer0_intermediates']['output']
    
    print(f"\n=== Batch Norm Comparison ===")
    print(f"Layer 0 input shape: {layer0_input.shape}")
    print(f"Layer 0 input (first 5 values): {layer0_input[:5, 0]}")
    print(f"Layer 0 pre_s5 (first 5 values): {layer0_pre_s5[:5, 0]}")
    
    # Convert batch norm output to float
    bn_float = fxp_to_float(layer0_pre_s5)
    print(f"Full model batch norm as float (first 5): {bn_float[:5, 0]}")
    print(f"Simple model batch norm (first 5): {simple_results['after_batch_norm'][:5, 0]}")
    bn_error = np.abs(bn_float[:5, 0] - simple_results['after_batch_norm'][:5, 0])
    print(f"Batch norm error (first 5): {bn_error}")
    
    # SSM intermediates
    print(f"\n=== SSM Comparison ===")
    if 'Bu_elements' in full_results['ssm_intermediates']:
        Bu_elements = full_results['ssm_intermediates']['Bu_elements']
        print(f"SSM Bu_elements shape: {Bu_elements.real.shape}")
        print(f"SSM Bu_elements real (first 5 values): {Bu_elements.real[:5, 0]}")
        print(f"SSM Bu_elements imag (first 5 values): {Bu_elements.imag[:5, 0]}")
        
        # Convert to float
        bu_real_float = fxp_to_float(Bu_elements.real)
        bu_imag_float = fxp_to_float(Bu_elements.imag)
        print(f"Full model Bu real as float (first 5): {bu_real_float[:5, 0] if bu_real_float is not None else 'None'}")
        print(f"Full model Bu imag as float (first 5): {bu_imag_float[:5, 0] if bu_imag_float is not None else 'None'}")
        print(f"Simple model Bu real (first 5): {simple_results['Bu'][:5, 0].real}")
        print(f"Simple model Bu imag (first 5): {simple_results['Bu'][:5, 0].imag}")
    
    if 'xs' in full_results['ssm_intermediates']:
        xs = full_results['ssm_intermediates']['xs']
        print(f"SSM xs shape: {xs.real.shape}")
        print(f"SSM xs real (first 5 values): {xs.real[:5, 0]}")
        print(f"SSM xs imag (first 5 values): {xs.imag[:5, 0]}")
        
        # Convert to float
        xs_real_float = fxp_to_float(xs.real)
        xs_imag_float = fxp_to_float(xs.imag)
        print(f"Full model xs real as float (first 5): {xs_real_float[:5, 0] if xs_real_float is not None else 'None'}")
        print(f"Full model xs imag as float (first 5): {xs_imag_float[:5, 0] if xs_imag_float is not None else 'None'}")
        print(f"Simple model xs real (first 5): {simple_results['ssm_states'][:5, 0].real}")
        print(f"Simple model xs imag (first 5): {simple_results['ssm_states'][:5, 0].imag}")
    
    # Check C projection
    if 'Cxs' in full_results['ssm_intermediates']:
        cxs = full_results['ssm_intermediates']['Cxs']
        print(f"SSM Cxs (first 5 values): {cxs[:5, 0]}")
        cxs_float = fxp_to_float(cxs)
        print(f"Full model C projection as float (first 5): {cxs_float[:5, 0]}")
        print(f"Simple model C projection (first 5): {simple_results['C_projection'][:5, 0]}")
    
    if '2Cxs' in full_results['ssm_intermediates']:
        two_cxs = full_results['ssm_intermediates']['2Cxs']
        print(f"SSM 2Cxs (first 5 values): {two_cxs[:5, 0]}")
        two_cxs_float = fxp_to_float(two_cxs)
        print(f"Full model 2*C projection as float (first 5): {two_cxs_float[:5, 0]}")
    
    # Check Du (feedthrough) term - this is likely the key!
    if 'Du' in full_results['ssm_intermediates']:
        du = full_results['ssm_intermediates']['Du']
        print(f"SSM Du (feedthrough) shape: {du.shape}")
        print(f"SSM Du (first 5 values): {du[:5, 0]}")
        du_float = fxp_to_float(du)
        print(f"Full model Du as float (first 5): {du_float[:5, 0]}")
        print(f"Simple model D projection (first 5): {simple_results['D_projection'][:5, 0]}")
        
        # The key insight: check if the full model Du is using batch-norm fused input
        # while our simple model is using the normalized input
        print(f"Full model input to Du (layer0_input) as float: {fxp_to_float(layer0_input)[:5, 0]}")
        print(f"Simple model input to D: {simple_results['after_batch_norm'][:5, 0]}")
    
    if 'ys' in full_results['ssm_intermediates']:
        ys = full_results['ssm_intermediates']['ys']
        print(f"SSM ys shape: {ys.shape}")
        print(f"SSM ys (first 5 values): {ys[:5, 0]}")
        
        # Convert to float
        ys_float = fxp_to_float(ys)
        print(f"Full model SSM output as float (first 5): {ys_float[:5, 0]}")
        print(f"Simple model SSM output (first 5): {simple_results['ssm_output'][:5, 0]}")
        ssm_error = np.abs(ys_float[:5, 0] - simple_results['ssm_output'][:5, 0])
        print(f"SSM output error (first 5): {ssm_error}")
    
    # GLU comparison
    print(f"\n=== GLU Comparison ===")
    if 'post_GLU' in full_results['layer0_intermediates']:
        post_glu = full_results['layer0_intermediates']['post_GLU']
        print(f"Full model GLU output: {post_glu[:5, 0]}")
        
        post_glu_float = fxp_to_float(post_glu)
        print(f"Full model GLU as float (first 5): {post_glu_float[:5, 0]}")
        print(f"Simple model GLU (first 5): {simple_results['glu_output'][:5, 0]}")
        glu_error = np.abs(post_glu_float[:5, 0] - simple_results['glu_output'][:5, 0])
        print(f"GLU error (first 5): {glu_error}")
    
    # Check for other GLU-related intermediates
    if 'out2_sigmoid' in full_results['layer0_intermediates']:
        out2_sig = full_results['layer0_intermediates']['out2_sigmoid']
        print(f"Full model sigmoid output: {out2_sig[:5, 0]}")
        
        out2_sig_float = fxp_to_float(out2_sig)
        print(f"Full model sigmoid as float (first 5): {out2_sig_float[:5, 0]}")
        print(f"Simple model sigmoid (first 5): {simple_results['gate_values'][:5, 0]}")
    
    # Check pre_GLU values to see what goes into GLU
    if 'pre_GLU' in full_results['layer0_intermediates']:
        pre_glu = full_results['layer0_intermediates']['pre_GLU']
        print(f"Full model pre_GLU: {pre_glu[:5, 0]}")
        
        pre_glu_float = fxp_to_float(pre_glu)
        print(f"Full model pre_GLU as float (first 5): {pre_glu_float[:5, 0]}")
        print(f"Simple model after SSM ReLU (first 5): {np.maximum(simple_results['ssm_output'][:5, 0], 0)}")
    
    # Batch norm intermediates
    if 'norm_output' in full_results['bn_intermediates']:
        bn_output = full_results['bn_intermediates']['norm_output']
        print(f"\nBatch norm output shape: {bn_output.shape}")
        print(f"Batch norm output (first 5 values): {bn_output[:5, 0]}")
    
    # Final output comparison
    print(f"\n=== Final Output Comparison ===")
    layer0_output_float = fxp_to_float(layer0_output)
    print(f"Full model final output as float (first 5): {layer0_output_float[:5, 0]}")
    print(f"Simple model final output (first 5): {simple_results['final_output'][:5, 0]}")
    final_error = np.abs(layer0_output_float[:5, 0] - simple_results['final_output'][:5, 0])
    print(f"Final output error (first 5): {final_error}")
    
    # Print all available intermediate keys for debugging
    print(f"\n=== Available Intermediate Keys ===")
    print(f"SSM intermediates: {list(full_results['ssm_intermediates'].keys())}")
    print(f"Layer0 intermediates: {list(full_results['layer0_intermediates'].keys())}")
    print(f"BN intermediates: {list(full_results['bn_intermediates'].keys())}")
    
    return {
        'bn_error': bn_error if 'bn_error' in locals() else None,
        'ssm_error': ssm_error if 'ssm_error' in locals() else None,
        'glu_error': glu_error if 'glu_error' in locals() else None,
        'final_error': final_error if 'final_error' in locals() else None,
    }


def main():
    # Load data
    io, params, modeldict, fxp_qconfig = load_data()
    
    # Setup configuration
    recipe_fname = "recipes/ndns.json"
    with open(recipe_fname, "r") as f:
        recipe = json.load(f)
    
    n_layers = max([int(e.split("_")[1]) for e in modeldict['encoder'].keys() if e.startswith('layers_')]) + 1
    d_model = modeldict['encoder']['encoder']['kernel'].shape[-1]
    d_ssm = modeldict['encoder']['layers_0']['mixer']['Lambda_re'].shape[0]
    d_input = modeldict['encoder']['encoder']['kernel'].shape[0]
    d_output = modeldict['decoder']['kernel'].shape[-1]
    padded = recipe['dataset'] in ["imdb-classification", "listops-classification", "aan-classification"]
    
    print(f"{n_layers=}, {d_model=}, {d_ssm=}, {d_input=}, {d_output=}, {padded=}")
    
    # Setup model config
    cfg = FxpS5Config(
        q_config=QuantizationConfig.none(),
        relufication=True,
        H=d_model,
        P=d_ssm,
        discretization="zoh",
        conj_sym=True,
        bidirectional=False,
        associative_scan=False,
        n_layers=n_layers,
        d_model=d_model,
        d_output=d_output,
        padded=padded,
        glu_variant=recipe['glu_variant'],
        bn_momentum=0.95,
        step_rescale=1.0,
        prenorm=recipe['prenorm'],
        batchnorm=recipe['batchnorm'],
        dropout=0.0,
        training=False,
        fuse_batchnorm_linear=False,
    )
    
    # Setup model classes
    mixer_cls = FxpSSM.init_fn(
        H=cfg.H,
        P=cfg.P,
        discretization=cfg.discretization,
        conj_sym=cfg.conj_sym,
        q_config=cfg.q_config,
        bidirectional=cfg.bidirectional,
        relufication=cfg.relufication,
        associative_scan=cfg.associative_scan,
    )
    
    model_cls = partial(
        FxpRegressionModel,
        modeldict=modeldict,
        fxp_qconfig=fxp_qconfig,
        scope="model",
        mixer_cls=mixer_cls,
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        batchnorm=cfg.batchnorm,
        prenorm=cfg.prenorm,
        bn_momentum=cfg.bn_momentum,
        glu_variant=cfg.glu_variant,
        step_rescale=cfg.step_rescale,
        relufication=cfg.relufication,
        fuse_batchnorm_linear=cfg.fuse_batchnorm_linear,
        q_config=cfg.q_config,
        dropout=cfg.dropout,
        training=cfg.training,
        d_output=cfg.d_output,
        padded=cfg.padded,
    )
    
    # Setup input
    logger.info("Loading inputs")
    inputs = io["input"][0]
    seq_len = inputs.shape[0]
    inputs = inputs[:seq_len, :]
    
    fxp_inputs = inputs if isinstance(inputs, FxpArray) else fxp_from_fp(
        inputs,
        bits=fxp_qconfig['encoder']['inp_bits'],
        exp=fxp_qconfig['encoder']['inp_exp'],
        signed=True,
        round_mode=RoundingMode.FLOOR,
    )
    
    # Run full model and extract intermediates
    full_results = run_full_model_and_extract_intermediates(fxp_inputs, model_cls)
    
    # Run simple implementation
    simple_params = simple_first_encoder_layer(fxp_inputs, modeldict, fxp_qconfig, recipe)
    
    # Run simple forward pass
    encoder_output_relu = full_results['encoder_intermediates']['encoder_output_relu']
    simple_results = simple_first_encoder_layer_forward(encoder_output_relu, modeldict, fxp_qconfig, simple_params['layer0_qconfig'])
    
    # Compare results
    compare_intermediates(simple_results, full_results)
    
    # Plot comparisons
    plot_comparisons(simple_results, full_results)


if __name__ == "__main__":
    main() 