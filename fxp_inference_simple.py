import pickle
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from sparseRNNs.utils.logging_np import logger
from sparseRNNs.fxparray_np import (
    FxpArray, ComplexFxpArray, RoundingMode, fxp_from_fp, fxp_add, fxp_sub, 
    fxp_mul, fxp_matmul, fxp_complex_add, fxp_complex_mul
)
from sparseRNNs.fxpmodel_np import (
    FxpRegressionModel, FxpSSM, FxpS5Config, fxp_relu, FxpSigmoid
)
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


def simple_fxp_batch_norm(x_fxp, modeldict, fxp_qconfig, scope="norm"):
    """Apply batch normalization using FxpArray operations"""
    print(f"\n--- FXP Batch Normalization ({scope}) ---")
    
    # Convert batch norm parameters to FxpArray
    partial_fxp_from_fp = partial(
        fxp_from_fp, signed=True, round_mode=RoundingMode.ROUND
    )
    
    bn_eps = 1e-5
    
    # Convert mean to FxpArray with negative sign
    minus_mean = partial_fxp_from_fp(
        -1 * modeldict["mean"],
        bits=fxp_qconfig["mean"]["bits"],
        exp=fxp_qconfig["mean"]["exp"],
    )
    
    # Convert inverse sqrt variance to FxpArray
    invsq_var = partial_fxp_from_fp(
        1.0 / np.sqrt(modeldict["var"] + bn_eps),
        bits=fxp_qconfig["invsq_var"]["bits"],
        exp=fxp_qconfig["invsq_var"]["exp"],
    )
    
    # Convert scale and bias if present
    scale = None
    bias = None
    if "scale" in modeldict:
        scale = partial_fxp_from_fp(
            modeldict["scale"],
            bits=fxp_qconfig["scale"]["bits"],
            exp=fxp_qconfig["scale"]["exp"],
        )
    if "bias" in modeldict:
        bias = partial_fxp_from_fp(
            modeldict["bias"],
            bits=fxp_qconfig["bias"]["bits"],
            exp=fxp_qconfig["bias"]["exp"],
        )
    
    print(f"Input: bits={x_fxp.bits}, exp={x_fxp.exp}, shape={x_fxp.shape}")
    print(f"minus_mean: bits={minus_mean.bits}, exp={minus_mean.exp}")
    print(f"invsq_var: bits={invsq_var.bits}, exp={invsq_var.exp}")
    
    # Apply batch norm: (x - mean) / sqrt(var + eps) * scale + bias
    x = fxp_add(x_fxp, minus_mean, result_exp="compute_best")
    print(f"After x - mean: bits={x.bits}, exp={x.exp}")
    
    x = fxp_mul(x, invsq_var, result_exp="compute_best")
    print(f"After / sqrt(var): bits={x.bits}, exp={x.exp}")
    
    if scale is not None:
        x = fxp_mul(x, scale, result_exp="compute_best")
        print(f"After * scale: bits={x.bits}, exp={x.exp}")
    
    if bias is not None:
        x = fxp_add(x, bias, result_exp="compute_best")
        print(f"After + bias: bits={x.bits}, exp={x.exp}")
    
    print(f"BN output: bits={x.bits}, exp={x.exp}")
    print(f"BN output values (first 5): {x.to_float()[:5, 0]}")
    
    return x


def simple_fxp_ssm_forward(x_fxp, modeldict, fxp_qconfig, scope="ssm"):
    """SSM forward pass using FxpArray operations"""
    print(f"\n--- FXP SSM Forward ({scope}) ---")
    
    # Get SSM parameters
    B_tilde = modeldict["B"][..., 0] + 1j * modeldict["B"][..., 1]
    Lambda_re = modeldict["Lambda_re"]
    Lambda_im = modeldict["Lambda_im"]
    Lambda = Lambda_re + 1j * Lambda_im
    C_tilde = modeldict["C"][..., 0] + 1j * modeldict["C"][..., 1]
    D = modeldict["D"]
    log_step = modeldict["log_step"]
    step = np.exp(log_step[:, 0])
    
    # Discretize
    Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
    
    print(f"Lambda_bar magnitude: {np.abs(Lambda_bar).max():.6f}")
    print(f"B_bar magnitude: {np.abs(B_bar).max():.6f}")
    print(f"C_tilde magnitude: {np.abs(C_tilde).max():.6f}")
    print(f"D magnitude: {np.abs(D).max():.6f}")
    
    # Convert to FxpArray
    partial_fxp_from_fp = partial(
        fxp_from_fp, signed=True, round_mode=RoundingMode.ROUND
    )
    
    # Convert Lambda_bar to ComplexFxpArray
    Lambda_bar_fxp = ComplexFxpArray(
        real=partial_fxp_from_fp(
            Lambda_bar.real,
            bits=fxp_qconfig["weights"]["A_re"]["bits"],
            exp=fxp_qconfig["weights"]["A_re"]["exp"],
        ),
        imag=partial_fxp_from_fp(
            Lambda_bar.imag,
            bits=fxp_qconfig["weights"]["A_im"]["bits"],
            exp=fxp_qconfig["weights"]["A_im"]["exp"],
        ),
    )
    
    # Convert B_bar to ComplexFxpArray
    B_bar_fxp = ComplexFxpArray(
        real=partial_fxp_from_fp(
            B_bar.real,
            bits=fxp_qconfig["weights"]["B_re"]["bits"],
            exp=fxp_qconfig["weights"]["B_re"]["exp"],
        ),
        imag=partial_fxp_from_fp(
            B_bar.imag,
            bits=fxp_qconfig["weights"]["B_im"]["bits"],
            exp=fxp_qconfig["weights"]["B_im"]["exp"],
        ),
    )
    
    # Convert C_tilde to ComplexFxpArray
    C_tilde_fxp = ComplexFxpArray(
        real=partial_fxp_from_fp(
            C_tilde.real,
            bits=fxp_qconfig["weights"]["C_re"]["bits"],
            exp=fxp_qconfig["weights"]["C_re"]["exp"],
        ),
        imag=partial_fxp_from_fp(
            C_tilde.imag,
            bits=fxp_qconfig["weights"]["C_im"]["bits"],
            exp=fxp_qconfig["weights"]["C_im"]["exp"],
        ),
    )
    
    # Convert D to FxpArray
    D_fxp = partial_fxp_from_fp(
        D,
        bits=fxp_qconfig["weights"]["D"]["bits"],
        exp=fxp_qconfig["weights"]["D"]["exp"],
    )
    
    print(f"Lambda_bar_fxp real: bits={Lambda_bar_fxp.real.bits}, exp={Lambda_bar_fxp.real.exp}")
    print(f"B_bar_fxp real: bits={B_bar_fxp.real.bits}, exp={B_bar_fxp.real.exp}")
    print(f"C_tilde_fxp real: bits={C_tilde_fxp.real.bits}, exp={C_tilde_fxp.real.exp}")
    print(f"D_fxp: bits={D_fxp.bits}, exp={D_fxp.exp}")
    
    # Ensure input has correct precision
    x_fxp = x_fxp.change_cfg(
        new_bits=fxp_qconfig["activations"]["u"]["bits"],
        new_exp=fxp_qconfig["activations"]["u"]["exp"],
        new_signed=True,
    )
    
    print(f"SSM input: bits={x_fxp.bits}, exp={x_fxp.exp}")
    
    # Compute Bu = B @ input
    Bu_elements = ComplexFxpArray(
        real=fxp_matmul(
            x_fxp,
            B_bar_fxp.real.transpose(),
            result_exp=fxp_qconfig["activations"]["Bu_re"]["exp"],
            result_bits=fxp_qconfig["activations"]["Bu_re"]["bits"],
        ),
        imag=fxp_matmul(
            x_fxp,
            B_bar_fxp.imag.transpose(),
            result_exp=fxp_qconfig["activations"]["Bu_im"]["exp"],
            result_bits=fxp_qconfig["activations"]["Bu_im"]["bits"],
        ),
    )
    
    print(f"Bu_elements shape: {Bu_elements.shape}")
    print(f"Bu_elements real: bits={Bu_elements.real.bits}, exp={Bu_elements.real.exp}")
    print(f"Bu_elements imag: bits={Bu_elements.imag.bits}, exp={Bu_elements.imag.exp}")
    print(f"Bu_elements real values (first 5): {Bu_elements.real.to_float()[:5, 0]}")
    print(f"Bu_elements imag values (first 5): {Bu_elements.imag.to_float()[:5, 0]}")
    
    # Check quantization threshold for Bu elements
    bu_threshold = 1.0 / (1 << 9)  # Based on exp=9 precision
    bu_real_below = np.abs(Bu_elements.real.to_float()) < bu_threshold
    bu_imag_below = np.abs(Bu_elements.imag.to_float()) < bu_threshold
    print(f"Bu real values below threshold: {np.sum(bu_real_below)} / {bu_real_below.size} ({100*np.sum(bu_real_below)/bu_real_below.size:.1f}%)")
    print(f"Bu imag values below threshold: {np.sum(bu_imag_below)} / {bu_imag_below.size} ({100*np.sum(bu_imag_below)/bu_imag_below.size:.1f}%)")
    
    # Simulate quantization by setting values below threshold to zero
    Bu_elements.real.data = np.where(bu_real_below, 0, Bu_elements.real.data)
    Bu_elements.imag.data = np.where(bu_imag_below, 0, Bu_elements.imag.data)
    
    print(f"After quantization threshold - Bu real values (first 5): {Bu_elements.real.to_float()[:5, 0]}")
    print(f"After quantization threshold - Bu imag values (first 5): {Bu_elements.imag.to_float()[:5, 0]}")
    
    # Initialize recurrent state
    xt_re_bits = fxp_qconfig["activations"]["x_re"]["bits"]
    xt_im_bits = fxp_qconfig["activations"]["x_im"]["bits"]
    xtshape = (Bu_elements.shape[-1],)
    
    xt = ComplexFxpArray(
        real=FxpArray(
            data=np.zeros(xtshape, dtype=np.int32),
            bits=xt_re_bits,
            exp=fxp_qconfig["activations"]["x_re"]["exp"],
            signed=True,
        ),
        imag=FxpArray(
            data=np.zeros(xtshape, dtype=np.int32),
            bits=xt_im_bits,
            exp=fxp_qconfig["activations"]["x_im"]["exp"],
            signed=True,
        ),
    )
    
    # Recurrence using manual loop (simpler to debug)
    xs_list = []
    for t in range(x_fxp.data.shape[0]):
        # A * x_t
        Axt = fxp_complex_mul(
            Lambda_bar_fxp,
            xt,
            result_exp=(xt.real.exp, xt.imag.exp),
            result_bits=(xt_re_bits + 4, xt_im_bits + 4),
        )
        
        # A * x_t + Bu_t
        xt = fxp_complex_add(
            Axt,
            Bu_elements[t],
            result_exp=(xt.real.exp, xt.imag.exp),
            result_bits=(xt_re_bits, xt_im_bits),
            warn_on_overflow=False,
            warn_on_clip=False,
        )
        
        xs_list.append(xt.copy())
    
    # Stack results
    xs = xt.copy()
    xs.real.data = np.stack([x.real.data for x in xs_list], dtype=np.int32)
    xs.imag.data = np.stack([x.imag.data for x in xs_list], dtype=np.int32)
    
    print(f"xs shape: {xs.shape}")
    print(f"xs real values (first 5): {xs.real.to_float()[:5, 0]}")
    print(f"xs imag values (first 5): {xs.imag.to_float()[:5, 0]}")
    
    # Apply ReLU
    xs = fxp_relu(xs)
    
    print(f"xs after ReLU real values (first 5): {xs.real.to_float()[:5, 0]}")
    print(f"xs after ReLU imag values (first 5): {xs.imag.to_float()[:5, 0]}")
    
    # C projection: C @ xs
    ys = fxp_sub(
        fxp_matmul(
            xs.real,
            C_tilde_fxp.real.transpose(),
            result_exp=fxp_qconfig["activations"]["y"]["exp"],
            result_bits=fxp_qconfig["activations"]["y"]["bits"],
        ),
        fxp_matmul(
            xs.imag,
            C_tilde_fxp.imag.transpose(),
            result_exp=fxp_qconfig["activations"]["y"]["exp"],
            result_bits=fxp_qconfig["activations"]["y"]["bits"],
        ),
        result_exp=fxp_qconfig["activations"]["y"]["exp"],
        result_bits=fxp_qconfig["activations"]["y"]["bits"],
    )
    
    print(f"C projection: {ys.to_float()[:5, 0]}")
    
    # Apply conjugate symmetry
    ys.data *= 2
    
    print(f"2 * C projection: {ys.to_float()[:5, 0]}")
    
    # D projection (feedthrough) - this is key!
    # Based on our analysis, the full model uses the input BEFORE batch norm
    # But we need to get that from the calling function
    Du = fxp_mul(
        D_fxp,
        x_fxp,  # This should be the input to the SSM
        result_exp=fxp_qconfig["activations"]["y"]["exp"],
        result_bits=fxp_qconfig["activations"]["y"]["bits"],
    )
    
    print(f"D projection: {Du.to_float()[:5, 0]}")
    
    # Final SSM output: C @ x + D @ u
    ysDu = fxp_add(
        ys,
        Du,
        result_exp=fxp_qconfig["activations"]["y"]["exp"],
        result_bits=fxp_qconfig["activations"]["y"]["bits"],
    )
    
    print(f"SSM output: {ysDu.to_float()[:5, 0]}")
    
    return ysDu, xs, Bu_elements


def simple_fxp_glu(x_fxp, out2_modeldict, fxp_qconfig, scope="glu"):
    """GLU using FxpArray operations"""
    print(f"\n--- FXP GLU ({scope}) ---")
    
    # Convert out2 layer parameters
    partial_fxp_from_fp = partial(
        fxp_from_fp, signed=True, round_mode=RoundingMode.ROUND
    )
    
    out2_weight = partial_fxp_from_fp(
        out2_modeldict["kernel"],
        bits=fxp_qconfig["out2"]["w_bits"],
        exp=fxp_qconfig["out2"]["w_exp"],
    )
    
    out2_bias = partial_fxp_from_fp(
        out2_modeldict["bias"],
        bits=fxp_qconfig["out2"]["b_bits"],
        exp=fxp_qconfig["out2"]["b_exp"],
    )
    
    print(f"out2_weight: bits={out2_weight.bits}, exp={out2_weight.exp}")
    print(f"out2_bias: bits={out2_bias.bits}, exp={out2_bias.exp}")
    
    # Apply ReLU to input
    x_relu = fxp_relu(x_fxp)
    
    print(f"After ReLU: {x_relu.to_float()[:5, 0]}")
    
    # Gate computation: out2(x)
    gate_logits = fxp_matmul(
        x_relu,
        out2_weight.transpose(),
        result_bits=fxp_qconfig["out2"]["out_bits"],
        result_exp=fxp_qconfig["out2"]["out_exp"],
    )
    
    gate_logits = fxp_add(
        gate_logits,
        out2_bias,
        result_bits=fxp_qconfig["out2"]["out_bits"],
        result_exp=fxp_qconfig["out2"]["out_exp"],
    )
    
    print(f"Gate logits: {gate_logits.to_float()[:5, 0]}")
    
    # Apply sigmoid
    sigmoid_x_exp = fxp_qconfig["out2"]["out_exp"]
    sigmoid_y_exp = fxp_qconfig["out2"]["out_bits"] - 2
    
    if sigmoid_x_exp > 6:
        sigmoid_x_exp = 6
    
    sigmoid_cls = FxpSigmoid(x_exp=sigmoid_x_exp, y_exp=sigmoid_y_exp)
    gate = sigmoid_cls.apply(gate_logits, output_fxp=True)
    
    print(f"Gate values: {gate.to_float()[:5, 0]}")
    
    # Multiply: x * gate
    x_gated = fxp_mul(
        x_relu,
        gate,
        result_exp=fxp_qconfig["multgate"]["res_exp"],
        result_bits=fxp_qconfig["multgate"]["res_bits"],
        round_mode=RoundingMode.FLOOR,
        warn_on_overflow=True,
    )
    
    print(f"GLU output: {x_gated.to_float()[:5, 0]}")
    
    return x_gated


def simple_fxp_first_encoder_layer(encoder_output_fxp, modeldict, fxp_qconfig, scope="layer0"):
    """Complete first encoder layer using FxpArray operations"""
    print(f"\n=== FXP First Encoder Layer ({scope}) ===")
    
    # Get layer 0 data
    layer0_data = modeldict['encoder']['layers_0']
    layer0_qconfig = fxp_qconfig['blocks']
    
    # Check if using per-layer configs
    if 'layers_0' in layer0_qconfig:
        layer0_qconfig = layer0_qconfig['layers_0']
    
    print(f"Input to layer0: bits={encoder_output_fxp.bits}, exp={encoder_output_fxp.exp}")
    print(f"Input values (first 5): {encoder_output_fxp.to_float()[:5, 0]}")
    
    # Store original input for residual connection
    skip_connection = encoder_output_fxp.copy()
    
    # Step 1: Batch Normalization
    x_bn = simple_fxp_batch_norm(
        encoder_output_fxp, 
        layer0_data['norm'], 
        layer0_qconfig['norm'],
        scope=f"{scope}.norm"
    )
    
    # Step 2: SSM Forward Pass
    # This is a key finding from our debugging
    ys, xs, Bu_elements = simple_fxp_ssm_forward(
        x_bn,  # Use batch-normalized input for SSM computation
        layer0_data['mixer'],
        layer0_qconfig['ssm'],
        scope=f"{scope}.ssm"
    )

    # Step 3: GLU
    x_glu = simple_fxp_glu(
        ys,
        layer0_data['out2'],
        layer0_qconfig,
        scope=f"{scope}.glu"
    )
    
    # Step 4: Residual connection
    print(f"\n--- Residual Connection ---")
    print(f"GLU output: {x_glu.to_float()[:5, 0]}")
    print(f"Skip connection: {skip_connection.to_float()[:5, 0]}")
    
    x_residual = fxp_add(
        x_glu,
        skip_connection,
        result_exp="compute_best",
        result_bits=layer0_qconfig["multgate"]["res_bits"],
    )
    
    print(f"After residual: {x_residual.to_float()[:5, 0]}")
    
    # Step 5: Final ReLU
    x_final = fxp_relu(x_residual)
    
    print(f"Final output: {x_final.to_float()[:5, 0]}")
    
    return {
        'input': encoder_output_fxp,
        'after_batch_norm': x_bn,
        'ssm_output': ys,
        'glu_output': x_glu,
        'residual_output': x_residual,
        'final_output': x_final,
        'ssm_states': xs,
        'Bu_elements': Bu_elements,
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


def compare_fxp_results(simple_results, full_results):
    """Compare FXP simple implementation with full model"""
    print("\n=== FXP Results Comparison ===")
    
    # Helper function to convert FxpArray to float
    def fxp_to_float(fxp_arr):
        if fxp_arr is None:
            return None
        return fxp_arr.data.astype(np.float32) / (1 << fxp_arr.exp)
    
    # Compare batch norm
    if 'norm_output' in full_results['bn_intermediates']:
        bn_full = fxp_to_float(full_results['bn_intermediates']['norm_output'])
        bn_simple = simple_results['after_batch_norm'].to_float()
        
        print(f"Batch Norm Comparison:")
        print(f"  Full model (first 5): {bn_full[:5, 0]}")
        print(f"  Simple model (first 5): {bn_simple[:5, 0]}")
        bn_error = np.abs(bn_full - bn_simple)
        print(f"  Error - Mean: {np.mean(bn_error):.6f}, Max: {np.max(bn_error):.6f}, RMS: {np.sqrt(np.mean(bn_error**2)):.6f}")
    
    # Compare SSM output
    if 'ys' in full_results['ssm_intermediates']:
        ssm_full = fxp_to_float(full_results['ssm_intermediates']['ys'])
        ssm_simple = simple_results['ssm_output'].to_float()
        
        print(f"\nSSM Output Comparison:")
        print(f"  Full model (first 5): {ssm_full[:5, 0]}")
        print(f"  Simple model (first 5): {ssm_simple[:5, 0]}")
        ssm_error = np.abs(ssm_full - ssm_simple)
        print(f"  Error - Mean: {np.mean(ssm_error):.6f}, Max: {np.max(ssm_error):.6f}, RMS: {np.sqrt(np.mean(ssm_error**2)):.6f}")
    
    # Compare GLU output  
    if 'post_GLU' in full_results['layer0_intermediates']:
        glu_full = fxp_to_float(full_results['layer0_intermediates']['post_GLU'])
        glu_simple = simple_results['glu_output'].to_float()
        
        print(f"\nGLU Output Comparison:")
        print(f"  Full model (first 5): {glu_full[:5, 0]}")
        print(f"  Simple model (first 5): {glu_simple[:5, 0]}")
        glu_error = np.abs(glu_full - glu_simple)
        print(f"  Error - Mean: {np.mean(glu_error):.6f}, Max: {np.max(glu_error):.6f}, RMS: {np.sqrt(np.mean(glu_error**2)):.6f}")
    
    # Compare final output
    if 'output' in full_results['layer0_intermediates']:
        final_full = fxp_to_float(full_results['layer0_intermediates']['output'])
        final_simple = simple_results['final_output'].to_float()
        
        print(f"\nFinal Output Comparison:")
        print(f"  Full model (first 5): {final_full[:5, 0]}")
        print(f"  Simple model (first 5): {final_simple[:5, 0]}")
        final_error = np.abs(final_full - final_simple)
        print(f"  Error - Mean: {np.mean(final_error):.6f}, Max: {np.max(final_error):.6f}, RMS: {np.sqrt(np.mean(final_error**2)):.6f}")
        
        # Plot comparison
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(final_full[:100, 0], 'b-', label='Full Model', alpha=0.7)
        plt.plot(final_simple[:100, 0], 'r--', label='Simple Model', alpha=0.7)
        plt.title('Final Output Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(final_error[:100, 0], 'g-', label='Error', alpha=0.7)
        plt.title('Error (Full - Simple)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.hist(final_error.flatten(), bins=50, alpha=0.7)
        plt.title('Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('fxp_layer0_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return final_error


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
    
    # Get encoder output for simple implementation
    encoder_output_relu = full_results['encoder_intermediates']['encoder_output_relu']

    #######
    # new: calculate encoder output from scratch
    partial_fxp_from_fp = partial(
        fxp_from_fp, signed=True, round_mode=RoundingMode.ROUND
    )
    w = partial_fxp_from_fp(
        modeldict['encoder']['encoder']['kernel'],
        bits=fxp_qconfig['encoder']['w_bits'],
        exp=fxp_qconfig['encoder']['w_exp'],
        signed=True,
    )
    b = partial_fxp_from_fp(
        modeldict['encoder']['encoder']['bias'],
        bits=fxp_qconfig['encoder']['b_bits'],
        exp=fxp_qconfig['encoder']['b_exp'],
    )
    y1 = fxp_matmul(
        inputs,
        w, 
        result_exp=fxp_qconfig['encoder']['out_exp'],
        result_bits=fxp_qconfig['encoder']['out_bits'],
    )
    y2 = fxp_add(
        y1,
        b,
        result_exp=fxp_qconfig['encoder']['out_exp'],
        result_bits=fxp_qconfig['encoder']['out_bits'],
    )
    y = fxp_relu(y2)

    print(f"first encoder linear layer")
    avg_err = np.abs(y.data - encoder_output_relu.data).mean()
    print(f" avg. error between calculated and stored: {avg_err}")
    #######
    
    # Run simple FXP implementation
    simple_results = simple_fxp_first_encoder_layer(
        encoder_output_relu, 
        modeldict, 
        fxp_qconfig, 
        scope="simple_layer0"
    )
    
    # Compare results
    final_error = compare_fxp_results(simple_results, full_results)
    
    print(f"\n=== SUMMARY ===")
    if final_error is not None:
        print(f"Overall Error Statistics:")
        print(f"  Mean: {np.mean(final_error):.6f}")
        print(f"  Max: {np.max(final_error):.6f}")
        print(f"  RMS: {np.sqrt(np.mean(final_error**2)):.6f}")
        print(f"  Std: {np.std(final_error):.6f}")
    
    print("FXP simple implementation comparison completed!")


if __name__ == "__main__":
    main() 