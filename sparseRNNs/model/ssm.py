from functools import partial
from typing import Callable

import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal

from sparseRNNs.model.layers import relu_top_k_sparsity
from sparseRNNs.model.ssm_init import (init_CV, init_log_steps, init_VinvB,
                                       trunc_standard_normal)
from sparseRNNs.utils.quantization import (FakeQuant, FakeQuantComplex,
                                           QuantizationConfig,
                                           QuantizedOperations,
                                           _calculate_qparams, _quantdequant)


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using bilinear transform method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Lambda_bar = np.exp(Lambda * Delta)
    B_bar = (1 / Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


# Parallel scan operations
def quant_binary_operator(q_i, q_j, qhad_fns):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    TODO: work out if the un-quantized addition is okay.
    """
    qhad_aa, qhad_abu = qhad_fns
    A_i, b_i = q_i
    A_j, b_j = q_j
    # # return A_j * A_i, A_j * b_i + b_j
    # A_out = qhad_fn(A_j, A_i)
    # Bu_out = qhad_fn(A_j, b_i) + b_j
    A_out_re = qhad_aa(A_j.real, A_i.real) - qhad_aa(
        A_j.imag, A_i.imag
    )  # TODO(stevenabreu): quantize activations
    A_out_im = qhad_aa(A_j.real, A_i.imag) + qhad_aa(A_j.imag, A_i.real)
    A_out = A_out_re + 1j * A_out_im
    Bu_out_re = qhad_abu(A_j.real, b_i.real) - qhad_abu(A_j.imag, b_i.imag)
    Bu_out_im = qhad_abu(A_j.real, b_i.imag) + qhad_abu(A_j.imag, b_i.real)
    Bu_out = Bu_out_re + 1j * Bu_out_im + b_j
    return A_out, Bu_out


def build_apply_ssm(q_ops: QuantizedOperations, associative_scan=True) -> Callable:

    q_bin_op = jax.vmap(jax.jit(partial(quant_binary_operator, qhad_fns=q_ops.a_had)))

    def _apply_ssm(
        Lambda_bar,
        B_bar,
        C_tilde,
        input_sequence,
        conj_sym,
        bidirectional,
        relufication=False,
        B_bias=None,
        topk=1.0,
        approx_topk=False,
    ):
        """Compute the LxH output of discretized SSM given an LxH input.
            Args:
                Lambda_bar (complex64): discretized diagonal state matrix    (P,)
                B_bar      (complex64): discretized input matrix             (P, H)
                C_tilde    (complex64): output matrix                        (H, P)
                input_sequence (float32): input sequence of features         (L, H)
                conj_sym (bool):         whether conjugate symmetry is enforced
                bidirectional (bool):    whether bidirectional setup is used,
                                      Note for this case C_tilde will have 2P cols
            Returns:
                ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)

        TODO:
        - real/imag separation below makes training ~2x slower (quantizing one matrix only)
          - might also mess with quantization (un-quantized addition of real and imag parts)
        """
        Lambda_elements = Lambda_bar * np.ones(
            (input_sequence.shape[0], Lambda_bar.shape[0])
        )

        def b_dot(u):
            re = q_ops.b_dot(B_bar.real, u.real) - q_ops.b_dot(B_bar.imag, u.imag)
            im = q_ops.b_dot(B_bar.real, u.imag) + q_ops.b_dot(B_bar.imag, u.real)
            if B_bias is not None:
                re += B_bias.real
                im += B_bias.imag
            return re + 1j * im

        Bu_elements = jax.vmap(jax.jit(b_dot))(input_sequence)

        if associative_scan:
            _, xs = jax.lax.associative_scan(q_bin_op, (Lambda_elements, Bu_elements))
        else:
            # Define the step function for scan
            def step(carry, inputs):
                A_t, Bu_t = inputs
                x_prev = carry

                qhad = q_ops.a_had[1]

                x_prev_re = x_prev.real
                x_prev_im = x_prev.imag
                A_t_re = A_t.real
                A_t_im = A_t.imag

                Ax_re = qhad(A_t_re, x_prev_re) - qhad(A_t_im, x_prev_im)
                Ax_im = qhad(A_t_re, x_prev_im) + qhad(A_t_im, x_prev_re)
                Ax = Ax_re + 1j * Ax_im

                x_t = Ax + Bu_t

                return x_t, x_t

            initial_state = np.zeros_like(Bu_elements[0])
            _, xs = jax.lax.scan(step, initial_state, (Lambda_elements, Bu_elements))

        if relufication and topk < 1.0 and approx_topk:
            xs = relu_top_k_sparsity(xs, int(topk * xs.shape[-1]))
        elif relufication and topk < 1.0:
            raise NotImplementedError(
                "Top-k sparsity without approx_topk not implemented"
            )
        elif relufication:
            xs = jax.nn.relu(xs)

        if bidirectional:
            if associative_scan:
                _, xs2 = jax.lax.associative_scan(
                    q_bin_op, (Lambda_elements, Bu_elements), reverse=True
                )
            else:
                print(
                    "[WARNING] bidirectional non-associative scan doesn't make"
                    " sense..."
                )
                _, xs2 = jax.lax.scan(
                    step,
                    initial_state,
                    (Lambda_elements[::-1], Bu_elements[::-1]),
                )
                xs2 = xs2[::-1]  # Reverse the outputs back
            xs = np.concatenate((xs, xs2), axis=-1)

        def c_dot_real(x):
            return q_ops.c_dot(C_tilde.real, x.real) - q_ops.c_dot(C_tilde.imag, x.imag)

        if conj_sym:
            return jax.vmap(lambda x: 2 * c_dot_real(x))(xs), xs
        else:
            return jax.vmap(jax.jit(c_dot_real))(xs), xs

    return _apply_ssm  # NOTE: jitting this function breaks the bidirectional argument


def build_apply_ssm_static_quant(
    q_ops: QuantizedOperations, associative_scan=True
) -> Callable:

    def _apply_ssm(
        Lambda_bar,
        B_bar,
        C_tilde,
        input_sequence,
        conj_sym,
        bidirectional,
        relufication=False,
        B_bias=None,
        quant_fns=None,
        topk=None,
        approx_topk=None,
    ):
        """Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (L, H)
            conj_sym (bool):         whether conjugate symmetry is enforced
            bidirectional (bool):    whether bidirectional setup is used,
                                  Note for this case C_tilde will have 2P cols
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
        """
        (
            quant_A,
            quant_B,
            quant_C,
            quant_D,
            quant_xt,
            quant_ut,
            quant_yt,
            quant_But,
        ) = quant_fns

        quant_u_t = quant_ut(input_sequence)
        B_bar = quant_B(B_bar)
        Lambda_bar = quant_A(Lambda_bar)
        C_tilde = quant_C(C_tilde)

        Lambda_elements = Lambda_bar * np.ones(
            (input_sequence.shape[0], Lambda_bar.shape[0])
        )

        def b_dot(u):
            res = q_ops.b_dot(B_bar, u)
            if B_bias is not None:
                res = res + B_bias
            return res

        Bu_elements = jax.vmap(jax.jit(lambda u: b_dot(u)))(quant_u_t)
        Bu_elements = (quant_But)(Bu_elements)

        if associative_scan:
            raise NotImplementedError(
                "Associative scan not implemented for static quantization"
            )
        else:
            # extract the scale, then pass it into the scan function
            if quant_xt.quant_real.has_variable("batch_stats", "observer"):
                # jax.debug.print("using scales inside recurrent forward")
                # real scale
                observer_dict = quant_xt.quant_real.get_variable(
                    "batch_stats", "observer"
                )
                minval = observer_dict["observer_min"]
                maxval = observer_dict["observer_max"]
                kwargs = dict(
                    pow2scale=quant_xt.quant_real.pow2scale,
                    qscheme=quant_xt.quant_real.qscheme,
                )
                real_scale, _ = _calculate_qparams(
                    minval, maxval, quant_xt.quant_real.bits, **kwargs
                )
                # imag scale
                observer_dict = quant_xt.quant_imag.get_variable(
                    "batch_stats", "observer"
                )
                minval = observer_dict["observer_min"]
                maxval = observer_dict["observer_max"]
                kwargs = dict(
                    pow2scale=quant_xt.quant_imag.pow2scale,
                    qscheme=quant_xt.quant_imag.qscheme,
                )
                imag_scale, _ = _calculate_qparams(
                    minval, maxval, quant_xt.quant_imag.bits, **kwargs
                )
            else:
                # jax.debug.print("NOT using scales inside recurrent forward")
                real_scale, imag_scale = None, None

            # Define the step function for scan
            def step(carry, inputs):
                A_t, Bu_t = inputs
                x_prev = carry

                qhad = q_ops.a_had[1]
                Ax = qhad(A_t, x_prev)

                x_t = Ax + Bu_t
                if real_scale is not None and imag_scale is not None:
                    x_tre = _quantdequant(
                        x_t.real,
                        scale=real_scale,
                        zero_point=0.0,
                        precision=quant_xt.quant_real.bits,
                    )
                    x_tim = _quantdequant(
                        x_t.imag,
                        scale=imag_scale,
                        zero_point=0.0,
                        precision=quant_xt.quant_imag.bits,
                    )
                    x_t = x_tre + 1j * x_tim

                return x_t, x_t

            initial_state = np.zeros_like(Bu_elements[0])
            _, xs = jax.lax.scan(step, initial_state, (Lambda_elements, Bu_elements))
            quant_xt(xs)  # NOTE: this is for the observers, x_t is already quantized
            # xs_q = quant_xt(xs)
            # def debug_print(xs, xs_q, atol):
            #     close = np.all(np.allclose(xs, xs_q, atol=atol))
            #     if not close:
            #         print(f"xs==quant(xs): {close} ({atol})")
            # jax.debug.callback(debug_print, xs, xs_q, 1e-6)
            # jax.debug.callback(debug_print, xs, xs_q, 1e-4)
            # jax.debug.callback(debug_print, xs, xs_q, 1e-2)

        if relufication:
            xs = jax.nn.relu(xs)

        if bidirectional:
            raise NotImplementedError(
                "Bidirectional not implemented for static quantization"
            )

        if conj_sym:
            return (
                jax.vmap(lambda x: 2 * q_ops.c_dot(C_tilde, x).real)(xs),
                xs,
                quant_u_t,
            )
        else:
            return (
                jax.vmap(lambda x: q_ops.c_dot(C_tilde, x).real)(xs),
                xs,
                quant_u_t,
            )

    return _apply_ssm  # NOTE: jitting this function breaks the bidirectional argument


class qS5SSM(nn.Module):
    Lambda_re_init: jax.Array
    Lambda_im_init: jax.Array
    V: jax.Array
    Vinv: jax.Array
    H: int
    P: int
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = False
    bidirectional: bool = False
    step_rescale: float = 1.0
    relufication: bool = False
    q_config: QuantizationConfig = QuantizationConfig.none()
    associative_scan: bool = True
    topk: float = 1.0
    approx_topk: bool = False

    """ The S5 SSM
        Args:
            Lambda_re_init (complex64): Real part of init diag state matrix  (P,)
            Lambda_im_init (complex64): Imag part of init diag state matrix  (P,)
            V           (complex64): Eigenvectors used for init           (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init   (P,P)
            H           (int32):     Number of features of input seq
            P           (int32):     state size
            C_init      (string):    Specifies How C is initialized
                         Options: [trunc_standard_normal: sample from truncated standard normal
                                                        and then multiply by V, i.e. C_tilde=CV.
                                   lecun_normal: sample from Lecun_normal and then multiply by V.
                                   complex_normal: directly sample a complex valued output matrix
                                                    from standard normal, does not multiply by V]
            conj_sym    (bool):    Whether conjugate symmetry is enforced
            clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                   constrain real part of eigenvalues to be negative.
                                   True recommended for autoregressive task/unbounded sequence
                                   lengths. Discussed in https://arxiv.org/pdf/2206.11893.pdf.
            bidirectional (bool):  Whether model is bidirectional, if True, uses two C matrices
            discretization: (string) Specifies discretization method
                             options: [zoh: zero-order hold method,
                                       bilinear: bilinear transform]
            dt_min:      (float32): minimum value to draw timescale values from when
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when
                                    initializing log_step
            step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g.
                                    after training on a different resolution for the speech
                                    commands benchmark
            q_config:    (QuantizationConfig): Configuration for quantization.
    """

    def setup(self):
        """Initializes parameters once and performs discretization each time
        the SSM is applied to a sequence
        """

        self.q_ops = QuantizedOperations(self.q_config)
        if self.q_config.static_quant:
            kwargs = dict(pow2scale=True, calibrating=self.q_config.calibrating)
            self.quant_A = FakeQuantComplex(bits=self.q_config.a_precision, **kwargs)
            self.quant_xt = FakeQuantComplex(
                bits=self.q_config.ssm_act_precision, **kwargs
            )
            self.quant_B = FakeQuantComplex(bits=self.q_config.b_precision, **kwargs)
            self.quant_ut = FakeQuant(bits=self.q_config.ssm_act_precision, **kwargs)
            self.quant_But = FakeQuantComplex(
                bits=self.q_config.ssm_act_precision, **kwargs
            )
            self.quant_C = FakeQuantComplex(bits=self.q_config.c_precision, **kwargs)
            self.quant_D = FakeQuant(bits=self.q_config.d_precision, **kwargs)
            self.quant_yt = FakeQuant(bits=self.q_config.ssm_act_precision, **kwargs)
            self.apply_ssm = build_apply_ssm_static_quant(
                self.q_ops, self.associative_scan
            )
        else:
            self.apply_ssm = build_apply_ssm(self.q_ops, self.associative_scan)

        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            local_P = 2 * self.P
        else:
            local_P = self.P

        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = self.param(
            "Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,)
        )
        self.Lambda_im = self.param(
            "Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,)
        )
        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        # Initialize input to state (B) matrix
        B_init = lecun_normal()
        B_shape = (local_P, self.H)
        self.B = self.param(
            "B",
            lambda rng, shape: init_VinvB(B_init, rng, shape, self.Vinv),
            B_shape,
        )
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5**0.5)
        else:
            raise NotImplementedError(
                "C_init method {} not implemented".format(self.C_init)
            )

        if self.C_init in ["complex_normal"]:
            if self.bidirectional:
                C = self.param("C", C_init, (self.H, 2 * self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

            else:
                C = self.param("C", C_init, (self.H, self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

        else:
            if self.bidirectional:
                self.C1 = self.param(
                    "C1",
                    lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                    C_shape,
                )
                self.C2 = self.param(
                    "C2",
                    lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                    C_shape,
                )

                C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
                C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
                self.C_tilde = np.concatenate((C1, C2), axis=-1)

            else:
                self.C = self.param(
                    "C",
                    lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                    C_shape,
                )

                self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        # Initialize feedthrough (D) matrix
        self.D = self.param("D", normal(stddev=1.0), (self.H,))

        # Initialize learnable discretization timescale value
        self.log_step = self.param(
            "log_step", init_log_steps, (self.P, self.dt_min, self.dt_max)
        )
        step = self.step_rescale * np.exp(self.log_step[:, 0])

        # Discretize
        if self.discretization in ["zoh"]:
            self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
        elif self.discretization in ["bilinear"]:
            self.Lambda_bar, self.B_bar = discretize_bilinear(
                self.Lambda, B_tilde, step
            )
        else:
            raise NotImplementedError(f"Discretization method {self.discretization}")

    def __call__(
        self,
        input_sequence,
        bn_mean=None,
        bn_var=None,
        bn_eps=None,
        bn_scale=None,
        bn_bias=None,
    ):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (L, H)
        Returns:
            output sequence (float32): (L, H)
        """
        B_bar, D, D_bias = None, None, None
        if (
            bn_mean is not None
            and bn_var is not None
            and bn_eps is not None
            and bn_scale is not None
            and bn_bias is not None
        ):
            scale = bn_scale / np.sqrt(bn_var + bn_eps)
            bias = bn_bias - bn_mean * scale
            B_bar = self.B_bar * scale
            B_bias = self.B_bar @ bias
            D = self.D * scale
            D_bias = self.D * bias
        else:
            B_bar = self.B_bar
            B_bias = None
            D = self.D
            D_bias = None

        self.sow("intermediates", "Lambda_bar", self.Lambda_bar)
        self.sow("intermediates", "B_bar", B_bar)
        self.sow("intermediates", "C_tilde", self.C_tilde)
        self.sow("intermediates", "D", D)
        if D_bias is not None:
            self.sow("intermediates", "D_bias", D_bias)
        if B_bias is not None:
            self.sow("intermediates", "B_bias", B_bias)

        if self.q_config.static_quant:
            ys, xs, qu_t = self.apply_ssm(
                Lambda_bar=self.Lambda_bar,
                B_bar=self.B_bar,
                C_tilde=self.C_tilde,
                input_sequence=input_sequence,
                conj_sym=self.conj_sym,
                bidirectional=self.bidirectional,
                relufication=self.relufication,
                topk=self.topk,
                approx_topk=self.approx_topk,
                B_bias=B_bias,
                quant_fns=(
                    self.quant_A,
                    self.quant_B,
                    self.quant_C,
                    self.quant_D,
                    self.quant_xt,
                    self.quant_ut,
                    self.quant_yt,
                    self.quant_But,
                ),
            )
            # Add feedthrough matrix output Du
            quant_D = self.quant_D(self.D)
            Du = jax.vmap(lambda u: self.q_ops.d_had(quant_D, u))(qu_t)
            ysDu = ys + Du
            ysDu = self.quant_yt(ysDu)
            if D_bias is not None:
                ysDu = ysDu + D_bias
            return ysDu, xs

        else:
            ys, xs = self.apply_ssm(
                Lambda_bar=self.Lambda_bar,
                B_bar=B_bar,
                C_tilde=self.C_tilde,
                input_sequence=input_sequence,
                conj_sym=self.conj_sym,
                bidirectional=self.bidirectional,
                relufication=self.relufication,
                topk=self.topk,
                approx_topk=self.approx_topk,
                B_bias=B_bias,
            )
            # Add feedthrough matrix output Du;
            # self.D * u can be replaced with the quant vector product einsum now.
            Du = jax.vmap(lambda u: self.q_ops.d_had(D, u))(input_sequence)
            if D_bias is None:
                return ys + Du, xs
            else:
                return ys + Du + D_bias, xs


def init_qS5SSM(
    H,
    P,
    Lambda_re_init,
    Lambda_im_init,
    V,
    Vinv,
    C_init,
    discretization,
    dt_min,
    dt_max,
    conj_sym,
    clip_eigs,
    bidirectional,
    relufication,
    q_config,
    associative_scan=True,
):
    """Convenience function that will be used to initialize the SSM.
    Same arguments as defined in S5SSM above."""
    return partial(
        qS5SSM,
        H=H,
        P=P,
        Lambda_re_init=Lambda_re_init,
        Lambda_im_init=Lambda_im_init,
        V=V,
        Vinv=Vinv,
        C_init=C_init,
        discretization=discretization,
        dt_min=dt_min,
        dt_max=dt_max,
        conj_sym=conj_sym,
        clip_eigs=clip_eigs,
        bidirectional=bidirectional,
        relufication=relufication,
        q_config=q_config,
        associative_scan=associative_scan,
    )
