import itertools
import subprocess
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import numba
import numpy as np
import torch
import torch.nn.functional as F

from audio import HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND
from tokenizer import Tokenizer


def median_filter(x: torch.Tensor, filter_width: int):
    """Apply a median filter of width `filter_width` along the last dimension of `x`"""
    pad_width = filter_width // 2
    if x.shape[-1] <= pad_width:
        # F.pad requires the padding width to be smaller than the input dimension
        return x

    if (ndim := x.ndim) <= 2:
        # `F.pad` does not support 1D or 2D inputs for reflect padding but supports 3D and 4D
        x = x[None, None, :]

    assert (
        filter_width > 0 and filter_width % 2 == 1
    ), "`filter_width` should be an odd number"

    result = None
    x = F.pad(x, (filter_width // 2, filter_width // 2, 0, 0), mode="reflect")
    if x.is_cuda:
        try:
            from triton_ops import median_filter_cuda

            result = median_filter_cuda(x, filter_width)
        except (RuntimeError, subprocess.CalledProcessError):
            warnings.warn(
                "Failed to launch Triton kernels, likely due to missing CUDA toolkit; "
                "falling back to a slower median kernel implementation..."
            )

    if result is None:
        # sort() is faster than torch.median (https://github.com/pytorch/pytorch/issues/51450)
        result = x.unfold(-1, filter_width, 1).sort()[0][..., filter_width // 2]

    if ndim <= 2:
        result = result[0, 0]

    return result


@numba.jit(nopython=True)
def backtrace(trace: np.ndarray):
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    trace[0, :] = 2
    trace[:, 0] = 1

    result = []
    while i > 0 or j > 0:
        result.append((i - 1, j - 1))

        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        elif trace[i, j] == 2:
            j -= 1
        else:
            raise ValueError("Unexpected trace[i, j]")

    result = np.array(result)
    return result[::-1, :].T


@numba.jit(nopython=True, parallel=True)
def dtw_cpu(x: np.ndarray):
    N, M = x.shape
    cost = np.ones((N + 1, M + 1), dtype=np.float32) * np.inf
    trace = -np.ones((N + 1, M + 1), dtype=np.float32)

    cost[0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]

            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2

            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t

    return backtrace(trace)


def dtw_cuda(x, BLOCK_SIZE=1024):
    from triton_ops import dtw_kernel

    M, N = x.shape
    assert M < BLOCK_SIZE, f"M should be smaller than {BLOCK_SIZE=}"

    x_skew = (
        F.pad(x, (0, M + 1), value=np.inf).flatten()[: M * (N + M)].reshape(M, N + M)
    )
    x_skew = x_skew.T.contiguous()
    cost = torch.ones(N + M + 2, M + 2) * np.inf
    cost[0, 0] = 0
    cost = cost.cuda()
    trace = torch.zeros_like(cost, dtype=torch.int32)

    dtw_kernel[(1,)](
        cost,
        trace,
        x_skew,
        x_skew.stride(0),
        cost.stride(0),
        trace.stride(0),
        N,
        M,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    trace = trace.T.flatten()[: (M + 1) * (M + N + 3)].reshape(M + 1, M + N + 3)[
        :, : N + 1
    ]
    return backtrace(trace.cpu().numpy())


def dtw(x: torch.Tensor) -> np.ndarray:
    if x.is_cuda:
        try:
            return dtw_cuda(x)
        except (RuntimeError, subprocess.CalledProcessError):
            warnings.warn(
                "Failed to launch Triton kernels, likely due to missing CUDA toolkit; "
                "falling back to a slower DTW implementation..."
            )

    return dtw_cpu(x.double().cpu().numpy())


@dataclass
class WordTiming:
    word: str
    tokens: List[int]
    start: float
    end: float
    probability: float


###################################################################################
# TODO: Uncomment once re-designed export is completed since encoder
# and decoder-init will be separate inference passes (e.g. model.encoder(...)
# and model.logits(...) instead of just model(...) as it currently is)

# def find_alignment(
#     model: "Whisper",
#     tokenizer: Tokenizer,
#     text_tokens: List[int],
#     mel: torch.Tensor,
#     num_frames: int,
#     *,
#     medfilt_width: int = 7,
#     qk_scale: float = 1.0,
# ) -> List[WordTiming]:
#     if len(text_tokens) == 0:
#         return []

#     tokens = torch.tensor(
#         [
#             *tokenizer.sot_sequence,
#             tokenizer.no_timestamps,
#             *text_tokens,
#             tokenizer.eot,
#         ]
#     ).to(model.device)

#     # install hooks on the cross attention layers to retrieve the attention weights
#     QKs = [None] * model.dims.n_text_layer
#     hooks = [
#         block.cross_attn.register_forward_hook(
#             lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1][0])
#         )
#         for i, block in enumerate(model.decoder.blocks)
#     ]

#     with torch.no_grad():
#         logits = model(mel.unsqueeze(0), tokens.unsqueeze(0))[0]
#         sampled_logits = logits[len(tokenizer.sot_sequence) :, : tokenizer.eot]
#         token_probs = sampled_logits.softmax(dim=-1)
#         text_token_probs = token_probs[np.arange(len(text_tokens)), text_tokens]
#         text_token_probs = text_token_probs.tolist()

#     for hook in hooks:
#         hook.remove()

#     # heads * tokens * frames
#     weights = torch.stack([QKs[_l][_h] for _l, _h in model.alignment_heads.indices().T])
#     weights = weights[:, :, : num_frames // 2]
#     weights = (weights * qk_scale).softmax(dim=-1)
#     std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
#     weights = (weights - mean) / std
#     weights = median_filter(weights, medfilt_width)

#     matrix = weights.mean(axis=0)
#     matrix = matrix[len(tokenizer.sot_sequence) : -1]
#     text_indices, time_indices = dtw(-matrix)

#     words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot])
#     if len(word_tokens) <= 1:
#         # return on eot only
#         # >>> np.pad([], (1, 0))
#         # array([0.])
#         # This results in crashes when we lookup jump_times with float, like
#         # IndexError: arrays used as indices must be of integer (or boolean) type
#         return []
#     word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))

#     jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
#     jump_times = time_indices[jumps] / TOKENS_PER_SECOND
#     start_times = jump_times[word_boundaries[:-1]]
#     end_times = jump_times[word_boundaries[1:]]
#     word_probabilities = [
#         np.mean(text_token_probs[i:j])
#         for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
#     ]

#     return [
#         WordTiming(word, tokens, start, end, probability)
#         for word, tokens, start, end, probability in zip(
#             words, word_tokens, start_times, end_times, word_probabilities
#         )
#     ]
###################################################################################


###################################################################################
# TODO: Comment once re-designed export is completed since encoder
# and decoder-init will be separate inference passes (e.g. model.encoder(...)
# and model.logits(...) instead of just model(...) as it currently is)

def find_alignment(
    model: "WhisperONNX",
    tokenizer: Tokenizer,
    text_tokens: List[int],
    mel: torch.Tensor,
    num_frames: int,
    *,
    medfilt_width: int = 7,
    qk_scale: float = 1.0,
) -> List[WordTiming]:
    if len(text_tokens) == 0:
        return []

    tokens = torch.tensor(
        [
            *tokenizer.sot_sequence,
            tokenizer.no_timestamps,
            *text_tokens,
            tokenizer.eot,
        ]
    ).to(model.device)

    # get final combined cross QK tensor from generation loop
    # print("get cross qk")
    # QKs = torch.from_numpy(model.generator.get_output("cross_qk")).to(torch.float32)
    # print(QKs.shape)
    # import pdb; pdb.set_trace()

    # re-run inference to get logits for all tokens ([batch_size, prompt_length + decoded_length, vocab_size]),
    # not just the logits for the last token ([batch_size, 1, vocab_size])
    with torch.no_grad():
        logits = model(mel.unsqueeze(0), tokens.unsqueeze(0))[0]
        sampled_logits = logits[len(tokenizer.sot_sequence) :, : tokenizer.eot]
        token_probs = sampled_logits.softmax(dim=-1)
        text_token_probs = token_probs[np.arange(len(text_tokens)), text_tokens]
        text_token_probs = text_token_probs.tolist()

    # text_token_probs:
    # PT: [0.7890558242, 0.152467235922, 0.63745307922, 0.97963619232, 0.998051166, 0.51488977670, 0.97223603, 0.99275922775, 0.7263591, 0.97823899984, 0.853136003, 0.99160039424, 0.99617552, 0.54236465692, 0.5426307320, 0.346207231283, 0.9500872492, 0.701448976, 0.98865228891, 0.4968730807, 0.66240823268, 0.9679544568, 0.9207713007, 0.97268670797, 0.9972301125, 0.94439017772, 0.73084712028, 0.9990655779, 0.985873699, 0.99188846349, 0.68298316001, 0.4589944481, 0.90137696266, 0.464057654, 0.62091094255, 0.883975207, 0.99762409925, 0.99066585302, 0.9970781803, 0.96405315399]
    # ORT: [0.7939453125, 0.155517578125, 0.63525390625, 0.97900390625, 0.998046875, 0.51904296875, 0.97265625, 0.99267578125, 0.7265625, 0.97900390625, 0.853515625, 0.99169921875, 0.99609375, 0.54248046875, 0.5419921875, 0.350830078125, 0.9501953125, 0.701171875, 0.98876953125, 0.4931640625, 0.66259765625, 0.9677734375, 0.9208984375, 0.97314453125, 0.9970703125, 0.94287109375, 0.73583984375, 0.9990234375, 0.986328125, 0.99169921875, 0.67822265625, 0.4599609375, 0.90087890625, 0.462890625, 0.61767578125, 0.880859375, 0.99755859375, 0.99072265625, 0.9970703125, 0.96435546875]

    # heads * tokens * frames
    QKs = [torch.from_numpy(model.generator.get_output(f"output_cross_qk_{idx}")).to(dtype=mel.dtype, device=mel.device)[0] for idx in range(model.dims.n_text_layer)]
    weights = torch.stack([QKs[_l][:, _h] for _l, _h in model.alignment_heads.T])
    print(weights.shape)
    weights = weights[:, :, : num_frames // 2]
    print("QKs:")
    print(weights.shape)

    # PyTorch:
    # tensor([50258, 50259, 50359, 50363,   440,  1723,   322,   702,  7443,   920,
    #         37460,  3390,    13,   440, 29677,   295,   702, 48834,  7146,   274,
    #          2346,    13,  2754,   264,   370,  1921, 18451,   926,   796,   365,
    #          5383,   295,  6177,  3391,    11, 19817,  3337,   327,   530,   406,
    #          3163,  1953,   466,    13, 50257], device='cuda:0')

    # QKs:
    # tensor([[[ 1.1777,  0.0782, -1.5498,  ..., -2.4316, -2.9199, -0.6792],
    #          [ 2.0430, -1.0488, -2.4844,  ..., -4.3906, -4.1133, -2.6738],
    #          [ 2.2246, -0.4211, -1.8643,  ..., -5.1094, -4.8594, -2.7676],
    #          ...,
    #          [ 3.1504, -4.3164, -7.3164,  ...,  1.9307,  3.2363,  1.4521],
    #          [ 2.4844, -4.4336, -7.1719,  ..., -1.9521, -0.9497, -1.2344],
    #          [ 4.9727, -1.6006, -4.2578,  ..., -3.8125, -2.5645, -4.1172]],

    #         [[-1.0010,  3.3086,  0.9131,  ..., -1.2383, -1.3330, -0.1197],
    #          [ 0.8633,  1.4863,  0.1995,  ..., -0.3755, -0.4131, -0.7261],
    #          [ 4.9805,  4.5898,  4.1094,  ..., -2.7852, -2.1211, -1.6387],
    #          ...,
    #          [ 3.7031, -1.8525, -3.2754,  ...,  3.7930,  3.7871,  3.9590],
    #          [ 3.0312, -2.1719, -4.3203,  ..., -0.5679, -0.1044, -0.6982],
    #          [ 4.8008, -0.1677, -1.8711,  ..., -1.2920, -0.9058, -1.5527]],

    #         [[ 0.4844,  1.6309,  0.4387,  ..., -0.9902, -0.1098, -2.0234],
    #          [-0.3384,  2.0117,  1.5996,  ..., -0.6851,  0.1095, -1.4854],
    #          [ 3.5840,  6.3320,  6.4141,  ..., -3.0352, -2.6680, -1.0938],
    #          ...,
    #          [ 2.9688, -2.0742, -4.0391,  ..., -0.5249,  0.4822,  0.0187],
    #          [ 3.6387, -1.3105, -3.5820,  ..., -0.8745,  0.2269, -3.1777],
    #          [ 4.4141,  0.1512, -1.4727,  ..., -2.0020, -1.1279, -3.4688]],

    #         [[ 2.4121,  1.4854,  1.4805,  ...,  2.2285,  2.7734,  2.3164],
    #          [ 0.9175, -1.0156, -1.3838,  ..., -0.9097, -0.6655,  0.1354],
    #          [ 4.3750,  1.8184,  0.7529,  ..., -0.6963, -0.4099, -0.7148],
    #          ...,
    #          [ 3.1973, -2.6055, -5.0234,  ...,  1.4160,  1.9092, -0.0086],
    #          [ 3.9688, -2.2637, -4.5312,  ...,  0.3770,  1.3184, -1.1504],
    #          [ 3.6348, -1.2402, -3.1777,  ..., -0.9414, -0.2717, -1.6162]],

    #         [[ 0.9214,  2.3477, -0.4678,  ..., -1.3418, -1.4512, -0.9336],
    #          [ 2.6758,  0.0515, -1.8555,  ..., -2.8984, -3.0332, -0.1675],
    #          [ 2.8926,  1.9434,  0.9849,  ..., -0.5405, -0.6265,  3.4316],
    #          ...,
    #          [ 3.5215, -2.0840, -4.3633,  ...,  1.1279,  1.9229,  0.6240],
    #          [ 2.8281, -2.2031, -4.1484,  ..., -0.3696,  0.6426, -1.3926],
    #          [ 3.2070, -1.0869, -2.6562,  ..., -1.6631, -0.9785, -1.8320]],

    #         [[-0.6406,  2.1816,  0.0204,  ..., -0.7866, -0.5420, -1.6582],
    #          [-3.0332,  0.0662, -0.6006,  ...,  0.0490,  0.9985, -1.1973],
    #          [ 4.1992,  2.2090,  0.9443,  ...,  1.4189,  2.2109,  0.4099],
    #          ...,
    #          [ 1.3389, -2.7812, -5.1094,  ...,  0.8599,  1.2988, -1.0244],
    #          [ 3.3438, -1.8096, -4.0234,  ..., -0.9536, -0.3896, -2.4258],
    #          [ 3.2227, -0.8682, -2.6602,  ..., -1.7021, -1.4590, -2.1758]]],
    #        device='cuda:0')

    # ORT:
    # tensor([50258, 50259, 50359, 50363,   440,  1723,   322,   702,  7443,   920,
    #         37460,  3390,    13,   440, 29677,   295,   702, 48834,  7146,  2575,
    #            13,  2754,   264,   370,  1921, 18451,   926,   796,   365,  5383,
    #           295,  6177,  3391,    11, 19817,  3337,   327,   530,   406,  3163,
    #          1953,   466,    13, 50257], device='cuda:0')

    # tensor([[[ 1.2432,  0.2703, -1.4912,  ..., -1.7402, -2.4121, -1.9043],
    #          [ 2.0645, -0.6611, -2.4023,  ..., -4.2695, -4.1562, -4.2422],
    #          [ 2.2031, -0.1981, -1.9922,  ..., -4.7656, -4.9648, -4.5859],
    #          ...,
    #          [ 3.3438, -3.4766, -7.2852,  ...,  2.4297,  3.5000,  1.3330],
    #          [ 2.2871, -4.3945, -8.0312,  ..., -0.9526, -0.0615, -1.5850],
    #          [ 4.2344, -1.0947, -4.0078,  ..., -3.2383, -1.9521, -4.0312]],

    #         [[-0.8931,  3.3887,  1.2354,  ..., -1.3662, -1.4238, -1.8359],
    #          [ 0.8721,  1.5938,  0.3318,  ..., -0.5068, -0.2008, -1.4170],
    #          [ 4.8945,  4.5273,  4.3164,  ..., -3.1250, -2.5508, -3.0098],
    #          ...,
    #          [ 4.0469, -1.4316, -2.9102,  ...,  3.8887,  3.7891,  3.6582],
    #          [ 2.5996, -2.8164, -5.5664,  ...,  0.7778,  1.2295,  0.3623],
    #          [ 4.7578, -0.4927, -2.0566,  ...,  0.4492,  0.8394,  0.2379]],

    #         [[ 0.5532,  1.9443,  0.8188,  ..., -1.5166, -0.6265, -2.9492],
    #          [-0.3477,  2.2070,  1.8701,  ..., -0.2133,  0.6250, -1.3779],
    #          [ 3.5117,  5.9844,  6.6680,  ..., -3.1348, -2.7930, -2.7051],
    #          ...,
    #          [ 2.9922, -1.6172, -3.8008,  ...,  0.2852,  1.0361,  0.2549],
    #          [ 3.6152, -1.7598, -4.4492,  ..., -0.2615,  0.9014, -1.6680],
    #          [ 4.3555,  0.4302, -1.3115,  ..., -1.5078, -0.6729, -2.1133]],

    #         [[ 2.4902,  1.3838,  1.6064,  ...,  2.7910,  3.9395,  2.6094],
    #          [ 0.9106, -0.8340, -1.2686,  ...,  0.3396,  0.7808,  0.0413],
    #          [ 4.3633,  1.8223,  0.8877,  ..., -0.4829, -0.1075, -0.7446],
    #          ...,
    #          [ 3.7344, -1.9854, -4.5078,  ...,  2.0039,  1.9551,  0.8101],
    #          [ 3.5508, -2.9277, -5.5078,  ...,  0.6641,  1.4355, -0.3938],
    #          [ 3.1289, -0.8574, -3.1035,  ..., -0.6035, -0.4878, -1.1143]],

    #         [[ 0.9639,  2.4160, -0.3503,  ..., -1.2129, -1.5088, -1.7734],
    #          [ 2.6875,  0.4043, -1.7959,  ..., -2.2949, -2.8320, -1.7363],
    #          [ 2.9902,  1.9062,  1.1396,  ..., -1.1934, -1.8115,  0.8330],
    #          ...,
    #          [ 4.0117, -1.7305, -4.2422,  ...,  1.4492,  2.1309,  0.7500],
    #          [ 2.8750, -2.6152, -4.9922,  ...,  0.4287,  1.4951, -0.3088],
    #          [ 2.9668, -0.8701, -2.8730,  ..., -1.0234, -0.6963, -1.0225]],

    #         [[-0.4199,  2.3652,  0.2084,  ...,  0.4448,  0.9570, -1.3213],
    #          [-3.0098,  0.2245, -0.4065,  ...,  1.6475,  3.4238, -0.1981],
    #          [ 4.3281,  2.2090,  1.1680,  ...,  2.4863,  3.6797,  1.9902],
    #          ...,
    #          [ 2.0410, -2.1797, -4.7695,  ...,  0.9595,  1.2695, -0.3765],
    #          [ 4.1211, -2.1406, -4.9062,  ..., -0.0718,  0.4983, -1.1943],
    #          [ 2.9297, -0.9849, -3.2871,  ..., -1.0898, -1.1885, -1.6436]]],
    #        device='cuda:0', dtype=torch.float16)

    # weights = QKs.reshape([*QKs.shape[2:]])
    # print("QKs:")
    # print(weights)
    # weights = torch.from_numpy(np.load("/home/kvaishnavi/whisper/cross_qk_weights.npy"))
    weights = (weights * qk_scale).softmax(dim=-1)
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    weights = (weights - mean) / std
    weights = median_filter(weights, medfilt_width)

    matrix = weights.mean(axis=0)
    matrix = matrix[:, len(tokenizer.sot_sequence) : -1]
    text_indices, time_indices = dtw(-matrix)

    words, word_tokens = tokenizer.split_to_word_tokens(text_tokens + [tokenizer.eot])
    if len(word_tokens) <= 1:
        # return on eot only
        # >>> np.pad([], (1, 0))
        # array([0.])
        # This results in crashes when we lookup jump_times with float, like
        # IndexError: arrays used as indices must be of integer (or boolean) type
        return []
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))

    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps] / TOKENS_PER_SECOND
    start_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]
    word_probabilities = [
        np.mean(text_token_probs[i:j])
        for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
    ]

    return [
        WordTiming(word, tokens, start, end, probability)
        for word, tokens, start, end, probability in zip(
            words, word_tokens, start_times, end_times, word_probabilities
        )
    ]
###################################################################################


def merge_punctuations(alignment: List[WordTiming], prepended: str, appended: str):
    # merge prepended punctuations
    i = len(alignment) - 2
    j = len(alignment) - 1
    while i >= 0:
        previous = alignment[i]
        following = alignment[j]
        if previous.word.startswith(" ") and previous.word.strip() in prepended:
            # prepend it to the following word
            following.word = previous.word + following.word
            following.tokens = previous.tokens + following.tokens
            previous.word = ""
            previous.tokens = []
        else:
            j = i
        i -= 1

    # merge appended punctuations
    i = 0
    j = 1
    while j < len(alignment):
        previous = alignment[i]
        following = alignment[j]
        if not previous.word.endswith(" ") and following.word in appended:
            # append it to the previous word
            previous.word = previous.word + following.word
            previous.tokens = previous.tokens + following.tokens
            following.word = ""
            following.tokens = []
        else:
            i = j
        j += 1


def add_word_timestamps(
    *,
    segments: List[dict],
    model: "WhisperONNX",
    tokenizer: Tokenizer,
    mel: torch.Tensor,
    num_frames: int,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    last_speech_timestamp: float,
    **kwargs,
):
    # all_tokens_per_segment = []
    # for segment in segments:
    #     for token in segment["tokens"]:
    #         all_tokens_per_segment.append(token)
    # print(tokenizer.decode_with_timestamps(all_tokens_per_segment))
    # import pdb; pdb.set_trace()

    if len(segments) == 0:
        return

    text_tokens_per_segment = [
        [token for token in segment["tokens"] if token < tokenizer.eot]
        for segment in segments
    ]

    text_tokens = list(itertools.chain.from_iterable(text_tokens_per_segment))
    alignment = find_alignment(model, tokenizer, text_tokens, mel, num_frames, **kwargs)
    word_durations = np.array([t.end - t.start for t in alignment])
    word_durations = word_durations[word_durations.nonzero()]
    median_duration = np.median(word_durations) if len(word_durations) > 0 else 0.0
    median_duration = min(0.7, float(median_duration))
    max_duration = median_duration * 2

    # hack: truncate long words at sentence boundaries.
    # a better segmentation algorithm based on VAD should be able to replace this.
    if len(word_durations) > 0:
        sentence_end_marks = ".。!！?？"
        # ensure words at sentence boundaries are not longer than twice the median word duration.
        for i in range(1, len(alignment)):
            if alignment[i].end - alignment[i].start > max_duration:
                if alignment[i].word in sentence_end_marks:
                    alignment[i].end = alignment[i].start + max_duration
                elif alignment[i - 1].word in sentence_end_marks:
                    alignment[i].start = alignment[i].end - max_duration

    merge_punctuations(alignment, prepend_punctuations, append_punctuations)

    time_offset = segments[0]["seek"] * HOP_LENGTH / SAMPLE_RATE
    word_index = 0

    for segment, text_tokens in zip(segments, text_tokens_per_segment):
        saved_tokens = 0
        words = []

        while word_index < len(alignment) and saved_tokens < len(text_tokens):
            timing = alignment[word_index]

            if timing.word:
                words.append(
                    dict(
                        word=timing.word,
                        start=round(time_offset + timing.start, 2),
                        end=round(time_offset + timing.end, 2),
                        probability=timing.probability,
                    )
                )

            saved_tokens += len(timing.tokens)
            word_index += 1

        # hack: truncate long words at segment boundaries.
        # a better segmentation algorithm based on VAD should be able to replace this.
        if len(words) > 0:
            # ensure the first and second word after a pause is not longer than
            # twice the median word duration.
            if words[0]["end"] - last_speech_timestamp > median_duration * 4 and (
                words[0]["end"] - words[0]["start"] > max_duration
                or (
                    len(words) > 1
                    and words[1]["end"] - words[0]["start"] > max_duration * 2
                )
            ):
                if (
                    len(words) > 1
                    and words[1]["end"] - words[1]["start"] > max_duration
                ):
                    boundary = max(words[1]["end"] / 2, words[1]["end"] - max_duration)
                    words[0]["end"] = words[1]["start"] = boundary
                words[0]["start"] = max(0, words[0]["end"] - max_duration)

            # prefer the segment-level start timestamp if the first word is too long.
            if (
                segment["start"] < words[0]["end"]
                and segment["start"] - 0.5 > words[0]["start"]
            ):
                words[0]["start"] = max(
                    0, min(words[0]["end"] - median_duration, segment["start"])
                )
            else:
                segment["start"] = words[0]["start"]

            # prefer the segment-level end timestamp if the last word is too long.
            if (
                segment["end"] > words[-1]["start"]
                and segment["end"] + 0.5 < words[-1]["end"]
            ):
                words[-1]["end"] = max(
                    words[-1]["start"] + median_duration, segment["end"]
                )
            else:
                segment["end"] = words[-1]["end"]

            last_speech_timestamp = segment["end"]

        segment["words"] = words
