# Copied from https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge/blob/main/audio_dataloader.py

# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/


import glob
import os
import re
from typing import Any, Dict, Tuple

import numpy as np
import soundfile as sf
import torch


class DNSAudio:
    """Aduio dataset loader for DNS.

    Parameters
    ----------
    root : str, optional
        Path of the dataset location, by default './'.
    """

    def __init__(self, root: str = "./") -> None:
        self.root = root
        self.noisy_files = glob.glob(root + "noisy/**.wav")
        # self.noisy_files = self.noisy_files[:150] # uncomment this line for testing with a small subset of the N-DNS dataset
        self.file_id_from_name = re.compile("fileid_(\d+)")
        self.snr_from_name = re.compile("snr(-?\d+)")
        self.target_level_from_name = re.compile("tl(-?\d+)")
        self.source_info_from_name = re.compile("^(.*?)_snr")
        # self.root = root
        # files_ids = [1392,3359,6019,7727,11083,11295,4999,1635,5508,6874,3215,2232,4689,10459,7492,6661,6867,4563,214,7129,8933,4431,10006,2251,5448,3171,8974,9680,10662,5435,7817,6677,4987,3479,3845,3258,3250,11946,1204,4776,10949,3202,839,3273,6540,3227,5882,5720,10727,7871,11261,10522,3084,8994,1533,8553,11675,1195,7658,10599,1129,1233,6312,4899,9323,4075,5593,1314,4457,1648,10990,10892,3096,754,4608,6336,219,3571,10630,5883,4441,4347,10001,4155,7120,6642,11774,9344,11950,1319,4350,7283,201,6567,6238,5604,4519,6142,11484,2874,1557,7997,9262,3909,6335,4338,2407,9957,11506,6991,8259,11027,3858,6902,8300,5557,7488,9933,830,7501,10094,7836,7813,744,2339,1064,10384,5188,2795,6581,1556,5784,4825,11028,1209,1534,1428,6151,3557,10622,523,8699,2456,4749,10189,2320,8562,3821,8581,961,11543,2246,5263,11007,7614,8915,8492,4916,9831,5669,10993,6147,1068,4700,7636,1972,4806,2669,7053,11632,8110,3491,8865,1718,11613,3619,5221,10936,3399,20,3746,11580,9983,4837,4551,11071,2105,2078,7347,4065,9749,368,5033,67,7472,5650,3043,4707,3839,6696,1849,1458,6719,5154,9102,10975,5378,9676,5170,6815,9898,10008,2794,574,2430,8521,5797,7032,10543,7842,7669,3436,9145,2532,5995,369,119,8085,6979,5168,2262,5679,4874,6119,79,7850,9187,3649,3759,723,11759,7555,11265,4181,11758,10939,10700,286,3639,4205,11211,3087,4947,10538,7687,11312,10441,4808,4252,1805,5632,3211,2507,11469,8966,5673,3442,7299,203,3037,553,5392,6566,3304,9100,10966,7719,3708,8104,5541,596,10408,1762,1188,4468,8600,11095,9600,6518,6395,7998,6014,2505,5945,11249,3457,4287,4804,1008,3908,3390,6859,3325,143,7349,7786,2024,10565,475,2721,11827,7809,3102,7028,5866,6899,10208,1449,10896,4538,8685,7405,8744,3509,565,8541,787,10241,2400,2123,11601,5989,7003,4795,3026,1608,7022,11599,5929,11172,2265,4390,4517,4114,4171,10018,4862,7220,4059,10269,10598,2395,3599,8550,9752,7938,589,9845,11362,5994,1807,10363,8442,8715,1435,10425,1162,7756,7841,3875,2902,893,8474,3460,2303,7751,6030,7339,10699,4367,2981,9592,10850,5487,307,6086,263,7212,1011,9109,8250,4747,829,8483,6098,5656,384,7594,6649,10082,7597,9143,6780,11745,9083,3771,3572,224,10539,6782,9064,606,2074,595,7237,8943,5960,4701,8696,10230,3750,3905,11549,7781,4669,9131,2808,8241,10940,6799,11385,7676,10638,9011,8365,10163,11786,7354,10952,5103,11032,5099,706,10116,5402,10941,2308,2504,4934,7139,7899,739,5986,949,7633,7814,4244,11349,9491,9354,8020,5885,6945,10421,10999,8296,6036,4878,1950,5952,10837,9973,2954,868,9316,4729,5949,4663,2298,5680,1212,1993,11707,9149,9799,2641,3692,9790,5077,11541,8206,678,5817,10313,5907,1980,8499,5345,9621,9248,8909,4904,2548,3948,426,10130,4830,4542,3220,4055,972,11416,6981,4045,8856,5775,5638,8078,1523,7618,3025,5778,8167,3281,6883,8905,1102,2512,2920,7584,6942,9189,11975,8005,3208,7218,6871,2258,3306,7438,10647,366,7150,9014,7242,6219,3550,5305,11365,4557,9287,2647,4011,7012,6429,4709,1393,2598,3748,872,4871,597,11250,3676,7949,6160,9675,11629,11990,5714,1224,908,9474,11198,1656,4748,4652,10597,9048,1009,2063,8874,7586,2710,4794,7819,8722,5436,4892,10551,10033,2565,9095,1060,10474,7785,6350,9908,8073,3486,2547,3962,2560,1477,738,8022,9893,4177,1413,6655,4640,206,8011,7790,3788,4148,619,4031,7040,8418,9926,8137,11185,9823,2192,7267,2570,2068,9458,11423,1906,6332,11464,10205,2908,11096,1040,10346,5251,10909,7856,9688,3426,4117,3072,10271,2833,9467,6424,53,7414,5925,3654,6150,1117,1020,663,1053,640,2871,5928,11914,7673,6971,11165,10526,932,5415,6363,10128,361,3365,11960,6522,4500,10403,11847,9485,4282,1159,7861,5732,4985,9761,9004,2126,11316,8797,6000,742,6441,9051,1124,265,2091,7461,2506,38,8091,7991,8870,3235,4193,9958,7961,8291,7897,11389,10210,1610,6463,914,6194,6494,1288,2914,5531,11855,4684,5235,11751,1842,9602,336,3154,10247,8257,10913,4433,291,10756,6360,1485,1841,8227,11445,11374,3134,3036,3269,10012,5158,3937,2016,117,10407,5101,7893,2712,5916,6356,11588,8681,9853,7358,11869,9355,11865,9266,3329,10874,10788,11142,10795,5980,2453,6612,861,10194,855,1616,11313,3497,5853,11300,7869,3308,4840,9071,4362,6620,5993,4095,6323,827,3260,9067,10779,2381,4530,1041,7960,1132,1330,6069,10497,11382,5640,6444,5346,1955,11120,810,2916,2595,11473,6731,471,807,9401,7872,11119,4824,5932,10415,2346,10464,11422,7695,6116,1452,8115,5312,7680,11600,10280,3187,2576,6394,10059,6910,8325,2923,5979,9868,6181,6943,121,7193,2840,3140,5148,10229,8054,10826,3115,140,5074,5977,915,7446,9670,10215,8758,4283,4090,4842,4013,4482,9124,2412,9054,8052,5826,6286,11439,8888,10070,5683,6062,321,2990,724,5580,6897,5528,8414,956,89,2859,6107,4890,8390,11876,11328,7723,6110,6130,11246,8341,354,10165,1746,10213,1032,5469,11240,2502,5420,11167,10508,7087,7988,3408,8204,11944,4775,9110,8973,3696,8219,4004,8935,7799,4957,5779,5433,427,9285,5304,5282,11993,8230,10639,2468,4573,6843,3896,2525,11014,3816,9238,884,10785,5324,8349,5568,10345,717,11208,11251,1002,7025,10274,3772,2900,6604,10400,1344,8294,2290,3474,2275,11456,9557,382,8672,2182,3555,7798,4528,9130,10283,8636,4416,9954,397,3514,2649,2608,10259,9990,8756,6358,937,10714,10428,8904,9848,9934,3320,5624,8765,2694,5735,11277,3446,8655,5411,9656,5361,7280,11166,5464,1744,10015,1325,6277,10093,5588,1596,5214,8598,11176,9334,2374,2116,8750,8813,7704,4196,5517,8028,875,1094,2659,6896,6599,10882,8205,3360,10510,7049,10056,123,11757,5842,8587,158,1163,11501,8387,10664,114,8097,494,9000,958,4485,1929,3164,1475,9225,1680,8881,5150,11309,10872,4613,11710,3837,376,5161,6413,2128,22,11428,11778,10137,3527,5207,374,8618,10216,1959,5756,500,6040,111,4152,6252,9739,266,9057,9659,7399,5174,7,6835,2714,4852,3240,9246,6924,9560,814,7923,10572,5350,835,5495,9687,9730,7503,8552,9069,4443,8842,8076,9558,1099,5918,6939,7079,7223,3162,856,10410,7337,9275,3054,2337,2036,10073,4564,3118,2335,1947,3589,406,3248,11317,7708,535,5290,3934,4554,7928,3847,1494,10115,4910,3528,5333,4087,3971,4173,2580,9586,11352,4050,11919,86,3482,6768,132,6918,451,2156,4440,2970,3353,10099,4452,8656,5473,1876,3103,9702,5651,3186,3860,3575,2099,9115,10249,5844,2493,7089,1267,9974,11455,532,10729,4129,1923,11626,6079,9768,6108,3603,2289,5502,4233,3445,8951,10609,1692,2094,7155,3727,11510,4697,10091,3398,3880,4483,4400,6808,9689]
        # self.noisy_files = glob.glob(root + 'noisy/**.wav')
        # self.noisy_files = [
        #     name for name in self.noisy_files if int(self.file_id_from_name.findall(name.split(os.sep)[-1])[0]) in files_ids
        # ]
        # assert len(self.noisy_files) == 1200

    def _get_filenames(self, n: int) -> Tuple[str, str, str, Dict[str, Any]]:
        noisy_file = self.noisy_files[n % self.__len__()]
        filename = noisy_file.split(os.sep)[-1]
        file_id = int(self.file_id_from_name.findall(filename)[0])
        clean_file = self.root + f"clean/clean_fileid_{file_id}.wav"
        noise_file = self.root + f"noise/noise_fileid_{file_id}.wav"
        snr = int(self.snr_from_name.findall(filename)[0])
        target_level = int(self.target_level_from_name.findall(filename)[0])
        source_info = self.source_info_from_name.findall(filename)[0]
        metadata = {
            "snr": snr,
            "target_level": target_level,
            "source_info": source_info,
        }
        return noisy_file, clean_file, noise_file, metadata

    def __getitem__(
        self, n: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Gets the nth sample from the dataset.

        Parameters
        ----------
        n : int
            Index of the dataset sample.

        Returns
        -------
        np.ndarray
            Noisy audio sample.
        np.ndarray
            Clean audio sample.
        np.ndarray
            Noise audio sample.
        Dict
            Sample metadata.
        """
        noisy_file, clean_file, noise_file, metadata = self._get_filenames(n)
        noisy_audio, sampling_frequency = sf.read(noisy_file)
        clean_audio, _ = sf.read(clean_file)
        noise_audio, _ = sf.read(noise_file)
        num_samples = 30 * sampling_frequency  # 30 sec data
        metadata["fs"] = sampling_frequency

        if len(noisy_audio) > num_samples:
            noisy_audio = noisy_audio[:num_samples]
        else:
            noisy_audio = np.concatenate(
                [noisy_audio, np.zeros(num_samples - len(noisy_audio))]
            )
        if len(clean_audio) > num_samples:
            clean_audio = clean_audio[:num_samples]
        else:
            clean_audio = np.concatenate(
                [clean_audio, np.zeros(num_samples - len(clean_audio))]
            )
        if len(noise_audio) > num_samples:
            noise_audio = noise_audio[:num_samples]
        else:
            noise_audio = np.concatenate(
                [noise_audio, np.zeros(num_samples - len(noise_audio))]
            )
        return noisy_audio, clean_audio, noise_audio, metadata

    def __len__(self) -> int:
        """Length of the dataset."""
        return len(self.noisy_files)


if __name__ == "__main__":
    train_set = DNSAudio(
        root="/export/share/datasets/dns_fullband/MicrosoftDNS_4_ICASSP/training_set/"
    )
    validation_set = DNSAudio(
        root="/export/share/datasets/dns_fullband/MicrosoftDNS_4_ICASSP/validation_set/"
    )

    os.environ["JAX_PLATFORMS"] = "cpu"
    import os

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.scipy.signal import istft
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        noisy, clean, noise = [], [], []

        for sample in batch:
            noisy += [torch.FloatTensor(sample[0])]
            clean += [torch.FloatTensor(sample[1])]
            noise += [torch.FloatTensor(sample[2])]

        return torch.stack(noisy), torch.stack(clean), torch.stack(noise)

    batch_size = 512
    nfft = 512
    hop_length = 128
    noverlap = nfft - hop_length
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    for i, (noisy, clean, noise) in enumerate(train_loader):
        noisy = jnp.array(noisy.numpy())
        f, t, jax_stft = jax.scipy.signal.stft(
            noisy,
            nperseg=nfft,
            nfft=nfft,
            noverlap=noverlap,
            window="boxcar",
            return_onesided=True,
        )
        jax_stft_mag = jnp.abs(jax_stft)
        print(f"{jnp.mean(jax_stft_mag)=}")
        print(f"{jnp.var(jax_stft_mag)=}")
        print(f"{jnp.max(jax_stft_mag)=}")
        print(f"{jnp.min(jax_stft_mag)=}")
        if i == 1:
            break

    # Conclusion: jnp.mean(jax_stft_mag) is about 0.0007

    # # Perform STFT using JAX
    # f, t, jax_stft = jax.scipy.signal.stft(random_sequences_jax, nperseg=nfft, nfft=nfft, noverlap=noverlap, window='boxcar', return_onesided=True)

    # jax_stft_mag = jnp.abs(jax_stft)
    # jax_stft_phase = jnp.angle(jax_stft)

    # # Perform inverse STFT using JAX
    # _, jax_reconstructed = istft(jax_stft, nperseg=nfft, nfft=nfft, window='boxcar',noverlap=noverlap,  input_onesided=True)

    # _, jax_reconstructed_from_split = istft(jax_stft_mag * jnp.exp(1j * jax_stft_phase), nperseg=nfft, nfft=nfft, window='boxcar',noverlap=noverlap,  input_onesided=True)

    # # Print the generated sequences
    # print(f'{random_sequences_jax=}')
    # print(f'{jax_reconstructed}')

    # original_seq_length = random_sequences_jax.shape[1]
    # jax_reconstructed_seq_length = jax_reconstructed.shape[1]

    # max_error = jnp.max(jnp.abs(random_sequences_jax - jax_reconstructed[:,:original_seq_length]))
    # max_error_from_split = jnp.max(jnp.abs(random_sequences_jax - jax_reconstructed_from_split[:,:original_seq_length]))

    # print(f'{max_error=}')
    # print(f'{max_error_from_split=}')
    # print(f'{original_seq_length=}')
    # print(f'{jax_reconstructed_seq_length=}')
    # print(f'{jax_stft.shape=}')

    # # torch stft doesn't seem to match jax stft, but jax istft reproduces the original sequence at least

    # # Print the generated sequences
    # print(jax_stft)

    #     self.stft_mean = 0.2
    #         self.stft_var = 1.5
    #         self.stft_max = 140

    print("done")
