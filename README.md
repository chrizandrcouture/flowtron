![Flowtron](https://nv-adlr.github.io/images/flowtron_logo.png "Flowtron")

## Flowtron: an Autoregressive Flow-based Network for Text-to-Mel-spectrogram Synthesis

### Rafael Valle, Kevin Shih, Ryan Prenger and Bryan Catanzaro

In our recent [paper] we propose Flowtron: an autoregressive flow-based
generative network for text-to-speech synthesis with control over speech
variation and style transfer. Flowtron borrows insights from Autoregressive Flows and revamps
[Tacotron] in order to provide high-quality and expressive mel-spectrogram
synthesis. Flowtron is optimized by maximizing the likelihood of the training
data, which makes training simple and stable. Flowtron learns an invertible
mapping of data to a latent space that can be manipulated to control many
aspects of speech synthesis (pitch, tone, speech rate, cadence, accent).

Our mean opinion scores (MOS) show that Flowtron matches state-of-the-art TTS
models in terms of speech quality. In addition, we provide results on control of
speech variation, interpolation between samples and style transfer between
speakers seen and unseen during training.

Visit our [website] for audio samples.


## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Setup
1. Clone this repo: `git clone https://github.com/NVIDIA/flowtron.git`
2. CD into this repo: `cd flowtron`
3. Initialize submodule: `git submodule update --init; cd tacotron2; git submodule update --init`
4. Install [PyTorch]
5. Install python requirements or build docker image
    - Install python requirements: `pip install -r requirements.txt`

## Training from scratch
1. Update the filelists inside the filelists folder to point to your data
2. Train using the attention prior and the alignment loss (CTC loss) until attention looks good
    `python train.py -c config.json -p train_config.output_directory=outdir data_config.use_attn_prior=1`
3. Resume training without the attention prior once the alignments have stabilized
    `python train.py -c config.json -p train_config.output_directory=outdir data_config.use_attn_prior=0`
`train_config.checkpoint_path=model_niters `
4. (OPTIONAL) If the gate layer is overfitting once done training, train just the gate layer from scratch
    `python train.py -c config.json -p train_config.output_directory=outdir` `train_config.checkpoint_path=model_niters data_config.use_attn_prior=0`
`train_config.ignore_layers='["flows.1.ar_step.gate_layer.linear_layer.weight","flows.1.ar_step.gate_layer.linear_layer.bias"]'` `train_config.finetune_layers='["flows.1.ar_step.gate_layer.linear_layer.weight","flows.1.ar_step.gate_layer.linear_layer.bias"]'`
5. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence.
Dataset dependent layers can be [ignored]

1. Download our published [Flowtron LJS], [Flowtron LibriTTS] or [Flowtron LibriTTS2K] model
2. `python train.py -c config.json -p train_config.ignore_layers=["speaker_embedding.weight"] train_config.checkpoint_path="models/flowtron_ljs.pt"`

## Fine-tuning for few-shot speech synthesis
1. Download our published [Flowtron LibriTTS2K] model
2. `python train.py -c config.json -p train_config.finetune_layers=["speaker_embedding.weight"] train_config.checkpoint_path="models/flowtron_libritts2k.pt"`

## Multi-GPU (distributed) and Automatic Mixed Precision Training ([AMP])
1. `python -m torch.distributed.launch --use_env --nproc_per_node=NUM_GPUS_YOU_HAVE train.py -c config.json -p train_config.output_directory=outdir train_config.fp16=true`

## Inference demo
Disable the attention prior and run inference:
1. `python inference.py -c config.json -f models/flowtron_ljs.pt -w models/waveglow_256channels_v4.pt -t "It is well know that deep generative models have a rich latent space!" -i 0`
2. `python inference.py -c config.json -f models/flowtron_libritts2p3k.pt -w tacotron2/waveglow/checkpoints/waveglow_256channels_ljs_v3.pt -t "It is well know that deep generative models have a rich latent space!" -i 0`
<!-- results/model_147999 -->
## Related repos
[WaveGlow](https://github.com/NVIDIA/WaveGlow) Faster than real time Flow-based
Generative Network for Speech Synthesis

## Acknowledgements
This implementation uses code from the following repos: [Keith
Ito](https://github.com/keithito/tacotron/), [Prem
Seetharaman](https://github.com/pseeth/pytorch-stft) and [Liyuan Liu](https://github.com/LiyuanLucasLiu/RAdam) as described in our code.

[ignored]: https://github.com/NVIDIA/flowtron/config.json#L12
[paper]: https://arxiv.org/abs/2005.05957
[Flowtron LJS]: https://drive.google.com/open?id=1Cjd6dK_eFz6DE0PKXKgKxrzTUqzzUDW-
[Flowtron LibriTTS]: https://drive.google.com/open?id=1KhJcPawFgmfvwV7tQAOeC253rYstLrs8
[Flowtron LibriTTS2K]: https://drive.google.com/open?id=1sKTImKkU0Cmlhjc_OeUDLrOLIXvUPwnO
[WaveGlow]: https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF
[PyTorch]: https://github.com/pytorch/pytorch#installation
[website]: https://nv-adlr.github.io/Flowtron
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp
[Tacotron]: https://arxiv.org/abs/1712.05884


libreTTS IDS: {1069: 0, 1088: 1, 1116: 2, 118: 3, 1246: 4, 125: 5, 1263: 6, 1502: 7, 1578: 8, 1841: 9, 1867: 10, 196: 11, 1963: 12, 1970: 13, 200: 14, 2092: 15, 2136: 16, 2182: 17, 2196: 18, 2289: 19, 2416: 20, 2436: 21, 250: 22, 254: 23, 2836: 24, 2843: 25, 2911: 26, 2952: 27, 3240: 28, 3242: 29, 3259: 30, 3436: 31, 3486: 32, 3526: 33, 3664: 34, 374: 35, 3857: 36, 3879: 37, 3982: 38, 3983: 39, 40: 40, 4018: 41, 405: 42, 4051: 43, 4088: 44, 4160: 45, 4195: 46, 4267: 47, 4297: 48, 4362: 49, 4397: 50, 4406: 51, 446: 52, 460: 53, 4640: 54, 4680: 55, 4788: 56, 5022: 57, 5104: 58, 5322: 59, 5339: 60, 5393: 61, 5652: 62, 5678: 63, 5703: 64, 5750: 65, 5808: 66, 587: 67, 6019: 68, 6064: 69, 6078: 70, 6081: 71, 6147: 72, 6181: 73, 6209: 74, 6272: 75, 6367: 76, 6385: 77, 6415: 78, 6437: 79, 6454: 80, 6476: 81, 6529: 82, 669: 83, 6818: 84, 6836: 85, 6848: 86, 696: 87, 7059: 88, 7067: 89, 7078: 90, 7178: 91, 7190: 92, 7226: 93, 7278: 94, 730: 95, 7302: 96, 7367: 97, 7402: 98, 7447: 99, 7505: 100, 7511: 101, 7794: 102, 78: 103, 7800: 104, 8051: 105, 8088: 106, 8098: 107, 8108: 108, 8123: 109, 8238: 110, 83: 111, 831: 112, 8312: 113, 8324: 114, 8419: 115, 8468: 116, 8609: 117, 8629: 118, 87: 119, 8770: 120, 8838: 121, 887: 122}
