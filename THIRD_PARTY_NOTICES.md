# Third-party notices

## MiniMind

Ryn uses the model implementation from [jingyaogong/minimind](https://github.com/jingyaogong/minimind), pinned by the `upstream/minimind` Git submodule. The upstream project is licensed under Apache License 2.0. Its license and notices remain applicable to the submodule; Ryn does not relicense MiniMind files. The upstream source and license are available in the submodule and at <https://github.com/jingyaogong/minimind/blob/master/LICENSE>.

## Tokenizer

The default tokenizer is downloaded from [`Qwen/Qwen2.5-Coder-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) at training time. It is not bundled in this repository. Follow the source model's license and terms for the tokenizer assets.

## Training data

No third-party training corpus or model weights are bundled. Dataset providers retain their own terms and licenses. Review each source before training or releasing a checkpoint.
