# char-diffusion

`char-diffusion` is a toy implementation of a character-level diffusion language model that hallucinates (unconditionally generates) text within a fixed window. It is based on the naive bit-diffusion approach from the paper, [Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning](https://arxiv.org/pdf/2208.04202.pdf).


## TODOs

- [ ] Add EMA.

- [ ] Add "softmax facorization" from Chen et al. to the output layer instead of using the final linear layer to directly predict analog bits.
  > Instead of using a linear output layer to predict analog bits directly, we first predict a probability distribution over 256 classes per sub-pixel (with each class corresponds to one of the 256 different 8-bit codes), and then map class distribution into analog bits by taking weighted average over all 256 different 8-bit codes.

- [ ] Add conditional text generation.


## Citations 


Bit-Diffusion

```bibtex
@article{Chen2022AnalogBG,
    title   = {Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning},
    author  = {Ting Chen and Ruixiang Zhang and Geoffrey E. Hinton},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2208.04202}
}
```

`sqrt` schedule 

```bibtex
@article{Li-2022-DiffusionLM,
  title={Diffusion-LM Improves Controllable Text Generation},
  author={Xiang Lisa Li and John Thickstun and Ishaan Gulrajani and Percy Liang and Tatsunori Hashimoto},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.14217}
}
```
