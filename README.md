# char-diffusion

`char-diffusion` is a toy implementation of a character-level diffusion language model that hallucinates (unconditionally generates) text within a fixed window. It is based on the naive bit-diffusion approach from the paper, [Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning](https://arxiv.org/pdf/2208.04202.pdf).


## TODOs

- Add conditional text generation.


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
