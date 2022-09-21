# char-diffusion

`char-diffusion` is a toy implementation of a character-level diffusion language model that hallucinates (unconditionally generates) text within a fixed window. It is based on the elegant bit-diffusion approach from the paper, [Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning](https://arxiv.org/pdf/2208.04202.pdf).

There's no reason a priori that I expected the model to create coherent text, but interestingly enough, the model can, at the very least, hallucinate some tiny semblance of sentence structure. See samples from a training run on Tolstoy's *War and Peace* with a 64 character window:

__Step 1,000__

```
edd````````````````````````````````````````````````````````d````
```

__Step 5,000__

```
on ind and `id fere aaret he hat ler`on iet hesaon indaseld asas
```

__Step 50,000__
```
 y, and they, and be would gate the wive or were fine that the Mo
```


## Findings

- Asymmetric time intervals significantly improve sample spelling. Recommend time deltas in [1.0, 10.0] for 2,000 step processes; note that larger deltas seem to lead to word repetition (word collapse?).


## TODOs

- [ ] Add EMA.

- [ ] Stabilize `sqrt` schedule from Li et al. [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217)

- [ ] Swap out U-net architecture with a Transformer/MLP first one.

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
