# char-diffusion

`char-diffusion` is a toy implementation of a character-level diffusion language model that hallucinates (unconditionally generates) text within a fixed window. It is based on the elegant bit-diffusion approach from the paper, [Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning](https://arxiv.org/pdf/2208.04202.pdf).

There's no reason a priori that I expected the model to create coherent text, but interestingly enough, the model can, at the very least, hallucinate some tiny semblance of sentence structure. See samples from a training run on Tolstoy's *War and Peace* with a 96 character window:

__Step 1,000__

```
➜ wufer to did enessuld alios lil lid dest ald illa emister in theeuer to dis tater the gint diine
➜ osud my teaor dos lild that a dmorlale thitn Mess in sesuer anld desqesed the lad andar lild dmi
➜ e seusifed int desesser in the the cesing lis wind ceenet all lilad ald sald the sesqer allild d
➜ outen tilling witl lis dicit. "Le lor esistereesqer lill ll le eisqt ald in tieser ant liit war
➜ osud widn did inteess salaled eor ald thid gist eilllald le  oyst eor dine in the lis doser the
➜ motte suosed will and dis pmaip oqny ald lile deisted all and giser eotl the wor desertind los d
➜ enldoc of thoql ald sewerd le ellat all le mqeeser deix ildiospd antle liceseld ale was sould en
```

__Step 25,000__
```
➜ d had stop up on the foons of the regalling of the came under the faver time for went when the d
➜ l and handrale, and did the doscest and the destroach was handed when the grent the one cale of
➜ ld a fact, where he reated calail the shope, wenting and closed at in a wat and lad the room dis
➜ estual and come and spicklined to the Empeross faces of fince and expired by him for lopes said
➜ stoul and knes that had goed to him at the fool and the barsage. He did not seep that that came
➜ nteged what had seemed, seemed them, was not to bestroecded that that they had at the deal to de
➜ tion were had at ear and was to beliess that he was still dning of that had been dis gored, and
```

__Step 125,000__
```
➜ here it had to be approaching with face, and he did not let her that the something gase that was
➜ he bears and window and had been ordered to fill the behind the officers to the genery command t
➜ to fer the same time and reached places, and they took evidently and simply to certain that time
➜  never became she had noticed that life. He had known her own sleck at the shoulders he had the
➜ ed at her what we were doing the wall she had performed on the he went down the firing and that
➜ nd at even talking but a renefentle stood by a saddle it with a criman with which when he began
➜  him, and on one of which they commised her inside her soul. She who was was a rench of delicate
```


See [**SED**: Self-Conditioned Embedding Diffusion for Text Generation](https://openreview.net/pdf?id=OpzV3lp3IMC) for a token-level word embedding based generation method.


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
