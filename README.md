# OQMscores-python
Objective Quality Measure scores for speech samples in python.

Subjective listening tests provide perhaps the most reliable method for assessment of speech quality. These tests, however, can be time consuming requiring in most cases access to trained listeners. For these reasons, several researchers have investigated the possibility of devising objective, rather than subjective, measures of speech quality. Ideally, the objective measure should be able to assess the quality of the processed speech without needing access to the original speech signal. For futher understanding refer to [link](https://ecs.utdallas.edu/loizou/cimplants/quality_assessment_chapter.pdf).

## The OQM scores implemented in this package are:
- PESQ (Perceptual Evaluation of Speech Quality)
- SegSNR (Segmental Signal to Noise ratio)
- LLR (Log likelihood ratio)
- WSS (Weighted spectral slope)

## Requirements:

- [Librosa](https://pypi.org/project/librosa/)
- [Scipy](https://pypi.org/project/scipy/)
- [Numpy](https://pypi.org/project/numpy/)
- [PESQ python wrapper](https://github.com/ludlows/python-pesq/tree/master/pesq)

## Installation:

```pip install oqmscore```

## References:

- The PESQ implementation is done using [PESQ python wrapper](https://github.com/ludlows/python-pesq/tree/master/pesq)
- The SegSNR, WSS and LLR implementations are take from [SEGAN-repository](https://github.com/santi-pdp/segan_pytorch/blob/master/segan/utils.py)

## to-dos:
- Write unit tests
- ~~Write doc strings for the classes and functions properly.~~
