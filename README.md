# Rectified Gradient

Experiment codes and additional samples for [*Why are Saliency Maps Noisy? Cause of and Solution to Noisy Saliency Maps*](https://arxiv.org/abs/1902.04893).


**The work is presented at [2019 ICCV Workshop on VXAI](http://xai.unist.ac.kr/workshop/2019/).**

![alt tag](https://github.com/1202kbs/Rectified-Gradient/blob/master/assets/comparison.png)

## Random Sample Links

We list the links to Google drives containing attribution maps for 1.5k randomly chosen ImageNet images. Samples for the same image have the same file name ``{image number}.png`` for ease of comparison between attribution maps with and without final threshold / attribution maps for original image and adversarial image. We did not include samples for adversarial attacks which failed to change the final decision.

[Random Samples W/O Baseline Final Threshold](https://drive.google.com/drive/folders/1F9k-Jvxe1OppDoIDHV1SWLzugkpPQMkY?usp=sharing)

[Random Samles with Baseline Final Threshold](https://drive.google.com/drive/folders/1LQOWtvJPV9nnUCjB2VRNDgFGixmpAnJB?usp=sharing)

[Adversarial Attack Samples W/O Baseline Final Threshold](https://drive.google.com/drive/folders/1kkgVQs2jBWWqLW5CTrKhpBnpwAUhZDJs?usp=sharing)

[Adversarial Attack Samples with Baseline Final Threshold](https://drive.google.com/drive/folders/1MXrmQCgWgpf84Mj1f9Q8QrOfOncSbvlf?usp=sharing)

## Results

### Effect of Threshold Percentile

![alt tag](https://github.com/1202kbs/Rectified-Gradient/blob/master/assets/threshold1.png)

![alt tag](https://github.com/1202kbs/Rectified-Gradient/blob/master/assets/threshold2.png)

![alt tag](https://github.com/1202kbs/Rectified-Gradient/blob/master/assets/threshold3.png)


### Comparison with Other Attribution Methods

![alt tag](https://github.com/1202kbs/Rectified-Gradient/blob/master/assets/inter_1.png)

![alt tag](https://github.com/1202kbs/Rectified-Gradient/blob/master/assets/inter_2.png)

![alt tag](https://github.com/1202kbs/Rectified-Gradient/blob/master/assets/inter_3.png)

![alt tag](https://github.com/1202kbs/Rectified-Gradient/blob/master/assets/inter_4.png)
