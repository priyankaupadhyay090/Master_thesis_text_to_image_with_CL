# Adversarial Text-to-Image Generation using Contrastive Learning

### Introduction
This is the official Pytorch implementation of my thesis.

**Thesis Title:** "Adversarial Text-to-Image generation using Contrastive Learning"

**Link:** 


### Requirements
* Python ≥ 3.6

* PyTorch ≥ 1.4.0

* tensorboardX

* easydict

* scipy==1.1.0


### Prepare Data


Download the [birds dataset](https://drive.google.com/file/d/1NN1ouwAUbvtkaNNAXmRWA7AMJVBMkp9r/view?usp=sharing) (includes preprocessed metadata) and save it to `data/`


### Pre-trained Models

- Pre-trained DAMSM Models
  - [DAMSM+CL for bird](https://drive.google.com/drive/folders/1mic3yRak8oTXE2f6H8gTG5e4Os8qAETM?usp=sharing). Download and save it to `DAMSMencoders/`

- Pre-trained our model 
  - [SSA-GAN+CL for bird](https://drive.google.com/file/d/14pSKzoUj8EDfaoNLdQyx6qmv8lzKXOfV/view?usp=sharing). Download and save it to `output_tmp_models/bird_sloss01/64/models/`



### Start training

- Pretrain DAMSM+CL:
  - For bird dataset: python pretrain_DAMSM.py --cfg cfg/DAMSM/bird.yml --gpu 0
 

- Train our model (SSA-GAN+CL):
  - For bird dataset: python main.py (where cfg/bird.yml/B_VALIDATION: False and cfg/bird.yml/NET_G: "" )
  



### Evaluation
- Sampling and get the R-precision:
  - python main.py (where cfg/bird.yml/B_VALIDATION: True and cfg/bird.yml/NET_G: "output_tmp_models/bird_sloss01/64/models/netG_550.pth" )
  
- Inception score (you can also visit https://github.com/hanzhanggit/StackGAN-inception-model):
  - cd IS/bird && python inception_score_bird.py --image_folder ../../output_tmp_models/bird_sloss01/64/models/netG_550

  
- FID (you can visit https://github.com/MinfengZhu/DM-GAN): 
  - cd FID && python fid_score.py --gpu 0 --batch-size 50 --path1 data/test_images --path2 output_tmp_models/bird_sloss01/64/models/netG_550

  

### Performance (Quantitative Results)
Comparision between our proposed approach and few state-of-the-art methods.

You will get the scores close to below after training our framework or using our pre-trained model:

![](https://github.com/priyankaupadhyay090/Master_thesis_CL_GANs/blob/main/results.PNG)


### Qualitative Results
Comparison between our proposed approach and some of the other T2I methods on the CUB dataset:

![](https://github.com/priyankaupadhyay090/Master_thesis_CL_GANs/blob/main/qualitative.PNG)



#### Acknowledgements

Our work is based on the following works and this implementation borrows part of the code from:
- [SSA-GAN: Text to Image Generation with Semantic-Spatial Aware GAN](https://arxiv.org/abs/2104.00567) [[code]](https://github.com/wtliao/text2image)
- [AttnGAN+CL: Improving Text-to-Image Synthesis Using Contrastive Learning](https://arxiv.org/abs/2107.02423?context=cs) [[code]](https://github.com/huiyegit/T2I_CL)
- [DM-GAN: Dynamic Memory Generative Adversarial Networks for Text-to-Image Synthesis](https://arxiv.org/abs/1904.01310)[[code]](https://github.com/MinfengZhu/DM-GAN)
