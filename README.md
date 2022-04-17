# Thesis Topic: Adversarial Text-to-Image generation using Contrastive Learning
This is the official Pytorch implementation of my thesis (thesis link: )

## Requirements
* Python ≥ 3.6

* PyTorch ≥ 1.4.0


## Prepare Data


Download the preprocessed datasets from link (provide dataset link)




## Training
- Pretrain DAMSM+CL:
  - For bird dataset: `python pretrain_DAMSM.py --cfg cfg/DAMSM/bird.yml --gpu 0`
 

- Train our model (SSA-GAN+CL):
  - For bird dataset: `python main.py` (B_VALIDATION: False and NET_G: '' inside cfg/bird.yml)
  


## Pretrained Models
- [DAMSM+CL for bird](provide link). Download and save it to `DAMSMencoders/`

### Main framwork
- [SSA-GAN+CL for bird](provide link). Download and save it to `models/`

### Ablation Study
- [SSA-GAN+CL1 for bird](provide link). Download and save it to `models/`
- [SSA-GAN+CL2 for bird](provide link). Download and save it to `models/`
- [SSA-GAN+CL or SSA-GAN+CL1+CL2 for bird](provide link). Download and save it to `models/`


## Evaluation
- Sampling and get the R-precision:
  - `python main.py' (B_VALIDATION: True and NET_G: 'output_tmp_models/bird_sloss01/64/models/netG_550.pth' inside cfg/bird.yml)
  
- Inception score:
  - 'cd IS/bird && python inception_score_bird.py --image_folder ../../outout_tmp_models/bird_sloss01/64/models/netG_550'

  

- FID: 
  - `python fid_score.py --gpu 0 --batch-size 50 --path1 data/test_images --path2 outout_tmp_models/bird_sloss01/64/models/netG_550

  
### Citation
If you find this work useful in your research, please consider citing:

```
@MISC{priyanka2022SSA-GAN_CL,
  title={Adversarial Text-to-Image generation using Contrastive Learning},
  author={Upadhyay, Priyanka},
  year={2022}
}
```
### Acknowledge
Our work is based on the following works:
- [SSA-GAN: Text to Image Generation with Semantic-Spatial Aware GAN](https://arxiv.org/abs/1711.10485) [[code]](https://github.com/wtliao/text2image)
