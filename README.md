# Two-stage Joint Underwater Image Enhancement and Marine Snow Removal via Teacher-Guided Feature Optimization



<hr />

> **Abstract:** Color shift and marine snow are two severe types of underwater imaging noise, which can significantly impact subsequent computer vision tasks. However, existing methods cannot perform underwater image enhancement (UIE) and marine snow removal (MSR) simultaneously with a unified network architecture and a set of pretrained weights. To address this issue, we propose the two-stage joint UIE and MSR network (TJUMnet). With the proposed teacher-guided feature optimization (TFO) and hierarchical reconstruction (HR), TJUMnet can effectively integrate and learn feature representations under various underwater noise conditions. Additionally, we constructed a new joint underwater-image enhancement and marine-snow removal dataset (JUEMR), comprising 2700 image sets covering color shift noise and different intensities of marine snow noise. Extensive experiments on multiple benchmark datasets and the JUEMR dataset demonstrate that TJUMnet reaches state-of-the-art levels in terms of quantitative and visual performance.


## Architecture
(It seems that the images do not display correctly at Anonymous GitHub. The following two images are paperImgs/fig1.png and paperImgs/fig2.png respectively)
<table>
  <tr>
    <td colspan="2" align="center"> <img src = "paperImgs/fig1.png" width="800"> </td>
  </tr>
  <tr>
    <td colspan="2" align="center"><p><b>TJUMnet Architecture</b></p></td>
  </tr>
  <tr>
    <td align="center"> <img src = "paperImgs/fig2.png" width="800"> </td>
  </tr>
  <tr>
    <td align="center"><p><b>Latent Space Architecture</b></p></td>
  </tr>
</table>

## How to use
#### JUEMR dataset & pre-trained Models
JUEMR dataset can be downloaded at [https://drive.google.com/file/d/1g9KeRR3sv_bxsHMV8HTfUK-E5o4zZ_A-/view?usp=sharing] 
(Anonymised)

#### Training
```shell
python train.py --dataset_train utils.dataset.DatasetForTrain --dataset_valid utils.dataset.DatasetForValid --train dataset/JUEMR/train --valid dataset/JUEMR/test --save-dir savedir
```
#### Inference

```shell
python inference.py 
```


