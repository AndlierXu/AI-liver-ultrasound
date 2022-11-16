# Improving artificial intelligence pipeline for liver malignancy diagnosis using ultrasound images and video frames

![image](https://github.com/AndlierXu/AI-liver-ultrasound/blob/main/fig/figure1.jpg)

#### Train the model:
```bash
python train.py -c config/run.config -g 0,1
```

#### Evaluate the model:
```bash
python eval.py -c config/run.config -g 0
```

#### Notes
The de-identified data are available at https://doi.org/10.5281/zenodo.7272660. The dataset despite being open to public access, is subject to copyright. Any use of data contained within this dataset must receive appropriate acknowledgement and credit.
