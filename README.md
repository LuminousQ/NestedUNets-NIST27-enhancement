# NIST27 enhancement

Fingerprint images to be enhanced should be put into `./img` folder. If images need to be segmented first, put the segmentation mask in `./mask`. We provide latent fingerprint enhancement model file which are saved in `./model` folder.

Run enhancement

```shell
python enhance.py
```

The enhancement results will be generated in `./enhanced`