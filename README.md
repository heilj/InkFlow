# InkFllow

## DS 301 project: Personalized Handwriting Style Transfer


### Environment

HiGan+ env: ```bash pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html```

Python version: 3.8 (HiGAN+)

Flow Matching python version: 3.12
Flow matching env: ```bash pip install -r requirements_fm.txt```

HiGan+ inderence, follow <https://github.com/ganji15/HiGANplus.git>, change plt.show() to plt.savefig() in `/HiGANplus/HiGAN+/networks/model.py`

previous demo env: same as flow matching

### Dataset

IAM dataset, raw data available at <https://fki.tic.heia-fr.ch/databases/iam-handwriting-database>

For convenience, here is the processed h5py files [trnvalset_words64_OrgSz.hdf5](https://github.com/ganji15/HiGANplus/releases/download/dataset/trnvalset_words64_OrgSz.hdf5)  [testset_words64_OrgSz.hdf5](https://github.com/ganji15/HiGANplus/releases/download/dataset/testset_words64_OrgSz.hdf5), which should put into the **./data/iam/** directory.

Processed data credit: <https://github.com/ganji15/HiGANplus.git>

For use of dataset and understanding of its structure, see `./IAM_data_process_demo.py`

Data processing files are under `./lib`

For additional data process on content image and style image, use `./Generate_font_img.py` and `./save_style_img.py`


### Flow matching

Main code include train and inference dependent on other `./lib`, `./networks`, `./flow_matching`, `.pretrianed`

#### Train

`python ./LCFM_style_cfg.py` 
latest distributed training script, hyper params defined inside

#### Inference

`python ./inference_with_random_writer.py`
inference with random writers form IAM test set, need specifcation on ckpt, cfg scale, and input text

`python ./inference_with_custom_writer.py`
inference with custom writers with preprocessed style image, need specifcation on ckpt, cfg scale, and input text


other details, ckpts... will be updated later


#### Results

current results of custom inputs are in `custom_outputs`, where the text label and style reference can be found under `./data`.

### Previous demos

*this is a legacy of previous attempts and does not related to the result of the project.*

`./Flow_matching_demo.py` : simple flow matching demo on MNIST

code credit: <https://zhuanlan.zhihu.com/p/28731517852>

`./GAN_demo.py` : simple GAN demo for performace comparison with flow matching on MNIST

`Flow_matching_style_IAM_new.py` : flow matching with few shot transfer example on portion of IAM set
