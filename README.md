# InkShift

## DS 301 project: Personalized Handwriting Style Transfer

### Environment

HiGan+ env: ```bash pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html```

Python version: 3.8

HiGan+ inderence, follow <https://github.com/ganji15/HiGANplus.git>, change plt.show() to plt.savefig()

Curent demo env: comming soon

### Dataset

IAM dataset, raw data available at <https://fki.tic.heia-fr.ch/databases/iam-handwriting-database>

For convenience, here is the processed h5py files [trnvalset_words64_OrgSz.hdf5](https://github.com/ganji15/HiGANplus/releases/download/dataset/trnvalset_words64_OrgSz.hdf5)  [testset_words64_OrgSz.hdf5](https://github.com/ganji15/HiGANplus/releases/download/dataset/testset_words64_OrgSz.hdf5), which should put into the **./data/iam/** directory.

Processed data credit: <https://github.com/ganji15/HiGANplus.git>

For use of dataset and understanding of its structure, see `./IAM_data_process_demo.py`

Data processing files are under `./lib`

### Current demos

`./Flow_matching_demo.py` : simple flow matching demo on MNIST

code credit: <https://zhuanlan.zhihu.com/p/28731517852>

`./GAN_demo.py` : simple GAN demo for performace comparison with flow matching on MNIST

`Flow_matching_style_IAM_new.py` : flow matching with few shot transfer example on portion of IAM set
