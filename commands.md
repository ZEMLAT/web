ipython3 nbconvert price_prediction.ipynb --to python
conda env export > environment3.x.yml
conda activate tf_gpu_env3.x
tree -H '.' --noreport --charset utf-8 -o index.html