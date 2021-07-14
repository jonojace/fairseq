source ~/.bashrc
conda update -y -n base -c defaults conda
# conda env remove -y --name fairseq
conda create -y -n fairseq python=3.6 # python must be <=3.7 for tensorflow 1.15 to work
conda activate fairseq

git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
