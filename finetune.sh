export PATH=/home/sdas/anaconda2/bin:$PATH
module load cuda/8.0 cudnn/5.1-cuda-8.0 opencv/3.4.1
mkdir -p weights_$2
python ./i3d_train.py $1 $2 $3
