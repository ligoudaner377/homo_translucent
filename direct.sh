set -ex
name="direct"
model="direct"
dataroot="./datasets/translucent"

python ./train.py --dataroot ${dataroot} --name ${name} --model ${model} --gpu_ids 0,1,2,3
python ./test.py --dataroot ${dataroot} --name ${name} --model ${model} --eval
