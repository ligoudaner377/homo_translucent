set -ex
name="edit_twoshot"
model="edit_twoshot"
dataroot="./datasets/translucent"

python ./train.py --dataroot ${dataroot} --name ${name} --model ${model} --gpu_ids 0,1,2,3
python ./test.py --dataroot ${dataroot} --name ${name} --model ${model} --eval
python ./inference_real.py --dataroot "./datasets/real" --dataset_mode real --name ${name} --model ${model} --eval