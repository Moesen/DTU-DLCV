class="hair" #{class}
echo ${class}
filename="../../data/${class}.txt"
n=1
echo $filename
while read line; do
# reading each line
python3 ../../stylegan2-ada-pytorch/projector.py --outdir=$class_w_out_$n --target=$line \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

#echo "Line No. $n : $line"
n=$((n+1))
done < $filename


