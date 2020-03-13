#python predict_lxx.py -i 01_test.jpg -o output.jpg -s 1.0 -m ./checkpoints/CP100.pth

#python predict_efficientNet.py -i data/REFUGE/images/g0001.jpg -o output_efficient.jpg -s 1.0 -m ./checkpoints_drive/CP68.pth
#python predict_efficientNet.py -i 01_test.jpg -o output_efficient.jpg -s 1.0 -m ./checkpoints_drive/CP68.pth

export CUDA_VISIBLE_DEVICES="4"
#/home/lixiaoxing/data/DRIVE/test/ \
python predict_efficientNet.py -i /home/lixiaoxing/data/DRIVE/test/ \
                               -o /home/lixiaoxing/github/Pytorch-UNet/results/ \
                               -s 1.0 \
                               -m ./checkpoints_drive1/CP250.pth



#dir=/home/lixiaoxing/data/DRIVE/test/
#for file in $dir/*; do
#    python predict_efficientNet.py -i $file -o output_efficient.jpg -s 1.0 -m ./checkpoints_drive/CP68.pth
#    echo $file
#done
