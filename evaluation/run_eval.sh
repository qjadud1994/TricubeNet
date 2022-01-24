checkpoint="test1/"
epoch=110
s_thresh=0.4
scale=2.2
backbone="hourglass104_MRCB_cascade"
flip="False"
ma="True"
exp=1
mode="testsplit_1024"


CUDA_VISIBLE_DEVICES=0 python eval_DOTA.py --backbone=${backbone}  --checkpoint=${checkpoint}/ckpt_${epoch}  --c_thresh=0.1 --s_thresh=${s_thresh}  --scale=${scale}  --flip=${flip}  --ma=${ma} --experiment=${exp}   --save_img=True  --input_size=1024  --kernel=3  --mode=${mode}
