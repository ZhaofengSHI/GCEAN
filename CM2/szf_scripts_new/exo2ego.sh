source /data1/zhaofeng/anaconda3/bin/activate
conda activate cm2dvc

GPU_ID=4

DIS_LOSS_COEF=5.0
ADV_LOSS_COEF=0.25
ADV_LOSS_MARGIN=0.75
GZ_PRE_LOSS_COEF=1.0
GZ_VIEW2_PRE_LOSS_COEF=1.0
GZ_DIS_LOSS_COEF=5.0
LR_HIGH=1e-4
LR_LOW=5e-5
GZ_ADV_LOSS=0.25
MULTI_SCALE_DIS_LOSS_COEF=1.0
KL_0=0.1
KL_1=0.1
KL_2=0.1
KL_3=0.1
NUM_ATTN_HEADS=8
PRETRAIN='full'
PRETRAIN_PATH=/data1/zhaofeng/DenseVideoCaption/EgoMe-DVC/EgoMe-CM2_DVC-processed_anno/save_logs/save_egome_official_clip_exo_3/egome_clip_cm2_exo_5fps/model-best.pth
SEED=777
SAVE_DIR='save_logs_pt/save_egome_DA_exo2ego_1e-4+5e-5+5.0+0.25+1.0+8+0.1_1'
bash szf_scripts_new/train_egome_exo2ego_szf.sh $GPU_ID $SAVE_DIR $DIS_LOSS_COEF $PRETRAIN $PRETRAIN_PATH $ADV_LOSS_COEF $ADV_LOSS_MARGIN $GZ_PRE_LOSS_COEF $GZ_VIEW2_PRE_LOSS_COEF $GZ_DIS_LOSS_COEF $LR_HIGH $LR_LOW $GZ_ADV_LOSS $MULTI_SCALE_DIS_LOSS_COEF \
$KL_0 $KL_1 $KL_2 $KL_3 $NUM_ATTN_HEADS $SEED
