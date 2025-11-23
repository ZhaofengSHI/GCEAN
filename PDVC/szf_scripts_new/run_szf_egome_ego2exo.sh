#source /data1/zhaofeng/anaconda3/bin/activate
#conda activate pdvc

# Training
GPU_ID=$1
EXP_NAME=egome_DA_clip_pdvc_ego2exo_5fps
SAVE_DIR=$2 #'save_egome_ego2exo'
DIS_LOSS_COEF=$3
PRETRAIN=$4
PRETRAIN_PATH=$5
ADV_LOSS_COEF=$6
ADV_LOSS_MARGIN=$7
GZ_PRE_LOSS_COEF=$8
GZ_VIEW2_PRE_LOSS_COEF=$9
GZ_DIS_LOSS_COEF=${10}
LR_HIGH=${11}
LR_LOW=${12}
GZ_ADV_LOSS=${13}
MULTI_SCALE_DIS_LOSS_COEF=${14}
KL_0=${15}
KL_1=${16}
KL_2=${17}
KL_3=${18}
NUM_ATTN_HEADS=${19}

config_path=cfgs/${EXP_NAME}.yml

python train.py --cfg_path ${config_path} --gpu_id ${GPU_ID} --save_dir ${SAVE_DIR} --dis_loss_coef ${DIS_LOSS_COEF} \
--pretrain ${PRETRAIN} --pretrain_path ${PRETRAIN_PATH} --adv_loss_coef ${ADV_LOSS_COEF} --adv_loss_margin ${ADV_LOSS_MARGIN} \
--gz_pre_loss_coef ${GZ_PRE_LOSS_COEF} --gz_view2_pre_loss_coef ${GZ_VIEW2_PRE_LOSS_COEF} --gz_dis_loss_coef ${GZ_DIS_LOSS_COEF} \
--lr_high ${LR_HIGH} --lr_low ${LR_LOW} --gz_adv_loss_coef ${GZ_ADV_LOSS} --multi_scale_dis_loss_coef ${MULTI_SCALE_DIS_LOSS_COEF} \
--kl_loss_0 ${KL_0} --kl_loss_1 ${KL_1} --kl_loss_2 ${KL_2} --kl_loss_3 ${KL_3} --num_attn_heads ${NUM_ATTN_HEADS}
# The script will evaluate the model for every epoch. The results and logs are saved in `./save`.

 # Evaluation
eval_json='/data1/zhaofeng/EgoMe_processing/EgoMe_new_jsons/official_split_processed/egome_val_exo.json'
eval_folder=${EXP_NAME} # specify the folder to be evaluated

python eval_szf.py --eval_folder ${eval_folder} \
 --eval_transformer_input_type queries \
 --gpu_id ${GPU_ID} \
 --eval_caption_file ${eval_json} \
 --eval_save_dir ${SAVE_DIR} \


 # test
eval_json='/data1/zhaofeng/EgoMe_processing/EgoMe_new_jsons/official_split_processed/egome_test_exo.json'
gt_file_para_test='/data1/zhaofeng/EgoMe_processing/EgoMe_new_jsons/official_split_processed/para_egome_test_exo.json'

eval_folder=${EXP_NAME} # specify the folder to be evaluated
python eval_szf.py --eval_folder ${eval_folder} \
 --eval_transformer_input_type queries \
 --gpu_id ${GPU_ID} \
 --eval_caption_file ${eval_json} \
 --eval_save_dir ${SAVE_DIR} \
 --eval_mode eval_testset \
 --gt_file_test ${eval_json} \
 --gt_file_for_para_test ${gt_file_para_test} \


# # val in domain
#eval_json='/data1/zhaofeng/EgoMe_processing/EgoMe_new_jsons/official_split_processed/egome_val_ego.json'
#gt_file_para_test='/data1/zhaofeng/EgoMe_processing/EgoMe_new_jsons/official_split_processed/para_egome_val_ego.json'
#
#eval_folder=${EXP_NAME} # specify the folder to be evaluated
#python eval_szf.py --eval_folder ${eval_folder} \
# --eval_transformer_input_type queries \
# --gpu_id ${GPU_ID} \
# --eval_caption_file ${eval_json} \
# --eval_save_dir ${SAVE_DIR} \
# --eval_mode eval_testset \
# --gt_file_test ${eval_json} \
# --gt_file_for_para_test ${gt_file_para_test} \
#
#
# # test in domain
#eval_json='/data1/zhaofeng/EgoMe_processing/EgoMe_new_jsons/official_split_processed/egome_test_ego.json'
#gt_file_para_test='/data1/zhaofeng/EgoMe_processing/EgoMe_new_jsons/official_split_processed/para_egome_test_ego.json'
#
#eval_folder=${EXP_NAME} # specify the folder to be evaluated
#python eval_szf.py --eval_folder ${eval_folder} \
# --eval_transformer_input_type queries \
# --gpu_id ${GPU_ID} \
# --eval_caption_file ${eval_json} \
# --eval_save_dir ${SAVE_DIR} \
# --eval_mode eval_testset \
# --gt_file_test ${eval_json} \
# --gt_file_for_para_test ${gt_file_para_test} \