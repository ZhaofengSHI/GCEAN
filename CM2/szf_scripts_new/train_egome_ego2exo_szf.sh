
GPU_ID=$1
EXP_NAME=egome_clip_cm2_ego2exo_5fps
SAVE_DIR=$2
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
SEED=${20}

config_path=cfgs/${EXP_NAME}.yml

# Training
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py --gpu_id ${GPU_ID} --save_dir ${SAVE_DIR} --text_crossAttn_loc after --text_crossAttn \
--ret_encoder avg --cfg_path ${config_path} --exp_name ${EXP_NAME} --sim_attention cls_token \
--target_domain egome_ego --bank_type egome_ego --soft_k 80 --window_size 50 \
--dis_loss_coef ${DIS_LOSS_COEF} \
--pretrain ${PRETRAIN} --pretrain_path ${PRETRAIN_PATH} --adv_loss_coef ${ADV_LOSS_COEF} --adv_loss_margin ${ADV_LOSS_MARGIN} \
--gz_pre_loss_coef ${GZ_PRE_LOSS_COEF} --gz_view2_pre_loss_coef ${GZ_VIEW2_PRE_LOSS_COEF} --gz_dis_loss_coef ${GZ_DIS_LOSS_COEF} \
--lr_high ${LR_HIGH} --lr_low ${LR_LOW} --gz_adv_loss_coef ${GZ_ADV_LOSS} --multi_scale_dis_loss_coef ${MULTI_SCALE_DIS_LOSS_COEF} \
--kl_loss_0 ${KL_0} --kl_loss_1 ${KL_1} --kl_loss_2 ${KL_2} --kl_loss_3 ${KL_3} --num_attn_heads ${NUM_ATTN_HEADS} --seed ${SEED}


#--retrieval_ablation no_ret  #--nvec_proj_use True #--soft_k 25


# Evaluation
eval_json='/data1/zhaofeng/EgoMe_processing/EgoMe_new_jsons/official_split_processed/egome_val_exo.json'
eval_folder=${EXP_NAME} # specify the folder to be evaluated

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_szf.py --eval_folder ${eval_folder} --gpu_id ${GPU_ID} --eval_save_dir ${SAVE_DIR} \
--test_video_feature_folder '/data1/zhaofeng/EgoMe_processing/EgoMe_dataset_new/video_clipl_features_5fps/' \
--eval_caption_file ${eval_json} \
--text_crossAttn_loc after --target_domain egome_ego --bank_type egome_ego --sim_match window_cos \
--ret_text token --down_proj deep --ret_vector nvec --eval_transformer_input_type queries \
--able_ret --sim_attention cls_token  #--nvec_proj_use #--soft_k 25    #--proj_use

# Test
eval_json='/data1/zhaofeng/EgoMe_processing/EgoMe_new_jsons/official_split_processed/egome_test_exo.json'
eval_folder=${EXP_NAME} # specify the folder to be evaluated
gt_file_para_test='/data1/zhaofeng/EgoMe_processing/EgoMe_new_jsons/official_split_processed/para_egome_test_exo.json'

CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_szf.py --eval_folder ${eval_folder} --gpu_id ${GPU_ID} --eval_save_dir ${SAVE_DIR} \
--test_video_feature_folder '/data1/zhaofeng/EgoMe_processing/EgoMe_dataset_new/video_clipl_features_5fps/' \
--eval_caption_file ${eval_json} \
--text_crossAttn_loc after --target_domain egome_ego --bank_type egome_ego --sim_match window_cos \
--ret_text token --down_proj deep --ret_vector nvec --eval_transformer_input_type queries \
--able_ret --sim_attention cls_token  \
--gt_file_for_para_test ${gt_file_para_test} \
--eval_mode eval_testset \

#--nvec_proj_use #--soft_k 25    #--proj_use