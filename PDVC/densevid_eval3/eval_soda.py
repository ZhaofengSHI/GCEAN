import sys
import os
from os.path import dirname, abspath

pdvc_dir = dirname(dirname(abspath(__file__)))
sys.path.append(pdvc_dir)
sys.path.append(os.path.join(pdvc_dir, 'densevid_eval3/SODA'))

import numpy as np
from densevid_eval3.SODA.soda import SODA
from densevid_eval3.SODA.dataset import ANETCaptions
from densevid_eval3.eval_para import eval_para

def eval_tool(prediction, referneces=None, metric='Meteor', soda_type='c', verbose=False):

    args = type('args', (object,), {})()
    args.prediction = prediction
    args.references = referneces
    args.metric = metric
    args.soda_type = soda_type
    args.tious = [0.3, 0.5, 0.7, 0.9]
    args.verbose = verbose
    args.multi_reference = False

    data = ANETCaptions.from_load_files(args.references,
                                        args.prediction,
                                        multi_reference=args.multi_reference,
                                        verbose=args.verbose,
                                        )
    data.preprocess()
    if args.soda_type == 'a':
        tious = args.tious
    else:
        tious = None
    # evaluator = SODA(data,
    #                  soda_type=args.soda_type,
    #                  tious=tious,
    #                  scorer=args.metric,
    #                  verbose=args.verbose
    #                  )
    # result = evaluator.evaluate()
    # meteor
    meteor_evaluator = SODA(data,
                     soda_type=args.soda_type,
                     tious=tious,
                     scorer='Meteor',
                     verbose=args.verbose
                     )
    meteor_result = meteor_evaluator.evaluate()
    # cider
    cider_evaluator = SODA(data,
                     soda_type=args.soda_type,
                     tious=tious,
                     scorer='Cider',
                     verbose=args.verbose
                     )
    cider_result = cider_evaluator.evaluate()
    # tIoU
    tiou_evaluator = SODA(data,
                     soda_type='d',
                     tious=tious,
                     scorer='Meteor', # NOTE: Meteor is set, but this variable is dummy
                     verbose=args.verbose
                     )
    tiou_result = tiou_evaluator.evaluate()    
    # to replace dummy 'Meteor' with 'tIoU'
    tiou_result["tIoU"] = tiou_result["Meteor"]
    del tiou_result["Meteor"]
    
    result = {
            "SODA_Meteor": meteor_result["Meteor"],
            "SODA_Cider": cider_result["Cider"],
            "SODA_tIoU": tiou_result["tIoU"]
            }
        
    return result

# def eval_soda(p, ref_list,verbose=False):
#     score_sum = []
#     for ref in ref_list:
#         r = eval_tool(prediction=p, referneces=[ref], verbose=verbose, soda_type='c')
#         score_sum.append(r['Meteor'])
#     soda_avg = np.mean(score_sum, axis=0) #[avg_pre, avg_rec, avg_f1]
#     soda_c_avg = soda_avg[-1]
#     results = {'soda_c': soda_c_avg}
#     return results
def eval_soda(p, ref_list,verbose=False):
    score_sum = {
        "SODA_Meteor": [],
        "SODA_Cider": [],
        "SODA_tIoU": [],
    }
    results = {}
    for ref in ref_list:
        r = eval_tool(prediction=p, referneces=[ref], verbose=verbose, soda_type='c')
        for eval_key in score_sum.keys():
            score_sum[eval_key].append(r[eval_key])
    for eval_key in score_sum.keys():        
        soda_avg = np.mean(score_sum[eval_key], axis=0) #[avg_pre, avg_rec, avg_f1]
        soda_avg_f1 = soda_avg[-1]
        results[eval_key] = soda_avg_f1
    return results

if __name__ == '__main__':

    p = 'save/exp_table/yc2_tsn_pdvc_v_2023-03-01-12-19-09/230303-092818_yc2_tsn_pdvc_v_2023-03-01-12-19-09_epoch14_num457_alpha1.0.json_rerank_alpha1.0_temp2.0.json'
    ref = ['data/yc2/captiondata/yc2_val.json']
    ref_para = ['data/yc2/captiondata/para/para_yc2_val.json']
    score=eval_soda(p, ref, verbose=True)
    print(score)
    # para_score = get_para_score(p, referneces=ref_para)
    # print(para_score)

    # p_new = '../save/old/cfgs--base_config_v2_0427--anet_c3d_pdvc_seed358/2021-08-21-21-47-13_debug_2021-08-21_20-46-20_epoch8_num4917_score0_top1000.json'
    # p_vitr= '../save/old/cfgs--base_config_v2_0427--anet_c3d_pdvc_seed358/2021-08-21-21-47-20_cfgs--base_config_v2_0427--anet_c3d_pdvc_seed358_epoch8_num4917_score0_top1000.json.tmp'
    
    # for p in [p_new, p_vitr]:
    #     print('\n')
    #     print(p)
    #     ref_list = ['data/anet/captiondata/val_1.json', 'data/anet/captiondata/val_2.json']
    #     score=eval_soda(p, ref_list, verbose=False)
    #     print(score)
    #     para_score = get_para_score(p, referneces=['../data/anet/captiondata/para/anet_entities_val_1_para.json', '../data/anet/captiondata/para/anet_entities_val_2_para.json'])
    #     print(para_score)


        # metric = ['Meteor', 'Cider']
        # score_type = ['standard_score', 'precision_recall', 'paragraph_score']
        # dvc_score = soda3.eval_tool(predictions=[p], referneces=ref_list, metric=metric,score_type=score_type)[0]
