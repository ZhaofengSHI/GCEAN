from densevid_eval3.evaluate2018 import main as eval2018
from densevid_eval3.evaluate2021 import main as eval2021

def eval_dvc(json_path, reference, no_lang_eval=False, topN=1000, version='2018'):
    args = type('args', (object,), {})()
    args.submission = json_path
    args.max_proposals_per_video = topN
    args.tious = [0.3,0.5,0.7,0.9]
    args.verbose = False
    args.no_lang_eval = no_lang_eval
    args.references = reference
    eval_func = eval2018 if version=='2018' else eval2021
    score = eval_func(args)
    return score

if __name__ == '__main__':
    p = 'save/exp_table/yc2_tsn_pdvc_v_2023-03-01-12-19-09/230303-092818_yc2_tsn_pdvc_v_2023-03-01-12-19-09_epoch14_num457_alpha1.0.json_rerank_alpha1.0_temp2.0.json'
    ref = ['data/yc2/captiondata/yc2_val.json']
    score = eval_dvc(p, ref, no_lang_eval=False, version='2018')
    score = {k: sum(v) / len(v) for k, v in score.items()}
    print(score)