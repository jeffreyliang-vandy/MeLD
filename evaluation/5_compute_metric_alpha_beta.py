from sklearn.preprocessing import OneHotEncoder
from synthcity.metrics import eval_detection, eval_performance, eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader
import pandas as pd
import numpy as np
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--dataname','-D', type=str, default='hiv')
parser.add_argument('--model','-M', type=str, default='model')
parser.add_argument('--syn_path','-S', type=str, default = None, help='The file path of the synthetic data')
parser.add_argument('--real_path','-R', type=str, default = None, help='The file path of the synthetic data')

args = parser.parse_args()


if __name__ == '__main__':
    syn_data = pd.read_csv(args.syn_path).drop(columns=['date','patient_id'])
    real_data = pd.read_csv(args.real_path).drop(columns=['date','patient_id'])

    min_len = np.min([syn_data.shape[0],real_data.shape[0]])

    syn_data = syn_data.sample(frac=1).iloc[:50000,:].fillna(0)
    real_data = real_data.sample(frac=1).iloc[:50000,:].fillna(0)

    result = []

    print('=========== All Features ===========')
    # print('Data shape: ', syn_data.head())

    X_syn_loader = GenericDataLoader(syn_data)
    X_real_loader = GenericDataLoader(real_data)
    print("Evaluating...")
    quality_evaluator = eval_statistical.AlphaPrecision()
    qual_res = quality_evaluator.evaluate(X_real_loader, X_syn_loader)
    qual_res = {
        k: v for (k, v) in qual_res.items() if "naive" in k
    }  # use the naive implementation of AlphaPrecision
    qual_score = np.mean(list(qual_res.values()))

    print('alpha precision: {:.6f}, beta recall: {:.6f}'.format(qual_res['delta_precision_alpha_naive'], qual_res['delta_coverage_beta_naive'] ))

    Alpha_Precision_all = qual_res['delta_precision_alpha_naive']
    Beta_Recall_all = qual_res['delta_coverage_beta_naive']

    save_dir = f'eval/quality/{args.dataname}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f'{save_dir}/{args.model}.txt', 'w') as f:
        f.write(f'{Alpha_Precision_all}\n')
        f.write(f'{Beta_Recall_all}\n')