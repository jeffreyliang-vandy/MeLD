import os
import numpy as np
import pandas as pd
import argparse
from scipy.stats import kstest, mannwhitneyu



if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--real_path","-R",type=str,default="/home/jeff/Documents/TimeAutoDiff/Dataset/hiv_missing_test.csv.gz")
    args_parser.add_argument("--test_path","-T",type=str,required=True)
    args_parser.add_argument("--save_path","-S",type=str,default="./eval/corr/")
    args_parser.add_argument("--seed",type=int,default=0)
    args_parser.add_argument("--model_name","-M",type=str,required=True)
    args_parser.add_argument("--serial",type=str,default="")
    args = args_parser.parse_args()

    assert os.path.exists(args.real_path), "Real data path not exist"
    assert os.path.exists(args.test_path), "Test data path not exist"
    os.makedirs(args.save_path,exist_ok=True)

    real_missing = pd.read_csv(args.real_path)
    real_missing = real_missing[['patient_id','cd4_v','rna_v','weight','height']]
    syn_df = pd.read_csv(args.test_path)[['patient_id','cd4_v','rna_v','weight','height']]
    if args.model_name == "Real": syn_missing = syn_df
    else: syn_missing = syn_df.isna() * 1
    syn_missing['patient_id'] = syn_df['patient_id']

    ## Get perpatient missing
    ### calculate per-patient missing percentage
    real_missing_per = real_missing.groupby('patient_id').mean()
    real_missing_per['total'] = real_missing_per.mean()
    syn_missing_per = syn_missing.groupby('patient_id').mean()
    syn_missing_per['total'] = syn_missing_per.mean()

    mean_per = pd.DataFrame(syn_missing_per.mean().to_dict(),index=['mean_per'])
    mean_overall = pd.DataFrame(syn_missing.drop(columns="patient_id").mean().to_dict(),index=['mean_overall'])

    #ks test
    ks_test = {}
    for v in ['cd4_v','rna_v','weight','height','total']:
        ks_test[v] = kstest(real_missing_per[v],syn_missing_per[v])[1]

    ks_test = pd.DataFrame(ks_test,index=['ks-test'])

    #wilcoxon test
    mutest = {}
    for v in ['cd4_v','rna_v','weight','height','total']:
        mutest[v] = mannwhitneyu(real_missing_per[v],syn_missing_per[v])[1]

    mutest = pd.DataFrame(mutest,index=['mannwhitneyu-test'])

    # Create a summary table
    summary_df = pd.concat([mean_overall,mean_per,ks_test,mutest],axis=0).reset_index(names="metrics")
    file_path = os.path.join(args.save_path,f"{args.model_name}_missing_summary.csv")
    summary_df.to_csv(
        file_path,
        mode="a" if os.path.exists(file_path) else "w",  # append if exists, else write
        header=not os.path.exists(file_path),            # write header only if new file
        index=False
    )
