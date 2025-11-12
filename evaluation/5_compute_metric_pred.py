import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from evaluation_metric.prediction_model import GRUClassifierPack,train_model,evaluate_auc
import argparse, os
from model import DP as dp
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--real_path","-R",type=str,default="/home/jeff/Documents/TimeAutoDiff/Dataset/hiv_test.csv.gz")
    args_parser.add_argument("--test_path","-T",type=str,required=True)
    args_parser.add_argument("--save_path","-S",type=str,default="./eval/pred/")
    args_parser.add_argument("--outcome","-O",type=str,default="ce_id_cardiovascular")
    args_parser.add_argument("--seed",type=int,default=0)
    args_parser.add_argument("--model_name","-M",type=str,required=True)
    args = args_parser.parse_args()

    assert os.path.exists(args.real_path), "Real data path not exist"
    assert os.path.exists(args.test_path), "Test data path not exist"
    os.makedirs(args.save_path,exist_ok=True)

    ### Loading data
    test_data = pd.read_csv(args.real_path).fillna(0)  # for testing
    train_data = pd.read_csv(args.test_path).fillna(0) # for training
    train_data['date'] = train_data.index
    center2site = {3: 2, 4: 2, 1: 1, 2: 2, 5: 3, 7: 5, 6: 5, 9: 6, 8: 4}
    
    ### Preprocessing
    #### Training Data
    ### X will be first 6 month
    X = train_data.copy()
    X['site'] = X.center.map(center2site)
    X['site'] = X.patient_id.map(X.groupby("patient_id").site.max().fillna(0).astype(int))
    X = X.loc[X.site != 4]
    for s in range(1,7):
        outcome_count = X[X.site == s].groupby("patient_id")[args.outcome].max().values
        if outcome_count.sum() == 0:
            X = X.loc[X.site != s]
    X.drop(columns='site',inplace = True)
    X['gap'] = np.exp(X.gap) - 1
    X.loc[X.groupby('patient_id').head(1).index,'gap'] = 0
    six_month = X.groupby('patient_id').gap.cumsum()
    assert args.outcome in X.columns, "Outcome not in Predictor"
    # X.loc[(six_month<30*6),args.outcome] = 0 ## mask six_month results
    y1 = X.loc[(six_month>=30*6)].groupby("patient_id")[args.outcome].max()
    X = X.loc[X.patient_id.isin(y1.index),]
    print(f"Outcome numbers: {y1.sum()}")
    y1 = torch.from_numpy(y1.values).long()
    X.drop(columns=args.outcome,inplace=True)
    assert args.outcome not in X.columns, "Outcome still in Predictor"
    # X['gap'] = np.exp(X.gap) - 1
    # X.loc[X.groupby('patient_id').head(1).index,'gap'] = 0
    # six_month = X.groupby('patient_id').gap.cumsum()
    X = X.loc[(six_month <= 30*6),]
    X_tensor,_,_,X_masking = dp.partition_multi_seq(X,1,'patient_id',max_len=20)

    assert y1.shape[0] == X_tensor.shape[0], "patient number do not match"

    #### Testing Data
    X_test = test_data.copy()
    X_test['site'] = X_test.center.map(center2site)
    X_test['site'] = X_test.patient_id.map(X_test.groupby("patient_id").site.max().fillna(0).astype(int))
    X_test = X_test.loc[X_test.site != 4]
    for s in range(1,7):
        outcome_count = X_test[X_test.site == s].groupby("patient_id")[args.outcome].max().values
        if outcome_count.sum() == 0:
            X_test = X_test.loc[X_test.site != s]
    X_test.drop(columns='site',inplace = True)
    X_test['gap'] = np.exp(test_data.gap) - 1
    X_test.loc[X_test.groupby('patient_id').head(1).index,'gap'] = 0
    six_month = X_test.groupby('patient_id').gap.cumsum()
    # X_test.loc[(six_month<30*6),args.outcome] = 0 ## mask six month outcome
    y1_test = X_test.loc[(six_month>=30*6),].groupby("patient_id")[args.outcome].max()
    X_test = X_test.loc[X_test.patient_id.isin(y1_test.index),]
    y1_test = torch.from_numpy(y1_test.values).long()
    print(f"Outcome numbers: {y1_test.sum()}")
    X_test.drop(columns = args.outcome,inplace = True)
    assert args.outcome not in X_test.columns, "Outcome still in Predictor"
    # X_test['gap'] = np.exp(test_data.gap) - 1
    # X_test.loc[X_test.groupby('patient_id').head(1).index,'gap'] = 0
    # six_month = X_test.groupby('patient_id').gap.cumsum()
    X_test = X_test.loc[(six_month <= 30*6),]
    X_test_tensor,_,_,X_test_masking = dp.partition_multi_seq(X_test,1,'patient_id',max_len=20)

    assert X_tensor.shape[-1] == X_test_tensor.shape[-1], "Training Testing shape mismatch"
    # Training
    #### Validation split
    train_idx,val_idx = train_idx, val_idx = train_test_split(
        np.arange(X_tensor.shape[0]),
        test_size=0.1,
        random_state=0,
        stratify=y1.numpy()    # or stratify=y1 if y1 is a numpy array
    )
    #### Defining model
    N,T,F = X_tensor.shape
    model = GRUClassifierPack(input_dim=F, hidden_dim=128, num_layers=1, num_classes=2)
    train_ds = TensorDataset(X_tensor[train_idx,...],1-X_masking[train_idx,:,1],y1[train_idx])
    val_ds = TensorDataset(X_tensor[val_idx,...],1-X_masking[val_idx,:,1],y1[val_idx])
    train_loader = DataLoader(train_ds,
                            batch_size=128,shuffle=True)
    val_loader = DataLoader(val_ds,
                            batch_size=128,shuffle=True)
    #### running
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=50,
        lr=1e-3,
        patience=5,
        device= 'cuda:0'
    )

    #### testing
    test_ds = TensorDataset(X_test_tensor, 1 - X_test_masking[...,1], y1_test)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    auc = evaluate_auc(trained_model.eval(), test_loader)
    os.system(f"echo {auc} >> {os.path.join(args.save_path,f"{args.model_name}_{args.outcome}_auc.txt")}")
    print(f"Test ROC AUC: {auc:.4f}")