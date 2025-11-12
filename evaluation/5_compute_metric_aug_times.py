%%capture
### Preprocessing
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from evaluation_metric.prediction_model import GRUClassifierPack,train_model,evaluate_auc

#### Training Data
center2site = {3: 2, 4: 2, 1: 1, 2: 2, 5: 3, 7: 5, 6: 5, 9: 6, 8: 4}
outcome = 'ce_id_cancer'
site = 1
### Preprocessing
#### Training Data
### X will be first 6 month
X = real_df.copy()
X['site'] = X.center.map(center2site)
X['site'] = X.patient_id.map(X.groupby("patient_id").site.max().fillna(0).astype(int))
X = X.loc[X.site != 4]
X.drop(columns='site',inplace = True)
assert outcome in X.columns, "Outcome not in Predictor"
y1 = X.groupby("patient_id")[outcome].max().values
print(f"Outcome numbers: {y1.sum()}")
y1 = torch.from_numpy(y1).long()
X.drop(columns=outcome,inplace=True)
assert outcome not in X.columns, "Outcome still in Predictor"
X['gap'] = np.exp(X.gap) - 1
X.loc[X.groupby('patient_id').head(1).index,'gap'] = 0
six_month = X.groupby('patient_id').gap.cumsum()
X = X.loc[(six_month <= 30*6),]
X_tensor,_,_,X_masking = dp.partition_multi_seq(X,1,'patient_id',max_len=20)

assert y1.shape[0] == X_tensor.shape[0], "patient number do not match"

#### Testing Data
X_test = test_data.copy()
X_test['site'] = X_test.center.map(center2site)
X_test['site'] = X_test.patient_id.map(X_test.groupby("patient_id").site.max().fillna(0).astype(int))
X_test = X_test.loc[X_test.site != 4]
X_test_site = X_test.copy().groupby("patient_id").site.max().values
X_test.drop(columns='site',inplace = True)
y1_test = X_test.groupby("patient_id")[outcome].max().values
y1_test = torch.from_numpy(y1_test).long()
print(f"Outcome numbers: {y1_test.sum()}")
X_test.drop(columns = outcome,inplace = True)
X_test['gap'] = np.exp(test_data.gap) - 1
X_test.loc[X_test.groupby('patient_id').head(1).index,'gap'] = 0
six_month = X_test.groupby('patient_id').gap.cumsum()
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
model1 = GRUClassifierPack(input_dim=F, hidden_dim=128, num_layers=1, num_classes=2)
train_ds = TensorDataset(X_tensor[train_idx,...],1-X_masking[train_idx,:,1],y1[train_idx])
val_ds = TensorDataset(X_tensor[val_idx,...],1-X_masking[val_idx,:,1],y1[val_idx])
train_loader = DataLoader(train_ds,
                        batch_size=128,shuffle=True)
val_loader = DataLoader(val_ds,
                        batch_size=128,shuffle=True)
#### running
trained_model1 = train_model(
    model1,
    train_loader,
    val_loader,
    num_epochs=50,
    lr=1e-3,
    patience=5,
    device= 'cuda:0'
)

%%capture
#### Augmenting Data
site = 1
aug_size = 5
pos_only = True


#### Additional Data
### X will be first 6 month
S = pd.read_csv(f"/home/jeff/Documents/TimeAutoDiff/Dataset/save/aug/site{site}.csv.gz")
if "site" in S.columns: S.drop(columns="site",inplace=True)
assert outcome in S.columns, "Outcome not in Predictor"
y1_s = S.groupby("patient_id")[outcome].max().values
print(f"Outcome numbers: {y1_s.sum()}")
y1_s = torch.from_numpy(y1_s).long()
S.drop(columns=outcome,inplace=True)
assert outcome not in S.columns, "Outcome still in Predictor"
S['gap'] = np.exp(S.gap) - 1
S.loc[S.groupby('patient_id').head(1).index,'gap'] = 0
six_month = S.groupby('patient_id').gap.cumsum()
S = S.loc[(six_month <= 30*6),]
S_tensor,_,_,S_masking = dp.partition_multi_seq(S,1,'patient_id',max_len=20)

if pos_only:
    pos_idx = np.where(y1_s > 0)[0]
    S_tensor = S_tensor[pos_idx,...]
    S_masking = S_masking[pos_idx,...]
    y1_s = y1_s[pos_idx]

assert y1_s.shape[0] == S_tensor.shape[0], "patient number do not match"

aug_idx = np.random.choice(S_tensor.shape[0],S_tensor.shape[0]//10 * aug_size) # size of augmenting 

### Augmenting
S_tensor = torch.concat([X_tensor,S_tensor[aug_idx]])
S_masking = torch.concat([X_masking,S_masking[aug_idx]])
y1_s = torch.concat([y1,y1_s[aug_idx]])

# Training
#### Validation split
train_idx,val_idx = train_idx, val_idx = train_test_split(
    np.arange(S_tensor.shape[0]),
    test_size=0.1,
    random_state=0,
    stratify=y1_s.numpy()    # or stratify=y1 if y1 is a numpy array
)
#### Defining model
N,T,F = S_tensor.shape
model2 = GRUClassifierPack(input_dim=F, hidden_dim=128, num_layers=1, num_classes=2)
train_ds = TensorDataset(S_tensor[train_idx,...],1-S_masking[train_idx,:,1],y1_s[train_idx])
val_ds = TensorDataset(S_tensor[val_idx,...],1-S_masking[val_idx,:,1],y1_s[val_idx])
train_loader = DataLoader(train_ds,
                        batch_size=128,shuffle=True)
val_loader = DataLoader(val_ds,
                        batch_size=128,shuffle=True)
#### running
trained_model2 = train_model(
    model2,
    train_loader,
    val_loader,
    num_epochs=50,
    lr=1e-3,
    patience=5,
    device= 'cuda:0'
)

test_ds = TensorDataset(X_test_tensor, 1 - X_test_masking[...,1], y1_test)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

auc = evaluate_auc(trained_model2.eval(), test_loader)
# os.system(f"echo {auc} >> {os.path.join(args.save_path,f"{args.model_name}_{args.outcome}_auc.txt")}")
print(f"Test augment ROC AUC: {auc:.4f}")

for s in range(1,7):
    if y1_test[X_test_site==s].sum()==0: 
        continue
    test_ds = TensorDataset(X_test_tensor[X_test_site==s,...], 1 - X_test_masking[X_test_site==s,:,1], y1_test[X_test_site==s])
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    auc = evaluate_auc(trained_model2.eval(), test_loader)
    # os.system(f"echo {auc} >> {os.path.join(args.save_path,f"{args.model_name}_{args.outcome}_auc.txt")}")
    print(f"Site {s} -Test augment ROC AUC: {auc:.4f}")

test_ds = TensorDataset(X_test_tensor, 1 - X_test_masking[...,1], y1_test)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

auc = evaluate_auc(trained_model1.eval(), test_loader)
# os.system(f"echo {auc} >> {os.path.join(args.save_path,f"{args.model_name}_{args.outcome}_auc.txt")}")
print(f"Test ROC AUC: {auc:.4f}")

for s in range(1,7):
    if y1_test[X_test_site==s].sum()==0: 
        continue
    test_ds = TensorDataset(X_test_tensor[X_test_site==s,...], 1 - X_test_masking[X_test_site==s,:,1], y1_test[X_test_site==s])
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    auc = evaluate_auc(trained_model1.eval(), test_loader)
    # os.system(f"echo {auc} >> {os.path.join(args.save_path,f"{args.model_name}_{args.outcome}_auc.txt")}")
    print(f"Site {s} -Test ROC AUC: {auc:.4f}")