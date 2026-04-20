#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[ ]:


dt_org = pd.read_csv("./hiv_train.csv.gz")
for col in ['weight','height','cd4_v','rna_v']:
    dt_org[col] = pd.qcut(dt_org[col], q=30, labels=False, duplicates='drop') + 1
dt_org.head()


# In[3]:


## Extract condition
conditions = {}
groups = dt_org.groupby("patient_id")

for pid, p in groups:
    s =  p.iloc[0, -5:]
    s.index = ['year', 'center', 'male_y', 'age', 'mode']
    s = s[['year', 'age', 'center', 'male_y',  'mode']]
    s['age'] = s['age'] + 17.9
    collapsed = [f"{k}:{v:.0f}" for k, v in s.items()]
    conditions[pid] = collapsed


# In[4]:


## Extract Date
_dates = {}
_ages = {}
_epoch_times = {}
_weeks = {}
for pid,p in groups:
    _dates[pid] = p.loc[:,'date'].values
    _epoch_times[pid] = pd.to_datetime(_dates[pid]).astype('int64') // 10**9 #days since 1970
    _weeks[pid] = _epoch_times[pid]//604800
    age = pd.to_datetime(_dates[pid]) - pd.to_datetime(_dates[pid][0])
    age = age.days//365
    age = age.to_list()
    age[0] = -1
    _ages[pid] = age

## Extract Time Gap
time_gaps = {}
for pid, p in groups:
    log_day = p['gap'].values[2:]
    time_gaps[pid] = np.exp(log_day).astype(int)


# In[5]:


## Extract categorical
categorical_code = {}
categorical_vals = {}

for pid, p in groups:
    codes = []
    vals = []
    
    for idx, row in p.iloc[1:, 5:-7].iterrows():
        row = row[row.fillna(0.)>0]
        codes.append(row.index.to_list())                 # row identifier (index)
        vals.append([0. for x in row.index])       # numeric values of first 4 columns
    
    categorical_code[pid] = codes
    categorical_vals[pid] = vals


# In[6]:


## Extract continuous
continuous_code = {}
continuous_vals = {}

for pid, p in groups:
    codes = []
    vals = []
    
    for idx, row in p.iloc[1:, :4].iterrows():
        row = row.dropna()
        # codes.append(row.index.to_list())                 # row identifier (index)
        code = [f"{c}:{v}" for c,v in zip(row.index.to_list(),row.to_list())]
        codes.append(code)
        vals.append(row.to_list())       # numeric values of first 4 columns
    
    continuous_code[pid] = codes
    continuous_vals[pid] = vals


# In[7]:


### ensemble sequence

#### get visit length
lengths = {}
for pid, p in groups:
    lengths[pid] = p.shape[0]

#### ensemble 
concept_ids = {}
concept_values = {}
concept_value_masks = {}
visit_concept_orders = visit_rank_orders = {}
visit_segments = {}
priorities = {}
concept_orders = {}
record_ranks = {}
orders = record_orders = {}
dates = {}
epoch_times = {}
ages = {}
num_of_visits = {}
num_of_concepts = {}
mlm_skip_values = {}
units = event_group_ids = {}
visit_concept_ids = {}
for pid, l in lengths.items():
    ### put categorical and continuous together
    codes = [x + y for x, y in zip(categorical_code[pid],continuous_code[pid])]
    vals = [x + y for x, y in zip(categorical_vals[pid],continuous_vals[pid])]
    assert len(codes) == len(vals), "code's length and value's length not matching"
    assert len(np.concatenate(codes)) == len(np.concatenate(vals)), "code's length and value's length not matching"
    ### add [VS],outpatient,[VE]
    for c,v in zip(codes,vals):
        c.insert(0,"outpatient");c.insert(0,"[VS]");c.append("[VE]")
        v.insert(0,0);v.insert(0,0);v.append(0)
    ### append timegap:
    for c,v,d in zip(codes,vals,time_gaps[pid]):
        # c.append(f"D{d}" if d < 365 else "LT") #days but if longer than 365, LT
        c.append(f"D{d}")
        v.append(0)
    ### add conditions
    codes.insert(0,conditions[pid])
    vals.insert(0,[0]*len(conditions[pid]))

    ### assign to concept_ids, vals
    assert len(np.concatenate(codes)) == len(np.concatenate(vals)), "code's length and value's length not matching"
    concept_ids[pid] = np.concatenate(codes).tolist()
    concept_values[pid] = np.concatenate(vals).tolist()

    ### meta data
    code_lengths = [len(x) for x in codes]
    num_of_visits[pid] = len(code_lengths) - 1
    num_of_concepts[pid] = len(concept_ids[pid])
    concept_value_masks[pid] = np.clip(concept_values[pid],a_max=1,a_min=0)
    mlm_skip_values[pid] = [0]*len(concept_ids[pid])
    event_group_ids[pid] = ['N/A']*len(concept_ids[pid])
    visit_concept_ids[pid] = ['outpatient']*len(concept_ids[pid])

    ### populate age
    age = [[d] * c for d,c in zip(_ages[pid],code_lengths)]
    age[0]= [-1]*code_lengths[0] #condition has -1 age by design
    ages[pid] = np.concatenate(age).tolist()

    ### populate date, epoch_time
    date = [[d] * c for d,c in zip(_weeks[pid],code_lengths)]
    date[0]= [0]*code_lengths[0]
    dates[pid] = np.concatenate(date).tolist()
    epoch = [[d] * c for d,c in zip(_epoch_times[pid],code_lengths)]
    epoch[0] = [0]*code_lengths[0]
    epoch_times[pid] = np.concatenate(epoch).tolist()
    
    ### summarize concept order
    concept_order = [[1] * c for c in code_lengths]
    for r,c in zip(concept_order,codes):
        r[0] = 0 # VS is 0
        r[1] = 0 # outpatient is 0
        r[-1] = 0 # D is 0 if exist
        r[c.index("[VE]") if "[VE]" in c else -1] = 2
    concept_order[0] = [0] * code_lengths[0] # condition is 0
    concept_orders[pid] = np.concatenate(concept_order).tolist()

    ### summarize record rank
    record_rank = [[0] * c for c in code_lengths]
    for r,c in zip(record_rank,codes):
        r[0] = 1 # VS is 1
        r[1] = 1 # outpatient is 1
        r[2] = 1 # code has the same ranks
        r[-1] = 1 # D is 1 if exist
        r[c.index("[VE]") if "[VE]" in c else -1] = 1
    record_rank[0] = [1] * code_lengths[0]
    record_rank = np.concatenate(record_rank).cumsum()
    record_ranks[pid] = record_rank.tolist()

    ### summarize visit segments, visit_rank_order, record_orders
    record_orders[pid] = list(range(1,len(np.concatenate(codes).tolist())+1))
    visit_rank_orders[pid] = [i for i, c in enumerate(code_lengths) for _ in range(l+1)]
    segment = [0] * code_lengths[0] # conditions are 0
    segment += [((i+1)%2) + 1 for i, c in enumerate(code_lengths[1:]) for _ in range(l)] # alternating 2,1,2...
    visit_segments[pid] = segment

    ### summarize priority
    YEAR_TOKEN_PRIORITY = -10
    AGE_TOKEN_PRIORITY = -9
    GENDER_TOKEN_PRIORITY = -8
    RACE_TOKEN_PRIORITY = -7
    ATT_TOKEN_PRIORITY = -3
    VS_TOKEN_PRIORITY = -2
    VISIT_TYPE_TOKEN_PRIORITY = -1
    FIRST_VISIT_HOUR_TOKEN_PRIORITY = -0.5
    DEFAULT_PRIORITY = 0
    DISCHARGE_TOKEN_PRIORITY = 100
    DEATH_TOKEN_PRIORITY = 199
    VE_TOKEN_PRIORITY = 200
    PREDICTION_TOKEN_PRIORITY = 1000
    priority = [[0] * c for c in code_lengths]
    for r,c in zip(priority,codes):
        r[0] = VS_TOKEN_PRIORITY # VS 
        r[1] = VISIT_TYPE_TOKEN_PRIORITY # outpatient
        r[-1] = ATT_TOKEN_PRIORITY # D
        r[c.index("follow_mode_death") if "follow_mode_death" in c else -1] = DEATH_TOKEN_PRIORITY
        r[c.index("[VE]") if "[VE]" in c else -1] = VE_TOKEN_PRIORITY # VE 
    priority[0] = list(range(-10,0))[:code_lengths[0]] # conditions
    priorities[pid] = np.concatenate(priority).tolist()
     


# In[8]:


dict_lists = [concept_ids, visit_segments,
       orders, dates, ages, visit_concept_orders, num_of_visits,
       num_of_concepts, concept_value_masks, concept_values,
       mlm_skip_values, priorities, visit_concept_ids,
       visit_rank_orders, concept_orders, record_ranks, units,
       event_group_ids, epoch_times]
dict_names = ['concept_ids', 'visit_segments',
       'orders', 'dates', 'ages', 'visit_concept_orders', 'num_of_visits',
       'num_of_concepts', 'concept_value_masks', 'concept_values',
       'mlm_skip_values', 'priorities', 'visit_concept_ids',
       'visit_rank_orders', 'concept_orders', 'record_ranks', 'units',
       'event_group_ids', 'epoch_times']

dt = pd.concat([pd.Series(p) for p in dict_lists],axis=1)
dt.columns = dict_names
dt.reset_index(names=["person_id"],inplace=True)
dt["cohort_member_id"] = dt.person_id
dt.head()


# In[9]:


dt.concept_ids[0]


# In[10]:


### checking
assert sum(dt.concept_ids.map(len) - dt.num_of_concepts) == 0, "concept length not matching"
assert sum(dt.concept_ids.map(lambda x: x.count("[VE]")) - dt.num_of_visits) == 0, "number of visit not matching"
assert sum(dt.concept_ids.map(len) - dt.concept_values.map(len)) == 0, "concept length and concept values length not matching"

print(f"max length of sequence: {max(dt.num_of_concepts)}")
dt.num_of_concepts.quantile(0.99)


# In[1]:


import os
os.makedirs("./pretrain",exist_ok=True)
dt.to_parquet("./pretrain/patient_sequence.parquet",index=False)


# In[3]:


### Check generated data
import pandas as pd

# dt = pd.read_csv("/home/jeff/Documents/cehrgpt/CCASAnet/generated/top_p10000/generated_sequences/cehrgpt_generate.csv.gz")
dt = pd.read_parquet("./pretrain/patient_sequence.parquet")
print(dt.shape)
display(dt.head())
print(dt.concept_ids[0])

