import tensorflow as tf
import time
import os, argparse
import numpy as np
from B_stage1_code_updated_w_gr import CodeModel
from C_stage1_interval_bin import IntervalModel
from E_gan_updated2_w_gr import Generator, DiscriminatorTest
import tensorflow_probability as tfp
from scipy.stats import pearson3
from tqdm import tqdm
import pickle as pk
from sklearn.model_selection import train_test_split

prefix = './data/datasets/synteg/'

from synteg_config import Config

def gen_dataset(k):
    
    path = "./results/synteg/" + 'syn' + args.serial + '_' + args.b_ckpt + '_'+ args.c_ckpt + '_'+ args.e_ckpt + '_' + str(k) + '/'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    
    config = Config()
#     code_model = CodeModel(config.lstm_dim, config.n_layer, config.vec_dims, config.num_code, config.max_num_visit)
    code_model = CodeModel(config.lstm_dim, config.n_layer, config.vec_dims, config.num_code, config.max_num_visit, config.batch_size, \
                           config.interval_space,\
                      config.age_space, config.gender_space, config.country_space, \
                        config.center_space, config.birth_space, config.aidsmode_space) #
#     interval_model = IntervalModel(config.lstm_dim, config.n_layer, config.num_code, config.max_num_visit)
    interval_model = IntervalModel(config.lstm_dim, config.n_layer, config.vec_dims, config.num_code, config.max_num_visit, config.batch_size, \
                                   config.interval_space,\
                      config.age_space, config.gender_space, config.country_space, \
                        config.center_space, config.birth_space, config.aidsmode_space) #
    generator = Generator(config.G_DIMS)

    checkpoint_directory_code = "./save/synteg/training_checkpoints_stage1_code_updated_w_gr"
    checkpoint_prefix_code = os.path.join(checkpoint_directory_code, "ckpt")

    checkpoint_directory_interval = "./save/synteg/training_checkpoints_stage1_interval_bin"
    checkpoint_prefix_interval = os.path.join(checkpoint_directory_interval, "ckpt")

    checkpoint_directory_generator = "./save/synteg/training_checkpoints_gan_updated2_" + args.serial
    checkpoint_prefix_generator = os.path.join(checkpoint_directory_generator, "ckpt")

    # # checkpoint_directory_filter = "training_checkpoints_filter_updated2_" + args.serial + "_epoch" + args.epoch
    # checkpoint_directory_filter = "training_checkpoints_filter_updated2_" + args.serial + "_epoch2"
    # checkpoint_prefix_filter = os.path.join(checkpoint_directory_filter, "ckpt")

    checkpoint_code = tf.train.Checkpoint(model=code_model)
    checkpoint_code.restore(checkpoint_prefix_code + '-' + args.b_ckpt).expect_partial()

    checkpoint_interval = tf.train.Checkpoint(model=interval_model)
    checkpoint_interval.restore(checkpoint_prefix_interval + "-" + args.c_ckpt).expect_partial()

    checkpoint_generator = tf.train.Checkpoint(generator=generator)
    checkpoint_generator.restore(checkpoint_prefix_generator + '-' + args.e_ckpt).expect_partial()

    # checkpoint_filter = tf.train.Checkpoint(discriminator=fil_2, model=fil_1)
    # checkpoint_filter.restore(checkpoint_prefix_filter + '-20').expect_partial()
    # # checkpoint_generator.restore(tf.train.latest_checkpoint(checkpoint_directory_generator)).expect_partial()

    @tf.function
    def insert_step(step_code, step_age, step_gender, step_country,\
                    step_center, step_birth, step_aidsmode, \
                     step_interval): #
        step_latent = code_model.gen(step_code, step_age, step_interval, step_gender, step_country,\
                                     step_center, step_birth, step_aidsmode, \
                                     )
        step_latent = tf.tile(tf.squeeze(step_latent), [25, 1])
        step_z = tf.random.normal(shape=[step_latent.shape.as_list()[0], config.Z_DIM])
        step_insert = tf.math.round(generator(step_z, step_latent))
        return step_insert

    @tf.function
    def interval_step(step_code, step_age, step_interval, step_gender, step_country, \
                      step_center, step_birth, step_aidsmode,\
                      next_code):
        return interval_model.gen(step_code, step_age, step_interval, step_gender, step_country,\
                                  step_center, step_birth, step_aidsmode, \
                                  next_code)

    # @tf.function
    # def filter_step(batch_code):
    #     return tf.squeeze(fil_2(fil_1.fil_gen(batch_code)))

    @tf.function
    def get_starter(starter_age, starter_gender, starter_country, \
                    starter_center, starter_birth,starter_aidsmode):
        starter_latent = code_model.gen_starter(starter_age, starter_gender, starter_country,\
                                                starter_center, starter_birth, starter_aidsmode \
                                                )
        starter_latent = tf.tile(tf.squeeze(starter_latent), [25, 1])
        starter_z = tf.random.normal(shape=[starter_latent.shape.as_list()[0], config.Z_DIM])
        starter_insert = tf.math.round(generator(starter_z, starter_latent))
        return starter_insert
    
    def gen_batch(starter_age, starter_gender, starter_country, \
                  starter_center, starter_birth, starter_aidsmode):
        for lstm_layer in code_model.lstm.layer:
            lstm_layer.reset_states()
        for lstm_layer in interval_model.lstm.layer:
            lstm_layer.reset_states()
        batch_code = np.zeros((config.batch_size, config.max_num_visit, config.max_code_visit), dtype='int32')

        # batch_age = np.zeros((config.batch_size, config.max_num_visit))
        batch_age = np.array([list(starter_age[ii]*np.ones( config.max_num_visit)) for ii in range(config.batch_size)]).astype('int')#

        batch_gender = np.array([list(starter_gender[ii]*np.ones( config.max_num_visit)) for ii in range(config.batch_size)]).astype('int')#
        batch_country = np.array([list(starter_country[ii]*np.ones( config.max_num_visit)) for ii in range(config.batch_size)]).astype('int')#
        batch_center = np.array([list(starter_center[ii]*np.ones( config.max_num_visit)) for ii in range(config.batch_size)]).astype('int')#
        batch_birth = np.array([list(starter_birth[ii]*np.ones( config.max_num_visit)) for ii in range(config.batch_size)]).astype('int')#
        batch_aidsmode = np.array([list(starter_aidsmode[ii]*np.ones( config.max_num_visit)) for ii in range(config.batch_size)]).astype('int')#

        # TODO: Need to add a starter interval
        batch_interval = np.zeros((config.batch_size, config.max_num_visit))
        # batch_interval[:,0] = config.interval_space-1

        cumulative_interval = np.zeros(config.batch_size)

        starter_insert = get_starter(tf.convert_to_tensor(starter_age, dtype=tf.int32),\
                                        tf.convert_to_tensor(starter_gender, dtype=tf.int32),\
                                        tf.convert_to_tensor(starter_country, dtype=tf.int32),\
                                        tf.convert_to_tensor(starter_center, dtype=tf.int32),\
                                        tf.convert_to_tensor(starter_birth, dtype=tf.int32),\
                                        tf.convert_to_tensor(starter_aidsmode, dtype=tf.int32)).numpy() #
        # batch_score = filter_step(starter_insert).numpy()
        selected_group = starter_insert[:config.batch_size]
        # correct_score = batch_score[:config.batch_size]
        for attempt in range(24):
            num_code_visit = np.sum(selected_group[:, :config.num_code] == 1, axis=-1)
            # violate = np.logical_or(
            #     np.logical_or(num_code_visit == 0, num_code_visit - config.max_code_visit >= -1),
            #     correct_score < 0.2)
            violate = np.logical_or(num_code_visit == 0, num_code_visit - config.max_code_visit >= -1)
            if np.sum(violate) == 0:
                break
            selected_group[violate] = starter_insert[(attempt + 1) * config.batch_size:(attempt + 2) * config.batch_size][
                violate]
            # correct_score[violate] = batch_score[(attempt + 1) * config.batch_size:(attempt + 2) * config.batch_size][
            #     violate]
        insert_code = [np.arange(config.num_code)[selected_group[n, :config.num_code] == 1] for n in
                       np.arange(config.batch_size)]
        try:
            insert_code = np.array(
                [np.pad(w, (0, config.max_code_visit - len(w)), 'constant', constant_values=-1) for w in
                 insert_code]) + 1
        except:
            print([len(w) for w in insert_code])

        ### This part report Error
        batch_code[:, 0] = insert_code
        batch_age[:, 0] = starter_age
#         batch_gender[:, 0] = starter_gender #
#         batch_country[:, 0] = starter_country #
        batch_length = np.ones(config.batch_size)
        updating = np.logical_not(selected_group[:, config.num_code])

        for j_step in range(1, config.max_num_visit):
            if np.sum(updating) == 0:
                break

            batch_length += updating
            step_insert = insert_step(
                tf.convert_to_tensor(batch_code[:, j_step - 1], dtype=tf.int32),
                tf.convert_to_tensor(batch_age[:, j_step - 1], dtype=tf.int32),
                tf.convert_to_tensor(batch_gender[:, j_step - 1], dtype=tf.int32),
                tf.convert_to_tensor(batch_country[:, j_step - 1], dtype=tf.int32),
                tf.convert_to_tensor(batch_center[:, j_step - 1], dtype=tf.int32),
                tf.convert_to_tensor(batch_birth[:, j_step - 1], dtype=tf.int32),
                tf.convert_to_tensor(batch_aidsmode[:, j_step - 1], dtype=tf.int32),
                tf.convert_to_tensor(batch_interval[:, j_step - 1], dtype=tf.float32)).numpy()

            # batch_score = filter_step(step_insert).numpy()
            selected_group = step_insert[:config.batch_size]
            # correct_score = batch_score[:config.batch_size]
            for attempt in range(24):
                num_code_visit = np.sum(selected_group[:, :config.num_code] == 1, axis=-1)
                # violate = np.logical_or(
                #     np.logical_or(num_code_visit == 0, num_code_visit - config.max_code_visit >= -1),
                #     correct_score < 0.2)
                violate = np.logical_or(num_code_visit == 0, num_code_visit - config.max_code_visit >= -1)
                if np.sum(violate[updating]) == 0:
                    break
                selected_group[violate] = step_insert[(attempt + 1) * config.batch_size:(attempt + 2) * config.batch_size][violate]
                # correct_score[violate] = batch_score[(attempt + 1) * config.batch_size:(attempt + 2) * config.batch_size][violate]
            insert_code = [np.arange(config.num_code)[selected_group[n, :config.num_code] == 1] for n in np.arange(config.batch_size)[updating]]
            try:
                insert_code = np.array(
                    [np.pad(w, (0, config.max_code_visit - len(w)), 'constant', constant_values=-1) for w in
                     insert_code]) + 1
            except:
                print([len(w) for w in insert_code])
            batch_code[updating, j_step] = insert_code
            gap_insert = interval_step(
                tf.convert_to_tensor(batch_code[:, j_step - 1], dtype=tf.int32),
                tf.convert_to_tensor(batch_age[:, j_step - 1], dtype=tf.int32),
                # tf.convert_to_tensor(batch_interval[:, j_step - 1] + 1, dtype=tf.float32),
                tf.convert_to_tensor(batch_interval[:, j_step - 1], dtype=tf.float32),
                tf.convert_to_tensor(batch_gender[:, j_step - 1], dtype=tf.int32), # 
                tf.convert_to_tensor(batch_country[:, j_step - 1], dtype=tf.int32), #
                tf.convert_to_tensor(batch_center[:, j_step - 1], dtype=tf.int32),
                tf.convert_to_tensor(batch_birth[:, j_step - 1], dtype=tf.int32),
                tf.convert_to_tensor(batch_aidsmode[:, j_step - 1], dtype=tf.int32),
                tf.convert_to_tensor(batch_code[:, j_step], dtype=tf.int32)).numpy()

            tmp = np.argsort(gap_insert[updating], axis=-1)
            res = []

            ## Do we need this, I want more than 2 years
            for g, t in zip(gap_insert, tmp):
                #### Two year restriction
                t = t[::-1]
                tt = np.arange(config.interval_space)[np.cumsum(g[t]) > 0.95][0]
                g[t[tt + 1:]] = 0
                #### Commented out
                g = tf.convert_to_tensor(g / np.sum(g, keepdims=True))
                dist = tfp.distributions.Categorical(probs=g)
                res.append(dist.sample().numpy())
                # tf.print(np.max(res))
            ## Do we need this, I want more than 2 years
            res = np.array(res)
            # tf.print(np.max(res))
            batch_interval[updating, j_step] = res

            # ## Recover continuous interval
            # parameters = np.array([interval_map.get(x) for x in res])
            # skewness = np.array([p[0] for p in parameters])
            # skewness = np.where(np.isfinite(skewness), skewness, 0)
            # mean = np.array([p[1] for p in parameters])
            # std_dev = np.array([p[2] for p in parameters])
            # res = pearson3.rvs(skew=skewness, loc=mean, scale=std_dev, size=len(skewness))
            # res[res<0] = 0.
            # cumulative_interval[updating] += res
            # batch_age[updating, j_step] = np.clip(batch_age[updating, 0]*5 + (cumulative_interval[updating]/365).astype('int'), 0, 100)//5
            # batch_age[updating, j_step] = np.clip(batch_age[updating, 0] + (cumulative_interval[updating]/365).astype('int'), 0, 88)

            updating = np.logical_and(updating, np.logical_not(selected_group[:, config.num_code]))
        batch_code[np.arange(config.batch_size, dtype='int32'), batch_length.astype('int32') - 1, -1] = config.num_code + 1
        
        return batch_code.astype('int32') - 1, \
            batch_interval.astype('int32'), \
            batch_length.astype('int32'), \
            batch_age.astype('int32'), \
            batch_gender.astype('int32'),\
            batch_country.astype('int32'), \
            batch_center.astype("int32"), \
            batch_birth.astype("int32"), \
            batch_aidsmode.astype("int32")

    code_tmp = []
    interval_tmp = []
    length_tmp = []
    age_tmp = []
    gender_tmp = [] #
    country_tmp = [] #
    center_tmp = []
    birth_tmp = []
    aidsmode_tmp = []
    
    # train_idx = np.load(prefix + 'train_idx.npy', allow_pickle = True).astype(np.int32)
    lengths = np.load(prefix + 'length.npy', allow_pickle = True).astype(np.int32)
    idx = np.arange(len(lengths))[lengths>1]
    train_idx, test_idx = train_test_split(idx,test_size=0.15, random_state=0) #
    age = np.repeat(np.load(prefix + 'age.npy', allow_pickle = True).astype(np.int32)[:,0][train_idx], 4) #
    gender = np.repeat(np.load(prefix + 'gender.npy', allow_pickle = True).astype(np.int32)[:,0][train_idx], 4) #
    country = np.repeat(np.load(prefix + 'country.npy', allow_pickle = True).astype(np.int32)[:,0][train_idx], 4) #
    center = np.repeat(np.load(prefix + 'center.npy', allow_pickle = True).astype(np.int32)[:,0][train_idx], 4)
    birth = np.repeat(np.load(prefix + 'birth.npy', allow_pickle = True).astype(np.int32)[:,0][train_idx], 4)
    aidsmode = np.repeat(np.load(prefix + 'mode.npy', allow_pickle = True).astype(np.int32)[:,0][train_idx], 4)
    
    interval_map = pk.load(open("interval_map.pkl","rb"))
    interval_map = [interval_map[x] for x in interval_map.keys()]
    interval_map = {x:y for x,y in enumerate(interval_map)}
    
    num_patient = len(age)*4
    age = np.concatenate((age, age[:config.batch_size]),axis=0)
    gender = np.concatenate((gender, gender[:config.batch_size]),axis=0)
    country = np.concatenate((country, country[:config.batch_size]),axis=0)
    center = np.concatenate((center, center[:config.batch_size]),axis=0)
    birth = np.concatenate((birth, birth[:config.batch_size]),axis=0)
    aidsmode = np.concatenate((aidsmode, aidsmode[:config.batch_size]),axis=0)
    
    total_records_to_generate = args.n  # Total records specified by the user
    records_generated = 0  # Counter for total records generated

    # n_batch = int(len(age) / config.batch_size)
    n_batch = int(total_records_to_generate / config.batch_size) + 1
    print(f"batch_num {n_batch}", flush=True)
    
    t0 = time.time()
    
    for j in tqdm(range(n_batch), total=n_batch):
        if records_generated >= total_records_to_generate:
            break  # Stop when the required number of records is generated
        
        # Calculate how many records to generate in this batch
        records_remaining = total_records_to_generate - records_generated
        batch_size = min(config.batch_size, records_remaining)  # Adjust batch size if needed

        if j % 10 == 9:
            print("batch %d/%d (%d)" % (j,n_batch,time.time() - t0), flush=True)
            t0 = time.time()
        x, y, z, u, p, q, r, s, t = gen_batch(age[j*config.batch_size:(j+1)*config.batch_size], \
                                              gender[j*config.batch_size:(j+1)*config.batch_size], \
                                                country[j*config.batch_size:(j+1)*config.batch_size], \
                                                center[j*config.batch_size:(j+1)*config.batch_size], \
                                                birth[j*config.batch_size:(j+1)*config.batch_size], \
                                                aidsmode[j*config.batch_size:(j+1)*config.batch_size])
        code_tmp.extend(x)
        interval_tmp.extend(y)
        length_tmp.extend(z)
        age_tmp.extend(u)
        gender_tmp.extend(p)
        country_tmp.extend(q)
        center_tmp.extend(r)
        birth_tmp.extend(s)
        aidsmode_tmp.extend(t)

        records_generated += batch_size  # Update the total records generated so far

    
    
    np.save(path + 'code', code_tmp[:total_records_to_generate])
    np.save(path + 'interval', interval_tmp[:total_records_to_generate])
    np.save(path + 'length', length_tmp[:total_records_to_generate])
    np.save(path + 'age', age_tmp[:total_records_to_generate])
    np.save(path + 'gender', gender_tmp[:total_records_to_generate])
    np.save(path + 'country', country_tmp[:total_records_to_generate])
    np.save(path + 'center', center_tmp[:total_records_to_generate])
    np.save(path + 'birth', birth_tmp[:total_records_to_generate])
    np.save(path + 'mode', aidsmode_tmp[:total_records_to_generate])
    print("FINISHED")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',"-g", type=str, default="0")
    parser.add_argument('--serial',"-S", type=str)
    parser.add_argument("--b_ckpt","-B",type=str)
    parser.add_argument("--c_ckpt","-C",type=str)
    parser.add_argument("--e_ckpt","-E",type=str)
    parser.add_argument("-k",type=int,default=1)
    parser.add_argument("-n",type=int,default=40000)
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
    for a in range(args.k):
        print("Generating %dth/%d batch, with B-%s, C-%s, E-%s" % (a,args.k,args.b_ckpt,args.c_ckpt,args.e_ckpt), flush=True)
        gen_dataset(a)
