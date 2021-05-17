#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import pickle
import numpy as np
import time
import datetime
import random
import os
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
# logging.basicConfig(level=logging.INFO, filename='log_Model', format=LOG_FORMAT)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

from tools.pos_encoder import *
from tools.time_encoder import *

VAL_POINT = 65623

# # 读取数据

# In[23]:


import multiprocessing
from multiprocessing import Manager, Process, Lock
import gc


def load_data(start, end):
    pickle_path = './3_part_train_mp/keylist_%d_%d.pickle' % (start, end)
    with open(pickle_path, 'rb') as f:
        key_list = pickle.load(f)

    pickle_path = './3_part_train_mp/mp_info_dict_%d_%d.pickle' % (start, end)
    with open(pickle_path, 'rb') as f:
        mp_info_dict = pickle.load(f)

    pickle_path = './3_part_train_mp/data_info_dict_%d_%d.pickle' % (start, end)
    with open(pickle_path, 'rb') as f:
        data_info_dict = pickle.load(f)

    return key_list, mp_info_dict, data_info_dict


# # 建立静态特征的功能函数

# # 建立交互特征的功能函数

# # 建立聚合特征的函数

# In[24]:


def get_agg_action_feature_dict(tracking_ids, action_types, cur_action, last_time, df_a_action, df_a_order,
                                df_a_distance, prefix):
    cur_tracking_id = cur_action.tracking_id
    cur_action_type = cur_action.action_type

    agg_feature_list = []
    for i in range(len(tracking_ids)):
        agg_feature_dict = {}
        df_a_distance_relation = df_a_distance[cur_tracking_id][tracking_ids[i]]
        next_action_type = action_types[i]
        query_row = df_a_distance_relation.query('source_type == @cur_action_type & target_type == @next_action_type')
        if query_row.shape[0] == 0:
            continue
        else:
            se_query_row = query_row.iloc[0]
        pos_dict = se_query_row[['source_lng', 'source_lat', 'target_lng', 'target_lat']].to_dict()
        pos_mutual_dict = mutual2pos(pos_dict['source_lat'], pos_dict['source_lng'], pos_dict['target_lat'],
                                     pos_dict['target_lng'], prefix)
        agg_feature_dict.update(pos_mutual_dict)
        agg_feature_dict[prefix + '_grid_distance'] = se_query_row['grid_distance']
        agg_feature_dict[prefix + '_create_time_sub_last'] = last_time - df_a_order.loc[tracking_ids[i]].create_time
        agg_feature_dict[prefix + '_confirm_time_sub_last'] = last_time - df_a_order.loc[tracking_ids[i]].confirm_time
        agg_feature_dict[prefix + '_assigned_time_sub_last'] = last_time - df_a_order.loc[tracking_ids[i]].assigned_time
        agg_feature_dict[prefix + '_promise_deliver_time_sub_last'] = df_a_order.loc[tracking_ids[
            i]].promise_deliver_time - last_time
        agg_feature_dict[prefix + '_estimate_pick_time_sub_last'] = df_a_order.loc[
                                                                        tracking_ids[i]].estimate_pick_time - last_time

        agg_feature_list.append(agg_feature_dict)

    df_agg_feature = pd.DataFrame(agg_feature_list)

    feature_dict = {}
    # 跟所有action[tracking_ids]的平均值、最大值、最小值
    se_mean = df_agg_feature.mean()
    se_mean.index = se_mean.index + '_mean'
    feature_dict.update(se_mean.to_dict())

    se_min = df_agg_feature.min()
    se_min.index = se_min.index + '_min'
    feature_dict.update(se_min.to_dict())

    se_max = df_agg_feature.max()
    se_max.index = se_max.index + '_max'
    feature_dict.update(se_max.to_dict())


    return feature_dict


# # 建立特征主函数

# In[25]:


# key_list, mp_info_dict, data_info_dict = load_data(0, 76)
# mp_action, mp_distance, mp_order, mp_couriers = mp_info_dict['action'], mp_info_dict['distance'], mp_info_dict['order'], mp_info_dict['couriers']
# know_lens_list, full_lens, impossible_idxs_list = data_info_dict['know_lens_list'], data_info_dict['full_lens'], data_info_dict['impossible_idxs_list']
# for i in range(1):
#     date, courier, wave_idx = key_list[i]
#     df_a_action = mp_action[date][courier][wave_idx]
#     df_a_action['action_type_ASSIGN']=0
#     df_a_distance = mp_distance[date][courier][wave_idx]
#     df_a_order = mp_order[date][courier][wave_idx]
# df_a_distance_relation = df_a_distance[2100074542836254538][2100074542836254538]
# df_a_distance_relation = df_a_distance_relation.sort_values(by=['source_type', 'target_type'])
# show_df(df_a_distance_relation)


# In[26]:


courier_delay_features = ['pickup_delay_rate', 'delivery_delay_rate', 'pickup_delay_time_avg',
                          'delivery_delay_time_avg',
                          'delivery_delay_count', 'pickup_delay_count']
mp_weather = {'正常天气': 0, '轻微恶劣天气': 1, '恶劣天气': 2, '极恶劣天气': 3}


def generate_gnn_df_train_part_data_feature(i, j, df_a_action, cur_action, last_action, pick_delay_num,
                                            send_delay_num, tracking_id_status_dict, know_len, full_len, se_a_couier,
                                            df_a_order, df_a_distance,action_n):
    features_dict = {}
    features_dict['origin_i'] = i
    features_dict['target_position'] = action_n  # action n
    # features_dict['assigened_num']=assigened_num features_dict['picked_num']=picked_num
    #需要修改一下对未知时间步的办法，不然默认已知顺序

    # pick_delay_num send_delay_num

    df_a_order.index = df_a_order.tracking_id
    action_row = cur_action

    if j < know_len:
        cur_time = action_row.expect_time
        assigen_num = df_a_action.iloc[: j + 1].query('action_type=="ASSIGN"').shape[0]
        pick_num = df_a_action.iloc[: j + 1].query('action_type=="PICKUP"').shape[0]
        delivery_num = df_a_action.iloc[: j + 1].query('action_type=="DELIVERY"').shape[0]
        assigened_num = assigen_num - pick_num
        picked_num = pick_num - delivery_num
    else:
        assigen_num = df_a_action.iloc[: know_len].query('action_type=="ASSIGN"').shape[0]
        pick_num = df_a_action.iloc[: know_len].query('action_type=="PICKUP"').shape[0]
        delivery_num = df_a_action.iloc[: know_len].query('action_type=="DELIVERY"').shape[0]
        if cur_action.action_type=="ASSIGN":
            assigen_num+=1
        elif cur_action.action_type=="PICKUP":
            pick_num+=1
        elif cur_action.action_type == "DELIVERY":
            delivery_num+=1
        assigened_num = assigen_num - pick_num
        picked_num = pick_num - delivery_num
        cur_time = last_action.expect_time

    features_dict['assigened_num'] = assigened_num
    features_dict['picked_num'] = picked_num
    tracking_id = action_row.tracking_id

    df_a_order.index = df_a_order.tracking_id
    # 聚合特征
    all_features = get_agg_action_feature_dict(list(df_a_action.tracking_id), list(df_a_action.action_type), action_row,
                                               cur_time, df_a_action, df_a_order, df_a_distance, 'all_agg')

    features_dict.update(all_features)
    df_a_order = df_a_order.reset_index(drop=True)
    df_order_info = df_a_order.query('tracking_id == @tracking_id')

    confirm_time = df_order_info.confirm_time.iloc[0]
    estimate_pick_time = df_order_info.estimate_pick_time.iloc[0]
    promise_deliver_time = df_order_info.promise_deliver_time.iloc[0]
    # 单一特征

    features_dict['expect_time'] = action_row.expect_time
    features_dict['weather_grade'] = mp_weather[df_order_info.weather_grade.iloc[0]]
    features_dict['confirm_time'] = confirm_time
    features_dict['estimate_pick_time'] = estimate_pick_time
    features_dict['promise_deliver_time'] = promise_deliver_time
    features_dict.update(se_a_couier[['level', 'speed', 'max_load']])

    if action_row.action_type == 'PICKUP':
        if cur_time > estimate_pick_time:
            pick_delay_num += 1
        if j < know_len:
            tracking_id_status_dict[tracking_id] = 1
    elif action_row.action_type == 'DELIVERY':
        if cur_time > promise_deliver_time:
            send_delay_num += 1
        if j < know_len:
            tracking_id_status_dict[tracking_id] = 2
    elif action_row.action_type == 'ASSIGN':
        if j < know_len:
            tracking_id_status_dict[tracking_id] = 0

    features_dict['pick_delay_num'] = pick_delay_num
    features_dict['send_delay_num'] = send_delay_num

    # make self and next
    df_a_distance_self = df_a_distance[tracking_id][tracking_id]
    df_a_distance_self = df_a_distance_self.loc[(df_a_distance_self.source_type == action_row.action_type)]
    features_dict['courier_wave_lng'] = df_a_distance_self.iloc[0]['source_lng']
    features_dict['courier_wave_lat'] = df_a_distance_self.iloc[0]['source_lat']
    features_dict['action_type'] = action_row.action_type
    df_a_distance_self = df_a_distance_self.sort_values(by=['source_type', 'target_type'])
    #         print('make self and next')
    #         show_df(df_a_distance_self)
    #         print(action_row.action_type)
    if action_row.action_type == 'ASSIGN':
        features_dict['self_assign_r'] = 1
        features_dict['self_pickup_r'] = df_a_distance_self.iloc[1]['grid_distance']
        features_dict['self_delivery_r'] = df_a_distance_self.iloc[0]['grid_distance']
    elif action_row.action_type == 'DELIVERY':
        features_dict['self_delivery_r'] = 1
        features_dict['self_assign_r'] = df_a_distance_self.iloc[0]['grid_distance']
        features_dict['self_pickup_r'] = df_a_distance_self.iloc[1]['grid_distance']
    elif action_row.action_type == 'PICKUP':
        features_dict['self_pickup_r'] = 1
        features_dict['self_delivery_r'] = df_a_distance_self.iloc[1]['grid_distance']
        features_dict['self_assign_r'] = df_a_distance_self.iloc[0]['grid_distance']
    if tracking_id_status_dict[tracking_id] == -1:
        features_dict['self_assign_p'] = cur_time - confirm_time
    else:
        features_dict['self_assign_p'] = 0
    if tracking_id_status_dict[tracking_id] == 0:
        features_dict['self_pickup_p'] = estimate_pick_time - cur_time
    else:
        features_dict['self_pickup_p'] = 0
    if tracking_id_status_dict[tracking_id] == 1:
        features_dict['self_delivery_p'] = promise_deliver_time - cur_time
    else:
        features_dict['self_delivery_p'] = 0
    return features_dict, tracking_id_status_dict, pick_delay_num, send_delay_num


def generate_rnn_df_train_part_data(key_list, mp_info_dict, data_info_dict, n_process):
    features_dict_list_process = Manager().list()
    lock = Lock()
    cnt = multiprocessing.Value("d", 0.0)

    def a_process(key_list, mp_info_dict, data_info_dict, start, end):

        mp_action, mp_distance, mp_order, mp_couriers = mp_info_dict['action'], mp_info_dict['distance'], mp_info_dict[
            'order'], mp_info_dict['couriers']
        know_lens_list, full_lens, impossible_idxs_list = data_info_dict['know_lens_list'], data_info_dict['full_lens'], \
                                                          data_info_dict['impossible_idxs_list']
        # features_dict_list = []
        for i in range(start, end):
            with lock:
                cnt.value += 1
                if cnt.value % 1000 == 0:
                    logging.info('finish %d samples' % cnt.value)
            # 每个波次的数据
            date, courier, wave_idx = key_list[i]
            df_a_action = mp_action[date][courier][wave_idx]
            df_a_distance = mp_distance[date][courier][wave_idx]
            df_a_order = mp_order[date][courier][wave_idx]
            se_a_couier = mp_couriers[courier][date]

            for list_index in range(len(know_lens_list[i])):
                # 通用的特征
                # 已知行为的时间
                know_len = know_lens_list[i][list_index]
                # 最后一个已知行
                # 未知action
                df_unknow_action = df_a_action.iloc[know_len: full_lens[i]]
                pick_delay_num = 0
                send_delay_num = 0
                tracking_id_status_dict = {}
                for row_raw in df_a_order.itertuples():
                    tracking_id_status_dict[row_raw.tracking_id] = -1
                # 对已知时间步进行特征构建
                action_n = 0
                for j in range(full_lens[i]):

                    # 已知时间步的行为
                    if j - know_len in impossible_idxs_list[i][know_len]:
                        continue
                    action_n += 1
                    cur_action = df_a_action.iloc[j]
                    last_action = df_a_action.iloc[know_lens_list[i][list_index] - 1]
                    full_len = full_lens[i]
                    features_dict, tracking_id_status_dict, pick_delay_num, send_delay_num = generate_gnn_df_train_part_data_feature(
                        i, j, df_a_action, cur_action,
                        last_action, pick_delay_num,
                        send_delay_num, tracking_id_status_dict, know_len,
                        full_len, se_a_couier,
                        df_a_order, df_a_distance,action_n)

                    features_dict['know_len'] = know_len
                    with lock:
                        features_dict_list_process.append(features_dict)

    len_key_list = len(key_list)
    process_list = []
    if (int(len_key_list / n_process) > 0):
        for i in range(0, len_key_list, int(len_key_list / n_process)):
            start, end = i, i + int(len_key_list / n_process)
            end = min(end, len_key_list)
            process = Process(target=a_process, args=(key_list, mp_info_dict, data_info_dict, start, end))
            process_list.append(process)
            process.start()
    else:
        process = Process(target=a_process, args=(key_list, mp_info_dict, data_info_dict, 0, len_key_list))
        process_list.append(process)
        process.start()

    for process in process_list:
        process.join()

    features_dict_list = []
    for data in features_dict_list_process:
        features_dict_list.append(data)
    df_features = pd.DataFrame(features_dict_list)

    df_target_position = df_features.target_position
    df_features = df_features.drop('target_position', axis=1)
    df_features.insert(0, 'target_position', df_target_position)

    df_know_len = df_features.know_len
    df_features = df_features.drop('know_len', axis=1)
    df_features.insert(0, 'know_len', df_know_len)

    df_origin_i = df_features.origin_i
    df_features = df_features.drop('origin_i', axis=1)
    df_features.insert(0, 'origin_i', df_origin_i)

    df_features.sort_values(by=['origin_i', 'know_len', 'target_position'])
    return df_features


def generate_gbdt_df_data_arugment_multiprocess():
    part = 32
    len_key_list = 82533
    s_list, e_list = [], []

    for i in range(0, len_key_list, int(len_key_list / part)):
        start, end = i, i + int(len_key_list / part)
        end = min(end, len_key_list)
        s_list.append(start)
        e_list.append(end)

    logging.info('start build part features')
    n_process = 32
    for (s, e) in zip(s_list, e_list):
        logging.info('start s:%d,fr e:%d' % (s, e))
        pickle_path = './4_generate_train_data_arugment_sample/df_feature_train_%d_%d.pickle' % (s, e)
        if os.path.exists(pickle_path):
            continue
        key_list, mp_info_dict, data_info_dict = load_data(s, e)
        logging.info('finish reading s:%d, e:%d' % (s, e))
        df_feature_train_part = generate_rnn_df_train_part_data(key_list, mp_info_dict, data_info_dict, n_process)
        logging.info('finish s:%d, e:%d' % (s, e))
        # write
        with open(pickle_path, 'wb') as f:
            pickle.dump(df_feature_train_part, f)

        del key_list, mp_info_dict, data_info_dict
        gc.collect()


# In[28]:


logging.info('start building df_train_features')
df_features_train = generate_gbdt_df_data_arugment_multiprocess()
logging.info('finish building df_train_features')
