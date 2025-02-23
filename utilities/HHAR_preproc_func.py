import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os

def sequence_extraction(df, timestamp_delta_threshold=0.5, offset= 1e9, discard_threshold=1):
    """
    Extracting sequence within threshold parameters
    -offset: unixtimestamp to real timesteps
    -timestamp_delta_threshold: threshold for sequence cutoff point
    -discard_threshold: discarding sequence below a certain threshold
    """
    start= df['Creation_Time'].iloc[0] #0 # timedelta(seconds=0)
    length=1

    lengths= []
    starts= [0]
    device_user_gt=[f"{df['Device'].iloc[0]}-{df['User'].iloc[0]}-{df['gt_label'].iloc[0]}"]

    # extracting continuous sequences from device: start, length, identifier
    for index, row in tqdm(df[1:].iterrows(), total= len(df[1:])):
        if abs(start-row['Creation_Time'])/offset< timestamp_delta_threshold and device_user_gt[-1]==f"{row['Device']}-{row['User']}-{row['gt_label']}":
            length+=1
            start=row['Creation_Time']
        else:
            lengths.append(length)
            length=1
            start=row['Creation_Time']
            starts.append(index)
            device_user_gt.append(f"{row['Device']}-{row['User']}-{row['gt_label']}")

    if length > 1:
        lengths.append(length)


    # Discarding sequences below a certain threshold length
    dictionary={}
    for key in np.unique(device_user_gt[:]):
        dictionary[key]={}

    for i, j, k in zip(starts, lengths[:], device_user_gt[:]):
        if j>discard_threshold:
            dictionary[k][i]= j
            tmp= "OK"
        else:
            tmp= "DISCARD"
        # print(k, i, j, tmp)

    return dictionary

def find_average_length(df, dictionary, threshold=2, offset=1e9):
# finding indices of approximate duration
# purpose 1: group such indices together for further processing later
# purpose 2: try to find the average length of indices for approximate duration -- main purpose

    lengths_for_window= []

    windows_dict={}
    for key, start_length_dict in dictionary.items():
        print(key, end="")
        windows_dict[key]=[]
        for start, length in start_length_dict.items():
    #         print(f"length of sequence {length}")
            seq_start=start
            duration=0
    #         count=1 ###
            window_count=1
            last_window_index=start
            window_list=[]

            for count in range(1, length):
    #         while count < length: # while sequence is not exhausted ###
                # duration is the start+count point - seq_start point
                duration = abs(df['Creation_Time'].iloc[start+count]-df['Creation_Time'].iloc[seq_start])/offset

    #             print(duration)

                if duration > threshold:
    #                 print(f"duration {duration}")
                    duration=0
                    lengths_for_window.append(window_count)
    #                 print(f"window length: {count-last_window_index}")
                    window_list.append((last_window_index, last_window_index+count))
    #                 print(f"window {window_list[-1]}")
                    last_window_index += count+1 # step window forward (noninclusive with last window)
                    seq_start= start+count+1 # step seq_start point forward (noninclusive with last window)
                    window_count=1
                else:
                    window_count += 1

    #             count += 1 # DON'T CHANGE ###

    #         if len(window_list)>0:
    #         print(window_list)
    #         print(lengths_for_window)
    #         print(np.mean(lengths_for_window))
            windows_dict[key].append(window_list) # could have empty lists
    #     break

        print(" Done")

    # windows_dict: dictionary containing tuples(window_index_start, window_index_end)
    return int(round(np.mean(lengths_for_window))), windows_dict

def process_sensor_for_extraction(sensor:int, scaling=True):
    """
    Reading csv, creating labels, scaling and extracting sequence
    """
    assert isinstance(sensor, int), "sensor parameter should be int"
    assert 1<= sensor and sensor <= 4, "sensor parameter should be 'accel' or 'gyro'"

    if sensor==1:
        df= pd.read_csv('./HHAR/Activity recognition exp/Phones_accelerometer.csv')
    elif sensor==2:
        df= pd.read_csv('./HHAR/Activity recognition exp/Phones_gyroscope.csv')
    elif sensor==3:
        df= pd.read_csv('./HHAR/Activity recognition exp/Watch_accelerometer.csv')
    elif sensor==4:
        df= pd.read_csv('./HHAR/Activity recognition exp/Watch_gyroscope.csv')

    df['Arrival_Time_datetime']= pd.to_datetime(df['Arrival_Time'])
    df['Creation_Time_datetime']= pd.to_datetime(df['Creation_Time'])

    # creating gt_label: gt in numeric form
    temp= np.array( [ -1 for i in range(len(df)) ] )
    gt= ["bike", "sit", "stand", "walk", "stairsup", "stairsdown"]
    for label_name, label_num in zip(gt, range(len(gt))):
        for i in np.nonzero((df['gt']==label_name).to_numpy())[0]:
            temp[i]= label_num

    for idx in np.nonzero((temp==-1))[0]:
        temp[idx]= label_num +1

    df['gt_label']= temp

    # ridding NaN gts
    df= df.drop(df[df['gt_label']==len(gt)].index, axis=0)
    df.index= [i for i in range(len(df))]

    if scaling:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        scaler= MinMaxScaler()

        # Scaling based on device
        for device in np.unique(df['Device']):
            df.loc[(df["Device"]==device), ["x", "y", "z"]]= scaler.fit_transform( df[(df["Device"]==device) ][["x", "y", "z"]] )


        # scaling based on device and gt_label
        # for device in np.unique(df['Device']):
        #     for label in np.unique(df['gt_label']):
        #         df.loc[(df["Device"]==device) & (df["gt_label"]==label), ["x", "y", "z"]]= scaler.fit_transform( df[(df["Device"]==device) & (df["gt_label"]==label) ][["x", "y", "z"]] )

    seq_dict= sequence_extraction(df)
    # seq_dict: dictionary of sequence (seq_dict[device-user-label][ts_start_index]= length of sequence)
    average_win_len, windows_dict= find_average_length(df, seq_dict)

    # purpose 2 continued: Creating windows from fused sequences
    window_size=average_win_len
    data_indices_dict={}
    for key, start_length_dict in seq_dict.items():
        data_indices_dict[key]= []

        for start, length in start_length_dict.items():
            tmp_list=[]
            for i in range(start, start+length-window_size, window_size//2):
                tmp_list.append(np.arange(i, i+window_size))
            if len(tmp_list)>0:
                data_indices_dict[key].append(tmp_list)

    # seq_dict: dictionary of sequence (seq_dict[device-user-label][ts_start_index]= length of sequence)
    # windows_dict: dictionary containing ( windows_dict[device-user-label]=(window_index_start, window_index_end) )
    return data_indices_dict, windows_dict, seq_dict, df

def process_csv(sensor:int, scaling=True):
    """
    Reading csv, creating labels, scaling and extracting sequence
    """
    assert isinstance(sensor, int), "sensor parameter should be int"
    assert 1<= sensor and sensor <= 4, "sensor parameter should be 'accel' or 'gyro'"

    if sensor==1:
        df= pd.read_csv('./HHAR/Activity recognition exp/Phones_accelerometer.csv')
    elif sensor==2:
        df= pd.read_csv('./HHAR/Activity recognition exp/Phones_gyroscope.csv')
    elif sensor==3:
        df= pd.read_csv('./HHAR/Activity recognition exp/Watch_accelerometer.csv')
    elif sensor==4:
        df= pd.read_csv('./HHAR/Activity recognition exp/Watch_gyroscope.csv')

    df['Arrival_Time_datetime']= pd.to_datetime(df['Arrival_Time'])
    df['Creation_Time_datetime']= pd.to_datetime(df['Creation_Time'])

    # creating gt_label: gt in numeric form
    temp= np.array( [ -1 for i in range(len(df)) ] )
    gt= ["bike", "sit", "stand", "walk", "stairsup", "stairsdown"]
    for label_name, label_num in zip(gt, range(len(gt))):
        for i in np.nonzero((df['gt']==label_name).to_numpy())[0]:
            temp[i]= label_num

    for idx in np.nonzero((temp==-1))[0]:
        temp[idx]= label_num +1

    df['gt_label']= temp

    # ridding NaN gts
    df= df.drop(df[df['gt_label']==len(gt)].index, axis=0)
    df.index= [i for i in range(len(df))]

    if scaling:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        scaler= MinMaxScaler()

        # Scaling based on device
        for device in np.unique(df['Device']):
            df.loc[(df["Device"]==device), ["x", "y", "z"]]= scaler.fit_transform( df[(df["Device"]==device) ][["x", "y", "z"]] )

    return df

def process_sensor(sensor:int, scaling=True):
    """
    Reading csv, creating labels, scaling and extracting sequence
    """
    assert isinstance(sensor, int), "sensor parameter should be int"
    assert 1<= sensor and sensor <= 4, "sensor parameter should be 'accel' or 'gyro'"

    if sensor==1:
        df= pd.read_csv('./HHAR/Activity recognition exp/Phones_accelerometer.csv')
    elif sensor==2:
        df= pd.read_csv('./HHAR/Activity recognition exp/Phones_gyroscope.csv')
    elif sensor==3:
        df= pd.read_csv('./HHAR/Activity recognition exp/Watch_accelerometer.csv')
    elif sensor==4:
        df= pd.read_csv('./HHAR/Activity recognition exp/Watch_gyroscope.csv')

    df['Arrival_Time_datetime']= pd.to_datetime(df['Arrival_Time'])
    df['Creation_Time_datetime']= pd.to_datetime(df['Creation_Time'])

    # creating gt_label: gt in numeric form
    temp= np.array( [ -1 for i in range(len(df)) ] )
    gt= ["bike", "sit", "stand", "walk", "stairsup", "stairsdown"]
    for label_name, label_num in zip(gt, range(len(gt))):
        for i in np.nonzero((df['gt']==label_name).to_numpy())[0]:
            temp[i]= label_num

    for idx in np.nonzero((temp==-1))[0]:
        temp[idx]= label_num +1

    df['gt_label']= temp

    # ridding NaN gts
    df= df.drop(df[df['gt_label']==len(gt)].index, axis=0)
    df.index= [i for i in range(len(df))]

    if scaling:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        scaler= MinMaxScaler()

        # Scaling based on device
        for device in np.unique(df['Device']):
            df.loc[(df["Device"]==device), ["x", "y", "z"]]= scaler.fit_transform( df[(df["Device"]==device) ][["x", "y", "z"]] )


        # scaling based on device and gt_label
        # for device in np.unique(df['Device']):
        #     for label in np.unique(df['gt_label']):
        #         df.loc[(df["Device"]==device) & (df["gt_label"]==label), ["x", "y", "z"]]= scaler.fit_transform( df[(df["Device"]==device) & (df["gt_label"]==label) ][["x", "y", "z"]] )

    seq_dict= sequence_extraction(df)
    # seq_dict: dictionary of sequence (seq_dict[device-user-label][ts_start_index]= length of sequence)
    return seq_dict, df

def process_csv_for_fusion(sensor:str, scaling=True):
    assert isinstance(sensor, str), f"sensor must be str. Found {sensor}"
    assert sensor=="Phone" or sensor=="Watch", f"sensor must be Phone or Watch. Found {sensor}"

    if sensor=="Phone":
        accel_df= process_csv(1, scaling=scaling)
        print("Accel DF DONE")
        gyro_df= process_csv(2, scaling=scaling)
        print("Gyro DF DONE")
    elif sensor=="Watch":
        accel_df= process_csv(3, scaling=scaling)
        print("Accel DF DONE")
        gyro_df= process_csv(4, scaling=scaling)
        print("Gyro DF DONE")

    return accel_df, gyro_df

def process_sensor_for_fusion(sensor:str, scaling=True, features=["x","y","z"]):
    """
    Reading csv, creating labels, scaling and extracting sequence
    """
    assert isinstance(sensor, str), f"sensor must be str. Found {sensor}"
    assert sensor=="Phone" or sensor=="Watch", f"sensor must be Phone or Watch. Found {sensor}"

    if sensor=="Phone":
        accel_seq_dict, accel_df= process_sensor(1, scaling=scaling)
        gyro_seq_dict, gyro_df= process_sensor(2, scaling=scaling)
    elif sensor=="Watch":
        accel_seq_dict, accel_df= process_sensor(3, scaling=scaling)
        gyro_seq_dict, gyro_df= process_sensor(4, scaling=scaling)

    fusion_dict= sensor_fusion(accel_seq_dict, accel_df, gyro_seq_dict, gyro_df)


    # finding indices of approximate duration
    # purpose 1: group such indices together for further processing later
    # purpose 2: try to find the average length of indices for approximate duration

    # default using he timestamp of accel to fuse
    threshold= 2
    offset= 1e9
    lengths_for_window= []

    windows_dict={}
    for key, sequences in fusion_dict.items():
        print(key)
        windows_dict[key]=[]
        for sequence in sequences:

            seq_start=sequence[0][0] # the accel timestamp from the first index of the sequence
            duration=0
            count=1
            last_window_index=0
            window_list=[]
            for (a_i, g_i) in sequence[1:]:
                duration += abs(accel_df['Creation_Time'].iloc[a_i]-accel_df['Creation_Time'].iloc[seq_start])/offset
                seq_start= a_i

                if duration > threshold:

                    duration=0
                    lengths_for_window.append(count)
                    window_list.append((last_window_index, last_window_index+count))
                    last_window_index=last_window_index+count
    #                 print(f"count {count}")
                    count=1
                else:
                    count += 1

    #         if len(window_list)>0:
            windows_dict[key].append(window_list) # could have empty lists
    #         print(f"window count {len(window_list)}")
    #         print(f"length_for_window {lengths_for_window}")
    #     break
    #         print()
    #     print()





    # purpose 2 continued: Creating windows from fused sequences
    window_size=int(round(np.mean(lengths_for_window)))
    data_indices_dict={}
    for key, sequences in fusion_dict.items():
        data_indices_dict[key]= []

        for sequence in sequences:
            tmp_list=[]
            for i in range(0, len(sequence)-window_size, window_size//2):
                tmp_list.append(sequence[i:i+window_size])
            if len(tmp_list)>0:
                data_indices_dict[key].append(tmp_list)



    return data_indices_dict, accel_df, gyro_df

def extract_original_measurements(data_indices_dict:dict, df:pd.DataFrame, features=["x", "y", "z"]):
    # original measurements
    data_dict={}
    for key, idx_lists in data_indices_dict.items():
        data_dict[key]=[]
        # print(key)
        for idx_list in tqdm(idx_lists, total=len(idx_lists)): # idx_list is a sequence
            tmp_list=[]
            for indices in idx_list:
                tmp_list.append(df.iloc[indices][features].to_numpy())
        data_dict[key].append(tmp_list)
    return data_dict

def extract_ecdf_features(data_indices_dict:dict,df:pd.DataFrame, features=["x", "y", "z"]):
    # ecdf features
    from statsmodels.distributions.empirical_distribution import ECDF
    from scipy.interpolate import interp1d

    ecdf_d_parameter= 30
    step= 1 / ecdf_d_parameter
    bins= list(map(lambda x: x*step, list(range(1,ecdf_d_parameter+1))))

    data_dict={}
    for key, idx_lists in data_indices_dict.items():
            data_dict[key]=[]
            # print(key)
            for idx_list in tqdm(idx_lists, total=len(idx_lists)): # eacj idx_list is a sequence
                tmp_list=[]
                for indices in idx_list:
                    original_features= df.iloc[indices][features].to_numpy()

                    ecdf_features=[]
                    for channel in range(3):
                        feature_set= list(set(original_features[:,channel]))
                        ecdf= ECDF(feature_set)
                        feature_quantile= ecdf(feature_set)
                        try:
                            inverse_ecdf= interp1d(feature_quantile, feature_set, kind='cubic', fill_value="extrapolate")
                        except ValueError:
                            inverse_ecdf= interp1d(feature_quantile, feature_set, fill_value="extrapolate")

                        feature_set_mean= np.mean(original_features[:, channel])

                        ecdf_features.append(list(inverse_ecdf(bins))+[feature_set_mean])
                    ecdf_features= np.stack(ecdf_features, axis=1)
                    tmp_list.append(ecdf_features)
                data_dict[key].append(tmp_list)
    return data_dict
    #         break

def extract_time_features(data_indices_dict:dict, df:pd.DataFrame, features=["x", "y", "z"]):
    # Time Domain
    # pairs= [(0,1), (0,2), (1,2)] #solo device
    # pairs= [(0,1), (0,2), (1,2)]
    pairs=[]
    channels= len(features)
    for i in range(channels-1):
        for j in range(i+1,channels):
            pairs += [(i,j)]

    data_dict={}
    for key, idx_lists in data_indices_dict.items():
            data_dict[key]=[]
            # print(key)
            for idx_list in tqdm(idx_lists, total=len(idx_lists)): # each idx_list is a sequence
                tmp_list=[]
                for indices in idx_list:

                    original_features= df.iloc[indices][features].to_numpy()

                    mean= np.mean(original_features, axis=0)
                    std= np.std(original_features, axis=0)
                    var= np.var(original_features, axis=0)

                    maximum= np.max(original_features, axis=0)
                    minimum= np.min(original_features, axis=0)
                    median= np.median(original_features, axis=0)
                    stat_range= maximum-minimum

                    rms= np.sqrt(np.sum(original_features ** 2, axis=0) / len(original_features))

                    integrated_rms= np.trapz(np.sqrt( original_features ** 2 / len(original_features) ), axis=0)

                    corrcoef= list(map(lambda x: np.corrcoef(original_features.T)[x[0], x[1]], pairs))

                    cross_cor= list(map(lambda x: np.dot(original_features[:,x[0]], original_features[:, x[1]]) / len(original_features), pairs))

    #                 print(np.stack([mean,std,var,maximum,minimum,median,stat_range,rms,integrated_rms, corrcoef, cross_cor]).shape)
                    tmp_list.append(np.stack([mean,std,var,maximum,minimum,median,stat_range,rms,integrated_rms, corrcoef, cross_cor]))
                data_dict[key].append(tmp_list)
    return data_dict

def extract_frequency_features(data_indices_dict:dict, df:pd.DataFrame, features=["x", "y", "z"]):
# Frequencey Domain
    from scipy.fft import fft, fftfreq, fftshift
    from scipy.stats import entropy
    # negative value causes -inf

    number_of_bins= 41
    fft_freq_bins= np.array(list(range(number_of_bins)))
    fft_freq_bins = fft_freq_bins / 2

    data_dict={}
    for key, idx_lists in data_indices_dict.items():
            data_dict[key]=[]
            # print(key)
            for idx_list in tqdm(idx_lists, total=len(idx_lists)): # each idx_list is a different sequence
                tmp_list=[]
                for indices in idx_list:
                    original_features= df.iloc[indices][features].to_numpy()

                    fft_features= fft(original_features)
                    positive_spectrum= len(fft_features) // 2
                    normalised_fft_features= fft_features[:positive_spectrum]/ positive_spectrum

                    # 0 to 20Hz bins --> 0.5 hz bin width --> 40 features, 41 features including dc_component
                    # fftfreq( len(fft_features), d=2/len(fft_features) )
                    selected_normalised_fft_features= normalised_fft_features[:number_of_bins]
    #                 fft_dc_component= fft_features[0]
                    selected_normalised_fft_magnitude= abs(selected_normalised_fft_features) # don't need this inside

                    selected_normalised_fft_sum= np.sum(selected_normalised_fft_features, axis=0)
                    selected_normalised_fft_entropy= entropy(selected_normalised_fft_magnitude[1:], axis=0)

                    dominant_magnitudes=[]
                    dominant_frequencies=[]
                    for i in range(len(features)):
                        index= list(selected_normalised_fft_features[:,i]).index(max(selected_normalised_fft_features[:,i]))
                        dominant_magnitudes.append(abs(selected_normalised_fft_features[index,i]))
                        dominant_frequencies.append(fft_freq_bins[index])

    #       partial   print(np.concatenate([selected_normalised_fft_sum, selected_normalised_fft_entropy, dominant_magnitudes, dominant_frequencies]).shape)
    #       full      print(np.concatenate([selected_normalised_fft_features.reshape(-1), selected_normalised_fft_sum, selected_normalised_fft_entropy, dominant_magnitudes, dominant_frequencies]).shape)

                    tmp_list.append(np.concatenate([selected_normalised_fft_sum, selected_normalised_fft_entropy, dominant_magnitudes, dominant_frequencies]))
                data_dict[key].append(tmp_list)
    return data_dict
#         break
def sensor_fusion(accel_dict, accel_df, gyro_dict, gyro_df):
    """
    fusion between accel df and gyro df
    """

    # extract keys present in both gyro and accel -- don't always have the same keys
    keys_in_both=[]
    tmp_list= list(gyro_dict.keys())
    for key in np.unique(list(accel_dict.keys())):
        try:
            tmp_list.index(key)
            if (len(gyro_dict[key]) > 0) & (len(accel_dict[key]) > 0):
                keys_in_both.append(key)
        except ValueError:
            continue

    # fusion of accel and gyro indices
    fusion_dict={}
    fusion_time_delta_threshold= 0.5
    offset= 1e9

    for key in keys_in_both:
        print(f"{key}...")
        fusion_dict[key]=[]
        accel_dict_list= list(accel_dict[key].items()) # device-user-label : start : length
        gyro_dict_list= list(gyro_dict[key].items()) # device-user-label : start : length

        accel_total_instances= sum([i[1] for i in accel_dict_list])
        gyro_total_instances= sum([i[1] for i in gyro_dict_list])

        accel_total_count=0
        gyro_total_count=0

        accel_list_count=0
        gyro_list_count=0

        accel_start, accel_length= accel_dict_list[accel_list_count]
        gyro_start, gyro_length= gyro_dict_list[gyro_list_count]

        gyro_count=0
        accel_count=0

        tmp_list=[]

        gyro_flag=0
        accel_flag=0

        while (gyro_total_count<gyro_total_instances) | (accel_total_count<accel_total_instances):

            # if duration is within fusion_time_delta_threshold
            if abs(gyro_df['Creation_Time'].iloc[gyro_start+gyro_count]-accel_df['Creation_Time'].iloc[accel_start+accel_count])/offset <= fusion_time_delta_threshold:
                tmp_list.append((accel_start+accel_count, gyro_start+gyro_count))

                accel_count += 1
                gyro_count += 1

                accel_total_count += 1
                gyro_total_count += 1

            # if duration is not within fusion_time_delta_threshold, but both sensors are still on the same corresponding sequence
            elif accel_list_count==gyro_list_count:
    #             print("Exhausting sequence: ", end="")

                # exhaust current sequence first
                if gyro_df['Creation_Time'].iloc[gyro_start+gyro_count] > accel_df['Creation_Time'].iloc[accel_start+accel_count]:
                    accel_count += 1
                    accel_total_count += 1
    #                 print("Accel", accel_count, accel_length, accel_total_count, accel_total_instances)

                else:
                    gyro_count += 1
                    gyro_total_count += 1
    #                 print("Gyro", gyro_count, gyro_length, gyro_total_count, gyro_total_instances)

            # if duration is not within fusion_time_delta_threshold, and sensors are not on the same corresponding sequence
            else:

    #             print("Difference in sequence: ",end="")
                if accel_list_count > gyro_list_count:
                    gyro_count += 1
                    gyro_total_count += 1
    #                 print("Gyro catching up", gyro_count, gyro_length, gyro_total_count, gyro_total_instances)
                else:
                    accel_count += 1
                    accel_total_count += 1
    #                 print("Accel catching up", accel_count, accel_length, accel_total_count, accel_total_instances)

            # if sequence for gyro has been exhausted
            if gyro_count==gyro_length:
                gyro_list_count+=1
                gyro_flag=1

                # if there are still sequences remaining
                if gyro_list_count<len(gyro_dict_list):
                    gyro_count=0
                    gyro_start, gyro_length= gyro_dict_list[gyro_list_count]
                else:
                # if there are no more sequences remaining
    #                 print("gyro overflow")
                    gyro_list_count=len(accel_dict_list)+1


            # if sequence for accel has been exhausted
            if accel_count==accel_length:
                accel_list_count+=1
                accel_flag=1

                # if there are still sequences remaining
                if accel_list_count<len(accel_dict_list):
                    accel_count=0
                    accel_start, accel_length= accel_dict_list[accel_list_count]
                else:
                # if there are no more sequences remaining
    #                 print("accel overflow")
                    accel_list_count=len(gyro_dict_list)+1

            # when both sequences have been exhausted and fused as best as possible
            if accel_flag & gyro_flag:
                accel_flag=0
                gyro_flag=0
                if len(tmp_list)>0:
                    fusion_dict[key].append(tmp_list)
                    tmp_list=[]

    return fusion_dict

def stratified_10_fold_split(data_dict:dict, save_name:str, save=False):
    # 10 fold stratified CV
    # 1 out of 10 sequences used for testing
    stratified_10_fold={}
    for i in range(10):
        stratified_10_fold[i]=[]

    count=0
    num_list=list(range(10))
    np.random.shuffle(num_list)
    for key, sequences in data_dict.items():
        print(key)
        for sequence in tqdm(sequences): #
            for window in sequence:
                stratified_10_fold[num_list[count]].append((window,int(key[-1])))
                if count==9:# window of all kinds everywhere
                    count=0
                else:
                    count += 1
    #         if count==9:# each fold only has windows from their respective sequence
    #             count=0
    #         else:
    #             count += 1
    if save:
        pickle.dump(stratified_10_fold, open(save_name, "wb"))
    return stratified_10_fold

def user_fold_split(data_dict, save_path:str, save=False):
    user_fold={}
    for key in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']:
        user_fold[key]=[]

    for key, sequences in data_dict.items():
        print(key)

        user= key.find("-")+1
        user= key[user]

        for sequence in tqdm(sequences): #
            for window in sequence:
                user_fold[user].append((window,int(key[-1])))
    #     break

    if save:
        pickle.dump(user_fold, open(save_path, "wb"))

    return user_fold

def model_fold_split(data_dict:dict, save_path:str, save=False):
# leave 1 phone model out
# np.unique(phone_accel_df['Model']) # ['nexus4', 's3', 's3mini', 'samsungold']

    model_fold={}
    for key in ['nexus4', 's3', 's3mini', 'samsungold']:
        model_fold[key]=[]

    for key, sequences in data_dict.items():
        print(key)

        index= key.find("-")

        model=key[:index-2]

        for sequence in tqdm(sequences): #
            for window in sequence:
                model_fold[model].append((window,int(key[-1])))
    if save:
        pickle.dump(model_fold, open(save_path,"wb"))

    return model_fold

def model_user_fold_split(data_dict:dict, save_path:str, save=False):
# leave 1 phone model, 1 user out

    model_user_fold={}
    for model in ['nexus4', 's3', 's3mini', 'samsungold']:
        model_user_fold[model]={}
        for user in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']:
            model_user_fold[model][user]=[]


    for key, sequences in data_dict.items():
        # print(key)

        index= key.find("-")
        model=key[:index-2]
        user= key[index+1]

        for sequence in tqdm(sequences): #
            for window in sequence:
                model_user_fold[model][user].append((window,int(key[-1])))

    if save:
        pickle.dump(model_user_fold, open(save_path, "wb"))

    return model_user_fold

def intra_model_fold_split(data_dict:dict, save_path:str, save=False):
    # intra model leave one device out
    # train with N-1 devices of one model , test on 1 device of model

    model_instance_fold={}
    for model in ['nexus4', 's3', 's3mini', 'samsungold']:
        model_instance_fold[model]={}
        for instance in range(1,3):
            model_instance_fold[model][str(instance)]=[]

    for key, sequences in data_dict.items():
        # print(key)
        index= key.find("-")

        model=key[:index-2]
        instance= key[index-1]

        for sequence in tqdm(sequences): #
            for window in sequence:
                model_instance_fold[model][instance].append((window,int(key[-1])))
    if save:
        pickle.dump(model_instance_fold, open(save_path,"wb"))
        # break
    return model_instance_fold
