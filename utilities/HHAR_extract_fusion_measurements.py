import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from typing import List
import os

# from utilities.HHAR_preproc_func import *
from HHAR_preproc_func import *
import argparse

if __name__=="__main__":
    parser= argparse.ArgumentParser(description='This script is used to process measurements from sensr, extract features and save data as a pickle.')
    parser.add_argument('-scaling', help='Scaling for data or not. Default: \"1\"', default="1")
    parser.add_argument("-device", help="Corresponding device and sensor's csv to be used. Options: phone or watch . Default: phone", default="phone")
    parser.add_argument("-features", help="Features to be extracted. Options: ecdf, time, frequency, original. Default: original", default="original")
    parser.add_argument("fold_setting", nargs="*", help="Fold setting for training and validation. Options: stratified10, userfold, modelfold. Default: stratified10", default=["stratified10"])
    args= parser.parse_args()
    args.device= str(args.device)
    args.scaling= bool(args.scaling)
    args.features=str(args.features)

    assert args.features=="original" or args.features=="ecdf" or args.features=="time" or args.features=="frequency", f"feature must be ecdf, time, frequency or orginal. Found {args.features} instead"
    assert isinstance(args.fold_setting, List), f"fold_setting must be a list. Found {args.fold_setting} instead."
    for i in args.fold_setting:
        assert i=="stratified10" or i=="userfold" or i=="modelfold", f"fold setting must be a list containing: stratified10, userfold or modelfold. Found {i} instead"

    print(f"FUSING SENSOR: {args.device}...", end="")
    save_path=f"./HHAR/Activity recognition exp/{args.device}/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(f"{save_path}fusion_indices_dict.pkl"):
        data_indices_dict, accel_df, gyro_df= process_sensor_for_fusion(args.device, args.scaling)
        pickle.dump(data_indices_dict, open(f"{save_path}fusion_indices_dict.pkl", "wb"))
    else:
        print("dictionary already exists...")
        accel_df, gyro_df = process_csv_for_fusion(args.device, args.scaling)
        data_indices_dict= pickle.load(open(f"{save_path}fusion_indices_dict.pkl", "rb"))
    print("DONE")

    if args.scaling:
        args.scaling="-minmax"
    else:
        args.scaling=""

    device_path= save_path
    if os.path.exists(f"{device_path}device-user-label_fusion_dict-{args.features}{args.scaling}.pkl"):
        data_dict= pickle.load(open(f"{device_path}device-user-label_fusion_dict-{args.features}{args.scaling}.pkl", "rb"))
    else:
        print(f"Extracting features: {args.features}...", end="")
        features=["x", "y", "z"]
        if args.features=="original":
            data_dict={}
            for key, idx_lists in data_indices_dict.items():
                    data_dict[key]=[]
                    print(key, end="")
                    for idx_list in idx_lists:
                        tmp_list=[]
                        for indices in idx_list:
                            accel_indices= [i[0] for i in indices]
                            gyro_indices= [i[1] for i in indices]
                            tmp_list.append(np.concatenate([accel_df.iloc[accel_indices][features].to_numpy(), gyro_df.iloc[gyro_indices][features].to_numpy()], axis=1))
                        data_dict[key].append(tmp_list)
                    print(" Done")
        elif args.features=="ecdf":
            from statsmodels.distributions.empirical_distribution import ECDF
            from scipy.interpolate import interp1d

            ecdf_d_parameter= 30
            step= 1 / ecdf_d_parameter
            bins= list(map(lambda x: x*step, list(range(1,ecdf_d_parameter+1))))

            data_dict={}
            for key, idx_lists in data_indices_dict.items():
                    data_dict[key]=[]
                    print(key, end="")
                    for idx_list in idx_lists:
                        tmp_list=[]
                        for indices in idx_list:
                            accel_indices= [i[0] for i in indices]
                            gyro_indices= [i[1] for i in indices]

                            accel_features= accel_df.iloc[accel_indices][features].to_numpy()

                            accel_ecdf_features=[]
                            for channel in range(len(features)):
                                feature_set= list(set(accel_features[:,channel]))
                                ecdf= ECDF(feature_set)
                                feature_quantile= ecdf(feature_set)
                                try:
                                    inverse_ecdf= interp1d(feature_quantile, feature_set, kind='cubic', fill_value="extrapolate")
                                except ValueError:
                                    inverse_ecdf= interp1d(feature_quantile, feature_set, fill_value="extrapolate")

                                feature_set_mean= np.mean(accel_features[:, channel])

                                accel_ecdf_features.append(list(inverse_ecdf(bins))+[feature_set_mean])
                            accel_ecdf_features= np.stack(accel_ecdf_features, axis=1)

                            gyro_features= gyro_df.iloc[gyro_indices][features].to_numpy()

                            gyro_ecdf_features=[]
                            for channel in range(len(features)):
                                feature_set= list(set(gyro_features[:,channel]))
                                ecdf= ECDF(feature_set)
                                feature_quantile= ecdf(feature_set)
                                try:
                                    inverse_ecdf= interp1d(feature_quantile, feature_set, kind='cubic', fill_value="extrapolate")
                                except ValueError:
                                    inverse_ecdf= interp1d(feature_quantile, feature_set, fill_value="extrapolate")

                                feature_set_mean= np.mean(gyro_features[:, channel])

                                gyro_ecdf_features.append(list(inverse_ecdf(bins))+[feature_set_mean])
                            gyro_ecdf_features= np.stack(gyro_ecdf_features, axis=1)

                            tmp_list.append(np.concatenate([accel_ecdf_features, gyro_ecdf_features], axis=1))

                        data_dict[key].append(tmp_list)
                    print(" Done")
        elif args.features=="time":
            channels= len(features)
            pairs=[]
            for i in range(channels-1):
                for j in range(i+1,channels):
                    pairs += [(i,j)]

            data_dict={}
            for key, idx_lists in data_indices_dict.items():
                    data_dict[key]=[]
                    print(key, end="")
                    for idx_list in idx_lists:
                        tmp_list=[]
                        for indices in idx_list:
                            accel_indices= [i[0] for i in indices]
                            gyro_indices= [i[1] for i in indices]

                            accel_features= accel_df.iloc[accel_indices][features].to_numpy()

                            accel_mean= np.mean(accel_features, axis=0)
                            accel_std= np.std(accel_features, axis=0)
                            accel_var= np.var(accel_features, axis=0)

                            accel_maximum= np.max(accel_features, axis=0)
                            accel_minimum= np.min(accel_features, axis=0)
                            accel_median= np.median(accel_features, axis=0)
                            accel_stat_range= accel_maximum-accel_minimum

                            accel_rms= np.sqrt(np.sum(accel_features ** 2, axis=0) / len(accel_features))
                            integrated_accel_rms= np.trapz(np.sqrt( accel_features ** 2 / len(accel_features) ), axis=0)

                            accel_corrcoef= list(map(lambda x: np.corrcoef(accel_features.T)[x[0], x[1]], pairs))

                            accel_cross_cor= list(map(lambda x: np.dot(accel_features[:,x[0]], accel_features[:, x[1]]) / len(accel_features), pairs))

                            accel_features= np.stack([accel_mean, accel_std, accel_var, accel_maximum, accel_minimum, accel_median, accel_stat_range, accel_rms, integrated_accel_rms, accel_corrcoef, accel_cross_cor])

                            gyro_features= gyro_df.iloc[gyro_indices][features].to_numpy()

                            gyro_mean= np.mean(gyro_features, axis=0)
                            gyro_std= np.std(gyro_features, axis=0)
                            gyro_var= np.var(gyro_features, axis=0)

                            gyro_maximum= np.max(gyro_features, axis=0)
                            gyro_minimum= np.min(gyro_features, axis=0)
                            gyro_median= np.median(gyro_features, axis=0)
                            gyro_stat_range= gyro_maximum-gyro_minimum

                            gyro_rms= np.sqrt(np.sum(gyro_features ** 2, axis=0) / len(gyro_features))
                            integrated_gyro_rms= np.trapz(np.sqrt( gyro_features ** 2 / len(gyro_features) ), axis=0)

                            gyro_corrcoef= list(map(lambda x: np.corrcoef(gyro_features.T)[x[0], x[1]], pairs))

                            gyro_cross_cor= list(map(lambda x: np.dot(gyro_features[:,x[0]], gyro_features[:, x[1]]) / len(gyro_features), pairs))

                            gyro_features= np.stack([gyro_mean, gyro_std, gyro_var, gyro_maximum, gyro_minimum, gyro_median, gyro_stat_range, gyro_rms, integrated_gyro_rms, gyro_corrcoef, gyro_cross_cor])

                            tmp_list.append(np.concatenate([accel_features, gyro_features], axis=1))
                        data_dict[key].append(tmp_list)
                    print(" Done")
        elif args.features=="frequency":
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
                    print(key, end="")
                    for idx_list in idx_lists:
                        tmp_list=[]
                        for indices in idx_list:
                            accel_indices= [i[0] for i in indices]
                            gyro_indices= [i[1] for i in indices]

                            accel_features= accel_df.iloc[accel_indices][features].to_numpy()

                            fft_features= fft(accel_features)
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
                            accel_features= np.concatenate([selected_normalised_fft_sum, selected_normalised_fft_entropy, dominant_magnitudes, dominant_frequencies])

                            gyro_features= gyro_df.iloc[gyro_indices][features].to_numpy()

                            fft_features= fft(gyro_features)
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
                            gyro_features= np.concatenate([selected_normalised_fft_sum, selected_normalised_fft_entropy, dominant_magnitudes, dominant_frequencies])

                            tmp_list.append(np.concatenate([accel_features, gyro_features]))
                        data_dict[key].append(tmp_list)
                    print(" Done")

        print("DONE")
        # if not os.path.exists(device_path):
        #     os.makedirs(device_path)

        pickle.dump(data_dict, open(f"{device_path}device-user-label_fusion_dict-{args.features}{args.scaling}.pkl", "wb"))
        print(f"SAVED AT: {device_path}")

    print("SAVING FOLD SETTING")
    for i in args.fold_setting:
        split_func=None
        if i=="stratified10":
            print("STRATIFIED10")
            fold_setting="stratified10"
            split_func= stratified_10_fold_split
        elif i=="modelfold":
            print("MODELFOLD")
            fold_setting="modelfold"
            split_func= model_fold_split
        elif i=="userfold":
            print("USERFOLD")
            fold_setting="userfold"
            split_func= user_fold_split

        if split_func!=None:
            save_path= f"{device_path}{fold_setting}-{args.features}{args.scaling}.pkl"
            split_func(data_dict, save_path, True)
