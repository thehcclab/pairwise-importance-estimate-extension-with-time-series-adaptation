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
    parser.add_argument("-device_sensor", help="Corresponding device and sensor's csv to be used. Options: (1) Phones Accelerometer (2) Phones Gyroscope (3) Watch Accelerometer (4) Watch Gyroscope . Default: '1'", default="1")
    parser.add_argument("-features", help="Features to be extracted. Options: ecdf, time, frequency, original. Default: original", default="original")
    parser.add_argument("fold_setting", nargs="*", help="Fold setting for training and validation. Options: stratified10, userfold, modelfold. Default: stratified10", default=["stratified10"])
    args= parser.parse_args()
    args.device_sensor= int(args.device_sensor)
    args.scaling= bool(args.scaling)
    args.features=str(args.features)

    assert args.features=="original" or args.features=="ecdf" or args.features=="time" or args.features=="frequency", f"feature must be ecdf, time, frequency or orginal. Found {args.features} instead"
    assert isinstance(args.fold_setting, List), f"fold_setting must be a list. Found {args.fold_setting} instead."
    for i in args.fold_setting:
        assert i=="stratified10" or i=="userfold" or i=="modelfold", f"fold setting must be a list containing: stratified10, userfold or modelfold. Found {i} instead"

    scaling= args.scaling
    if args.scaling:
        args.scaling="-minmax"
    else:
        args.scaling=""

    device_sensor= args.device_sensor
    if args.device_sensor==1:
        args.device_sensor="phone_accel"
    elif args.device_sensor==2:
        args.device_sensor="phone_gyro"
    elif args.device_sensor==3:
        args.devcie_sensor="watch_accel"
    elif args.device_sensor==4:
        args.device_sensor="watch_gyro"

    device_sensor_path= f"./HHAR/Activity recognition exp/{args.device_sensor}/"
    if not os.path.exists(device_sensor_path):
        os.makedirs(device_sensor_path)

    if os.path.exists(f"{device_sensor_path}device-user-label_dict-{args.features}{args.scaling}.pkl"):
        print("data dict already exists...")
        data_dict= pickle.load(open(f"{device_sensor_path}device-user-label_dict-{args.features}{args.scaling}.pkl", "rb"))
        print("PROCESSING DONE")
    else:
        if os.path.exists(f"{device_sensor_path}device-user-label_indicies{args.scaling}.pkl"):
            print("data indices already exists...")
            data_indices= pickle.load(open(f"{device_sensor_path}device-user-label_indicies{args.scaling}.pkl", "rb"))
            df= process_csv(device_sensor, scaling)
        else:
            print(f"PROCESSING SENSOR: {args.device_sensor}...", end="")
            data_indices, windows_dict, seq_dict, df= process_sensor_for_extraction(device_sensor, scaling)
            pickle.dump(data_indices, open(f"{device_sensor_path}device-user-label_indicies{args.scaling}.pkl", "wb"))

        print(f"Extracting features: {args.features}...", end="")
        if args.features=="original":
            data_dict= extract_original_measurements(data_indices, df)
        elif args.features=="ecdf":
            data_dict= extract_ecdf_features(data_indices, df)
        elif args.features=="time":
            data_dict= extract_time_features(data_indices, df)
        elif args.features=="frequency":
            data_dict= extract_frequency_features(data_indices, df)
        print("DONE")

        pickle.dump(data_dict, open(f"{device_sensor_path}device-user-label_dict-{args.features}{args.scaling}.pkl", "wb"))
        print(f"SAVED AT: {device_sensor_path}")
        print("DONE")

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
            save_path= f"{device_sensor_path}{fold_setting}-{args.features}{args.scaling}.pkl"
            split_func(data_dict, save_path, True)
