import numpy as np
import pandas as pd
import random
import cv2
import argparse

  
def main(args):
    random.seed(42)
    
    # from_json_p = f'./datasets/data_Allsight/json_data/{args.data_type}_train_{args.data_num}_{args.data_kind}.json'
    # trans_folder_path = './datasets/data_Allsight/'

    from_json_p = r'C:\Users\roblab20\Documents\yardenalon\Compression\datasets\paired_finger\json45\sim_train_5_aligned_fixed.json'
    trans_folder_path = r'C:\Users\roblab20\Documents\yardenalon\Compression\datasets\paired_finger'

    df_data = pd.read_json(from_json_p).transpose()
    
    if args.samples != 0:
        df_data = df_data.iloc[:args.samples,:]
        # df_data = df_data.sample(n=args.samples)
    
    for idx in range(len(df_data)):
        real_image = (cv2.imread(df_data['frame'][idx])).astype(np.uint8)
        save_path1 = trans_folder_path + '\\train' +args.folder_type + f'\\{idx}.jpg'  # Specify the path where you want to save the image
        save_path2 = trans_folder_path + '\\test' +args.folder_type + f'\\{idx}.jpg'
        cv2.imwrite(save_path1, real_image)
        cv2.imwrite(save_path2, real_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images and related JSON data.')
    parser.add_argument('--data_type', type=str, default='real', help='real, sim')
    parser.add_argument('--data_kind', type=str, default='transformed', help='transformed, aligned')
    parser.add_argument('--data_num', type=int, default=6, help='from JSON path')
    parser.add_argument('--folder_type', type=str, default='B', help='A, B')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples, if 0 -> not sample take all')
    args = parser.parse_args()

    main(args)