import pandas as pd
import glob
import os
import json

CORRUPTIONS = ['defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression','gaussian_noise', 'shot_noise', 'impulse_noise']
CO_LEVEL = ['1','2','3','4','5']
pattern_aacc = 'aAcc'
pattern_miou = 'mIoU'
pattern_macc = 'mAcc'


data_root = 'local_results/ss/ADE20k/gcnet_r50-d8_4xb4-80k_ade20k-512x512_original'

imc_index = ['clean']+CORRUPTIONS
imc_columns = CO_LEVEL

imc_df_aacc = pd.DataFrame(data=0, columns=imc_columns, index=imc_index)
imc_df_miou = pd.DataFrame(data=0, columns=imc_columns, index=imc_index)
imc_df_macc = pd.DataFrame(data=0, columns=imc_columns, index=imc_index)



for co in imc_index:
    
    for l in imc_columns:
        
        if co == 'clean':
            data_path = glob.glob(os.path.join(data_root,co,'*/*.json'))[0]
            
            
            contents = []
            with open(data_path, 'r') as file:  
                for line in file.readlines():
                    dic = json.loads(line)
                    contents.append(dic)
            
            value_aacc = contents[-1][pattern_aacc]
            value_miou = contents[-1][pattern_miou]
            value_macc = contents[-1][pattern_macc]

            imc_df_aacc.loc[co] = round(value_aacc,2)
            imc_df_miou.loc[co] = round(value_miou,2)
            imc_df_macc.loc[co] = round(value_macc,2)

            break
        else:
            data_path = glob.glob(os.path.join(data_root,co,l,'*/*.json'))[0]
            
            contents = []
            with open(data_path, 'r') as file:  
                for line in file.readlines():
                    dic = json.loads(line)
                    contents.append(dic)
                    
            value_aacc = contents[-1][pattern_aacc]
            value_miou = contents[-1][pattern_miou]
            value_macc = contents[-1][pattern_macc]

            imc_df_aacc.loc[co,l] = round(value_aacc,2)
            imc_df_miou.loc[co,l] = round(value_miou,2)
            imc_df_macc.loc[co,l] = round(value_macc,2)
            
imc_df_aacc['avg'] = imc_df_aacc.mean(axis=1).round(2)
average_excluding_first_row_aacc = imc_df_aacc[1:].avg.mean()

imc_df_miou['avg'] = imc_df_miou.mean(axis=1).round(2)
average_excluding_first_row_miou = imc_df_miou[1:].avg.mean()

imc_df_macc['avg'] = imc_df_macc.mean(axis=1).round(2)
average_excluding_first_row_macc = imc_df_macc[1:].avg.mean()


total_index = ['ADE20k']
total_columns = ['aAcc','mIoU','mAcc']
total_df = pd.DataFrame(data=0, columns=total_columns, index=total_index)
total_df.loc['ADE20k','aAcc'] = round(imc_df_aacc.loc['clean','1'],2)
total_df.loc['ADE20k','mIoU'] = round(imc_df_miou.loc['clean','1'],2)
total_df.loc['ADE20k','mAcc'] = round(imc_df_macc.loc['clean','1'],2)

total_df.loc['ADE20k-C','aAcc'] = round(average_excluding_first_row_aacc,2)
total_df.loc['ADE20k-C','mIoU'] = round(average_excluding_first_row_miou,2)
total_df.loc['ADE20k-C','mAcc'] = round(average_excluding_first_row_macc,2)

# total_df.loc['COCO','coco/bbox_mAP'] = round(imc_df.loc['clean','1'],1)
# total_df.loc['COCO-C','coco/bbox_mAP'] = round(average_excluding_first_row,1)
# total_df.loc['COCO-C_s','coco/bbox_mAP'] = round(average_excluding_first_row_s,1)
# total_df.loc['COCO-C_m','coco/bbox_mAP'] = round(average_excluding_first_row_m,1)
# total_df.loc['COCO-C_l','coco/bbox_mAP'] = round(average_excluding_first_row_l,1)


imc_df_path_aacc = os.path.join(data_root,'detailed_aAcc.csv')
imc_df_path_miou = os.path.join(data_root,'detailed_mIoU.csv')
imc_df_path_macc = os.path.join(data_root,'detailed_mAcc.csv')

total_path = os.path.join(data_root,'total.csv')


imc_df_aacc.to_csv(imc_df_path_aacc)
imc_df_miou.to_csv(imc_df_path_miou)
imc_df_macc.to_csv(imc_df_path_macc)


total_df.to_csv(total_path)
print('Done!')