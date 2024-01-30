
import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os


def load_models():

    with open('./Model/rf_with_cutoff.pkl', 'rb') as rfc:
        sample_df = pickle.load(rfc)    
    with open('./Model/scalar.pkl', 'rb') as sc:
        scaler = pickle.load(sc)
    with open('./Model/pca.pkl', 'rb') as pca:
        pca_final = pickle.load(pca)
        print(pca_final)
    return sample_df, scaler, pca_final

def dataPreprocessing(uploaded_file):
    sample_df, scaler, pca_final=load_models()
    
    final_train_vars = ['id', 'arpu_8', 'onnet_mou_8', 'offnet_mou_8', 'roam_ic_mou_8', 'roam_og_mou_8', 'loc_og_t2t_mou_8', 'loc_og_t2m_mou_8', 'loc_og_t2f_mou_8', 'loc_og_t2c_mou_8', 'loc_og_mou_8', 'std_og_t2t_mou_8', 'std_og_t2m_mou_8', 'std_og_t2f_mou_8', 'std_og_mou_8', 'isd_og_mou_8', 'spl_og_mou_8', 'og_others_8', 'total_og_mou_8', 'loc_ic_t2t_mou_8', 'loc_ic_t2m_mou_8', 'loc_ic_t2f_mou_8', 'loc_ic_mou_8', 'std_ic_t2t_mou_8', 'std_ic_t2m_mou_8', 'std_ic_t2f_mou_8', 'std_ic_mou_8', 'total_ic_mou_8', 'spl_ic_mou_8', 'isd_ic_mou_8', 'ic_others_8', 'total_rech_num_8', 'total_rech_amt_8', 'max_rech_amt_8', 'last_day_rch_amt_8', 'total_rech_data_8', 'max_rech_data_8', 'count_rech_2g_8', 'count_rech_3g_8', 'av_rech_amt_data_8', 'vol_2g_mb_8', 'vol_3g_mb_8', 'arpu_3g_8', 'arpu_2g_8', 'night_pck_user_8', 'monthly_2g_8', 'sachet_2g_8', 'monthly_3g_8', 'sachet_3g_8', 'fb_user_8', 'aug_vbc_3g', 'jul_vbc_3g', 'jun_vbc_3g', 'churn_probability', 'total_rech_data_amt_8', 'avg_arpu_av67', 'avg_onnet_mou_av67', 'avg_offnet_mou_av67', 'avg_roam_ic_mou_av67', 'avg_roam_og_mou_av67', 'avg_loc_og_t2t_mou_av67', 'avg_loc_og_t2m_mou_av67', 'avg_loc_og_t2f_mou_av67', 'avg_loc_og_t2c_mou_av67', 'avg_loc_og_mou_av67', 'avg_std_og_t2t_mou_av67', 'avg_std_og_t2m_mou_av67', 'avg_std_og_t2f_mou_av67', 'avg_std_og_mou_av67', 'avg_isd_og_mou_av67', 'avg_spl_og_mou_av67', 'avg_og_others_av67', 'avg_total_og_mou_av67', 'avg_loc_ic_t2t_mou_av67', 'avg_loc_ic_t2m_mou_av67', 'avg_loc_ic_t2f_mou_av67', 'avg_loc_ic_mou_av67', 'avg_std_ic_t2t_mou_av67', 'avg_std_ic_t2m_mou_av67', 'avg_std_ic_t2f_mou_av67', 'avg_std_ic_mou_av67', 'avg_total_ic_mou_av67', 'avg_spl_ic_mou_av67', 'avg_isd_ic_mou_av67', 'avg_ic_others_av67', 'avg_total_rech_num_av67', 'avg_total_rech_amt_av67', 'avg_max_rech_amt_av67', 'avg_last_day_rch_amt_av67', 'avg_total_rech_data_av67', 'avg_max_rech_data_av67', 'avg_count_rech_2g_av67', 'avg_count_rech_3g_av67', 'avg_av_rech_amt_data_av67', 'avg_vol_2g_mb_av67', 'avg_vol_3g_mb_av67', 'avg_arpu_3g_av67', 'avg_arpu_2g_av67', 'avg_night_pck_user_av67', 'avg_monthly_2g_av67', 'avg_sachet_2g_av67', 'avg_monthly_3g_av67', 'avg_sachet_3g_av67', 'avg_fb_user_av67', 'avg_total_rech_data_amt_av67', 'aon_mon']
    # Load the CSV file into a DataFrame
    unseen = pd.read_csv(uploaded_file)
                
    # Data Cleaning on the Unseen data
    # Imputing columns with more than 70% Null values with constant 0
    rech_data_test = ['total_rech_data_6', 'total_rech_data_7', 'total_rech_data_8', 'max_rech_data_6', 'max_rech_data_7', 'max_rech_data_8', 'av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8']
    unseen[rech_data_test] = unseen[rech_data_test].apply(lambda x: x.fillna(0))
            
    # Deriving the new features for Total Data Recharge by taking the product of av_rech_amt_data_x and total_rech_data_x
    unseen['total_rech_data_amt_6'] = unseen['av_rech_amt_data_6'] * unseen['total_rech_data_6']
    unseen['total_rech_data_amt_7'] = unseen['av_rech_amt_data_7'] * unseen['total_rech_data_7']
    unseen['total_rech_data_amt_8'] = unseen['av_rech_amt_data_8'] * unseen['total_rech_data_8']

    # Deriving the new features to find the sum of the recharge amounts for calls and data
    unseen['total_reach_and_data_amt_6'] = unseen['total_rech_amt_6'] + unseen['total_rech_data_amt_6']
    unseen['total_reach_and_data_amt_7'] = unseen['total_rech_amt_7'] + unseen['total_rech_data_amt_7']

    # Deriving the new features to find the average of the recharge amounts for June and July
    unseen['avg_amt_6_and_7'] = (unseen['total_reach_and_data_amt_6'] + unseen['total_reach_and_data_amt_7']) / 2

    # Dropping the old columns
    unseen = unseen.drop(['total_reach_and_data_amt_6', 'total_reach_and_data_amt_7', 'avg_amt_6_and_7'], axis=1)
    # Converting date columns to datetime type
    cols_types_test = unseen.select_dtypes(include=['object'])
    for col in cols_types_test.columns:
        unseen[col] = pd.to_datetime(unseen[col])

    # Creating derived columns holding the average values of 6th & 7th months of all features
    col_list_test = unseen.filter(regex='_6|_7').columns.str[:-2]
    col_list_test.unique()
    
    for idx, col in enumerate(col_list_test.unique()):
            if(unseen.dtypes[col+"_6"] != 'datetime64[ns]'):
                avg_col_name_test = "avg_"+col+"_av67"  # lets create the column name dynamically
                col_6_test = col+"_6"
                col_7_test = col+"_7"
                unseen[avg_col_name_test] = (unseen[col_6_test] + unseen[col_7_test]) / 2

    # Dropping the original columns of June and July
    col_to_drop_test = unseen.filter(regex='_6|_7').columns
    unseen.drop(col_to_drop_test, axis=1, inplace=True)

    # Creating a new column "aon_mon" which holds the Age on Network of the user in months
    unseen['aon_mon'] = unseen['aon'] / 30

    # Dropping the original "aon" column (in days), as old feature is not required
    unseen.drop('aon', axis=1, inplace=True)
    # According to tenure_months, segregating users into respective tenure slots
    
    ten_range_test = [0, 6, 12, 24, 60, 61]
    ten_label_test = ['0-6 Months', '6-12 Months', '1-2 Yrs', '2-5 Yrs', '5 Yrs and above']
    unseen['tenure_range'] = pd.cut(unseen['aon_mon'], ten_range_test, labels=ten_label_test)

    # Dropping date_of_last_rech_8, date_of_last_rech_data_8 which is in date time format. Can't be used further, so dropping them.
    unseen.drop(['date_of_last_rech_8', 'date_of_last_rech_data_8'], axis=1, inplace=True)

    # Dropping the tenure_range column
    unseen.drop(['tenure_range'], axis=1, inplace=True)

    # Replacing nan values with constant 0 in all columns in the test/unseen data
    from numpy import nan
    for i in unseen.columns:
        unseen[i] = unseen[i].replace(nan, 0)
    # Making a list of all the final test dataset columns
    final_test_vars = list(unseen.columns)

    # Making a list of all the final variables for prediction by comparing with the train set columns
    final_vars = [x for x in final_train_vars if x in final_test_vars]
     # Filtering the final test dataset based on the final_vars
    submission_data = unseen[final_vars]
    submission_data = scaler.transform(submission_data)
    submission_data = pca_final.transform(submission_data)
    # Predicting on the Test data using the best Random Forest Algorithm
    result = sample_df.predict(submission_data)

    final_df = pd.DataFrame(result, columns=['churn_probability'])
    # Include the 'id' column in the final DataFrame
    final_df['id'] = unseen['id']
    # Reorder the columns to have 'id' as the first column
    cols = final_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    final_df = final_df[cols]
    os.makedirs("./Prediction", exist_ok=True)
    final_df.to_csv('./Prediction/submission_telecom_case_study_test.csv', index=False)
    
    
    
    