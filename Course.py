import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#read file
loan_data_backup = pd.read_csv('loan_data_2007_2014.csv')
# Make copy from backup 
loan_data = loan_data_backup.copy()

# CLEANING DATA #--------------------------------------------------------------------

# parse employement length so we can turn it to integer
loan_data['emp_length_int'] = loan_data['emp_length'].str.replace('+ years','')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('< 1 year',str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('n/a',str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('years','')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('year','')

# convert variable to integer
loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])

# Convert String variable holding date, to datetime type
loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'],format = '%b-%y')

# Add new column holding the difference between earliest cr line date and base date, in months 
loan_data['mths_since_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date'])/ np.timedelta64(1,'M')))
# Because we noticed negative values due to the library having the oldest date newer than some dates we have, make the negative values have the maximum difference from the base date as the extra difference is negligible
loan_data['mths_since_earliest_cr_line'][loan_data['mths_since_earliest_cr_line']<0] = loan_data['mths_since_earliest_cr_line'].max()

# Similar process with variable term
loan_data['term'] = loan_data['term'].str.replace(' months','')
loan_data['term'] = loan_data['term'].str.replace(' ','')
loan_data['term'] = pd.to_numeric(loan_data['term'])

# Similar process with mths...
loan_data['mths_since_issue_d'] = pd.to_datetime(loan_data['issue_d'],format = '%b-%y')
loan_data['mths_since_issue_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - loan_data['mths_since_issue_d'])/np.timedelta64(1,'M')))

# Create dummy variables that will be used later on, store them in a new dataframe
# These are new variables that are sort of binary. New columns holding wether they are characterized by a specific category or not
loan_data_dummies = [pd.get_dummies(loan_data['grade'],prefix = 'grade',prefix_sep=':'),
                     pd.get_dummies(loan_data['sub_grade'],prefix = 'sub_grade',prefix_sep=':'),
                     pd.get_dummies(loan_data['home_ownership'],prefix = 'home_ownership',prefix_sep=':'),
                     pd.get_dummies(loan_data['verification_status'],prefix = 'verification_status',prefix_sep=':'),
                     pd.get_dummies(loan_data['loan_status'],prefix = 'loan_status',prefix_sep=':'),
                     pd.get_dummies(loan_data['purpose'],prefix = 'purpose',prefix_sep=':'),
                     pd.get_dummies(loan_data['addr_state'],prefix = 'addr_state',prefix_sep=':'),
                     pd.get_dummies(loan_data['initial_list_status'],prefix = 'initial_list_status',prefix_sep=':')]

#
loan_data_dummies = pd.concat(loan_data_dummies,axis=1)

# Add the columns with the dummy variables to the original dataframe
loan_data = pd.concat([loan_data,loan_data_dummies],axis=1)

# For the total_rev_hi_lim, wherever we find a missing value, we assume it is the funded amount, so we store it rather than the na
loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'],inplace=True)
# Here, we store 0 rather than na
loan_data['mths_since_earliest_cr_line'].fillna(0,inplace=True)
loan_data['acc_now_delinq'].fillna(0,inplace=True)
loan_data['total_acc'].fillna(0,inplace=True)
loan_data['pub_rec'].fillna(0,inplace=True)
loan_data['open_acc'].fillna(0,inplace=True)
loan_data['inq_last_6mths'].fillna(0,inplace=True)
loan_data['delinq_2yrs'].fillna(0,inplace=True)
loan_data['emp_length_int'].fillna(0,inplace=True)

# Create a new column describing wether the lessee has defaulted or not base on the loan status
loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(['Charged Off','Default','Does not meet the credit policy. Status:Charged Off','Late (31-120 days)']),0,1)

# Split data into train and test
loan_data_inputs_train,loan_data_inputs_test,loan_data_targets_train,loan_data_targets_test = train_test_split(loan_data.drop('good_bad',axis=1),loan_data['good_bad'],test_size=0.2,random_state = 42)
