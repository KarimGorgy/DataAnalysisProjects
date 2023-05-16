import numpy as np
import pandas as pd

loan_data_backup = pd.read_csv('loan_data_2007_2014.csv')
loan_data = loan_data_backup.copy()


loan_data['emp_length_int'] = loan_data['emp_length'].str.replace('+ years','')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('< 1 year',str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('n/a',str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('years','')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('year','')

loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])
loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'],format = '%b-%y')
##pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']
loan_data['mths_since_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date'])/ np.timedelta64(1,'M')))
#print(loan_data['mths_since_earliest_cr_line'].describe())
loan_data['mths_since_earliest_cr_line'][loan_data['mths_since_earliest_cr_line']<0] = loan_data['mths_since_earliest_cr_line'].max()
#print(min(loan_data['mths_since_earliest_cr_line']))

loan_data['term'] = loan_data['term'].str.replace(' months','')
loan_data['term'] = loan_data['term'].str.replace(' ','')
loan_data['term'] = pd.to_numeric(loan_data['term'])

#print(loan_data['issue_d'].unique())
loan_data['mths_since_issue_d'] = pd.to_datetime(loan_data['issue_d'],format = '%b-%y')
loan_data['mths_since_issue_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - loan_data['mths_since_issue_d'])/np.timedelta64(1,'M')))
#print(loan_data['mths_since_issue_d'].describe())
loan_data_dummies = [pd.get_dummies(loan_data['grade'],prefix = 'grade',prefix_sep=':'),
                     pd.get_dummies(loan_data['sub_grade'],prefix = 'sub_grade',prefix_sep=':'),
                     pd.get_dummies(loan_data['home_ownership'],prefix = 'home_ownership',prefix_sep=':'),
                     pd.get_dummies(loan_data['verification_status'],prefix = 'verification_status',prefix_sep=':'),
                     pd.get_dummies(loan_data['loan_status'],prefix = 'loan_status',prefix_sep=':'),
                     pd.get_dummies(loan_data['purpose'],prefix = 'purpose',prefix_sep=':'),
                     pd.get_dummies(loan_data['addr_state'],prefix = 'addr_state',prefix_sep=':'),
                     pd.get_dummies(loan_data['initial_list_status'],prefix = 'initial_list_status',prefix_sep=':')]
loan_data_dummies = pd.concat(loan_data_dummies,axis=1)
loan_data = pd.concat([loan_data,loan_data_dummies],axis=1)
print(loan_data['total_rev_hi_lim'].isnull().sum())
loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'],inplace=True)
#loan_data['total_rev-hi_lim'].fillna(loan_data['funded_amnt'],inplace=True)
print(loan_data['total_rev_hi_lim'].isnull().sum())
loan_data['mths_since_earliest_cr_line'].fillna(0,inplace=True)
loan_data['acc_now_delinq'].fillna(0,inplace=True)
loan_data['total_acc'].fillna(0,inplace=True)
loan_data['pub_rec'].fillna(0,inplace=True)
loan_data['open_acc'].fillna(0,inplace=True)
loan_data['inq_last_6mths'].fillna(0,inplace=True)
loan_data['delinq_2yrs'].fillna(0,inplace=True)
loan_data['emp_length_int'].fillna(0,inplace=True)