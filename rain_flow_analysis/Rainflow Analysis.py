
# coding: utf-8

# In[2]:


import os

from python_includes import rainflow as rf

import numpy as np

import pandas as pd

import itertools

from python_includes.progressbar import ProgressBar


# In[2]:


def read_h5_file(file_path):
    
    return pd.HDFStore(file_path, mode='r')


# In[3]:


def read_bar_stresses(file_path):
    
    model = read_h5_file(file_path=file_path)
    
    bar_stress_df = model.get('/NASTRAN/RESULT/ELEMENTAL/STRESS/BAR')
    
    model.close()
    
    return bar_stress_df


# In[4]:


def read_results_domain_info(file_path):
    
    model = read_h5_file(file_path=file_path)
    
    domains_info = model.get('/NASTRAN/RESULT/DOMAINS')
    
    model.close()
    
    return domains_info


# In[5]:


def get_subcase_mapping_dict(df):
    
    df = df.loc[:, ['ID', 'SUBCASE']].copy()

    df.set_index('ID', inplace=True)

    return df.to_dict()


# In[6]:


def custom_division(val1, val2):
    try:
        res  = val1 / val2
    except ZeroDivisionError:
        res = np.nan
    return res


# In[7]:


def sn_reorder(df):
    
    lst_order = ['EID', 'Max', 'Min', 'max_line_Num', 'min_line_Num', 'R', 'Kt_const', 'Smax_const',
                 'S_Max', 'Kt_sigma', 'Seq', 'Nf', '_1_Over_Nf', 'One_Over_Nf_sum_of_1_over_Nf',
                 'Life', 'pct_damage']
    
    df = df[lst_order].copy()
    
    return df


# In[8]:


def get_bar_stress_df(h5_filepath):
    
    bar_stress_df = read_bar_stresses(h5_filepath) # Read BAR Element Stresses

    bar_stress_df = bar_stress_df[['EID', 'AX', 'DOMAIN_ID']]

    domain_results = read_results_domain_info(h5_filepath)

    subcase_mapping = get_subcase_mapping_dict(domain_results)

    bar_stress_df['SUBCASE'] = bar_stress_df.DOMAIN_ID.map(subcase_mapping['SUBCASE']).copy()
    
    return bar_stress_df


# In[9]:


def custom_manipulations(**kwargs):
    
    STRESS_DF = kwargs.get('stress_df')
    
    TEMPLATE_DF = kwargs.get('template_df')
    
    ELEMENT_ID = kwargs.get('eid')
    
    TYPE = kwargs.get('type')

    try:
        
        spectrum_df = get_spectrum_df(eid = ELEMENT_ID, template_df = TEMPLATE_DF, stress_df = STRESS_DF)

        SN_df = get_SN_df(spectrum_df)

        spec_df = change_spectrum_data_format(spectrum_df)
        
        write_spectrum_data_to_file(spec_df, ELEMENT_ID, TYPE)
        
        sn_df = change_SN_decimal_format(SN_df)
        
        write_SN_data_to_file(sn_df, ELEMENT_ID, TYPE)
        
    except:
        
        print('Failed Writing Fatigue data for ' + TYPE + ' element id: {}'.format(ELEMENT_ID))
        
        pass
        
    df = output2_data_df(SN_df)
    
    return df


# In[10]:


def output2_data_df(df):
    try:
        pre = df.loc[df.One_Over_Nf_sum_of_1_over_Nf == df.One_Over_Nf_sum_of_1_over_Nf.max(), :]
        return pre
    except:
        return pd.DataFrame({'A' : []})


# In[11]:


def get_spectrum_df(**kwargs):
    
    stress_df = kwargs.get('stress_df')
    
    template_df = kwargs.get('template_df')
    
    eid = kwargs.get('eid')
    
#     print(eid)
    
    eid_stress = stress_df[stress_df.EID == eid]

    cols = ['EID', 'AX_1', 'DOMAIN_ID', 'Case_1']

    eid_stress.columns = cols

    indx_cols = ['Case_1']

    eid_stress = eid_stress.set_index(indx_cols)

    df3 = template_df.join(eid_stress, on=indx_cols)

    df3 = df3.drop(['DOMAIN_ID'], axis=1)

    df3.AX_1 = df3.AX_1 * df3.Factor_1

    eid_stress_comb =  eid_stress.drop(columns=['EID', 'DOMAIN_ID'])

    eid_stress_comb.columns = ['AX_2']

    eid_stress_comb.index.name = 'Case_2'

    df3.Case_2 = df3.Case_2.replace('',0)

    df3 = df3.join(eid_stress_comb, on=['Case_2'])

    df3.AX_2 = df3.AX_2 * df3.Factor_2

    df3.AX_2.fillna(0, inplace=True)

    eid_stress_comb.columns = ['AX_3']

    eid_stress_comb.index.name = 'Case_3'

    df3.Case_3 = df3.Case_3.replace('',0)

    df3 = df3.join(eid_stress_comb, on=['Case_3'])      

    df3.AX_3 = df3.AX_3 * df3.Factor_3

    df3.AX_3.fillna(0, inplace=True)

    df3['Sigma_FEM'] = df3.AX_1  + df3.AX_2 + + df3.AX_3

    return df3


# In[12]:


def get_SN_df(spectrum_df):
    
    results = []

    for low, high, mult in rf.extract_cycles(spectrum_df.Sigma_FEM.tolist()):
    #     mean = 0.5 * (high+low)
    #     rng = high - low
        results.append((high, low))
        
    if not results: # print('empty')
        results.append((0.0, 0.0))

    rain_table = pd.DataFrame(results, columns=['Max', 'Min'])

    smax_const = 6.895

    kt_const = 3.0

    rain_table['EID'] = spectrum_df.EID.unique()[0]
    
    rain_table['R'] = custom_division(rain_table.Min, rain_table.Max)
    
    rain_table['Kt_const'] = kt_const
    
    rain_table['Smax_const'] = smax_const

    rain_table['S_Max'] = custom_division(rain_table.Max, rain_table.Smax_const)

    rain_table['Kt_sigma'] = rain_table.S_Max * rain_table.Kt_const

    rain_table['Seq'] = rain_table.Kt_sigma * (1 - rain_table.R) ** 0.52

    rain_table['Nf'] = 10 ** (20.83 - 9.09 * np.log10(rain_table.Seq))

    rain_table['_1_Over_Nf'] = custom_division(1, rain_table.Nf)

    rain_table['Index'] = np.arange(1, len(results)+1)

    rain_table  = rain_table.set_index(['Index'])
    
    sum_1_over_nf = rain_table._1_Over_Nf.sum()
    
    rain_table['One_Over_Nf_sum_of_1_over_Nf'] = custom_division(rain_table._1_Over_Nf, sum_1_over_nf)
    
    rain_table['Life'] = custom_division(1, float(sum_1_over_nf))
    
    s1 = pd.Series(dict((v,k) for k,v in spectrum_df.Sigma_FEM.iteritems()))
    # s1

    rain_table['max_line_Num'] = rain_table.Max.map(s1)
    
    rain_table['min_line_Num'] = rain_table.Min.map(s1)
    
    rain_table['pct_damage'] = rain_table.One_Over_Nf_sum_of_1_over_Nf * 100
    
    rain_table['pct_damage'] = rain_table['pct_damage'].map('{:,.1f}%'.format)
    
    rain_table = sn_reorder(rain_table)
    
    return rain_table


# # Change Spectrum data decimal format.

# In[13]:


def change_spectrum_data_format(df):
    
    df.fillna(value=0, inplace=True)

    df[['Case_1', 'Case_2', 'Case_3']] = df[['Case_1', 'Case_2', 'Case_3']].astype(int)

    spec_decimals = pd.Series([2, 0, 4, 0, 4, 0, 0, 
                               2, 2, 2, 2], 
                         index=['Factor_1', 'Case_1', 'Factor_2', 'Case_2', 'Factor_3', 'Case_3', 'EID', 
                                'AX_1', 'AX_2', 'AX_3', 'Sigma_FEM'])

    df = df.round(spec_decimals).copy()

    # df

    # df.dtypes

    change_spectrum_labels =  ['Factor_1', 'Case_1', 'Factor_2', 'Case_2', 'Factor_3', 'Case_3', 'Description', 'EID', 
                               'σFEM_1', 'σFEM_2', 'σFEM_3', 'ΣσFEM']

    df.columns = change_spectrum_labels
    
    return df


# # Change S-N data table decimal format.

# In[14]:


def change_SN_decimal_format(df):
    
    sn_decimals = pd.Series([2, 2, 1, 3, 2, 3, 2, 2, 2, 0, 1], 
                         index=['Max',
                                'Min', 
                                'Mult', 
                                'R', 
                                'Kt_const', 
                                'Smax_const', 
                                'S_Max', 
                                'Kt_sigma', 
                                'Seq', 
                                'One_Over_Nf_sum_of_1_over_Nf',
                                'Life'])

    df = df.round(sn_decimals)
    
    
    change_SN_labels = ['EID', 'Max', 'Min', 'max_line_Num', 'min_line_Num', 'R', 'Kt_const', 'Smax_const', 
                        'S_Max', 'Kt·σ in MPa', 'Seq in MPa', 'Nf', '1/Nf', '1/(Nf·Σ(1/Nf))', 'Life', '% damage']
    
    df.columns = change_SN_labels
    
    return custom_SN_data(df)


# In[15]:


def custom_SN_data(df):
    
    df = df[['EID', 'Max', 'Min', 'max_line_Num', 'min_line_Num', 'R', 'Kt_const', 'Smax_const', 
                        'S_Max', 'Kt·σ in MPa', 'Seq in MPa', 'Nf', '1/Nf', '1/(Nf·Σ(1/Nf))']]
    
    return df


# In[16]:


def custom_result_table_columns(df):
    change_SN_labels = ['Max', 'Min', 'max_line_Num', 'min_line_Num', 'R', 'Kt_const', 
                        'Smax_const', 'S_Max', 'Kt·σ in MPa', 'Seq in MPa', 'Nf', '1/Nf', 
                        '1/(Nf·Σ(1/Nf))', 'Life', '% damage']
    
    
    decimals = pd.Series([2, 2, 0, 0, 3, 2, 3, 2, 2, 2, 0, 1], 
                     index=['Max',
                            'Min', 
                            'max_line_Num', 
                            'min_line_Num', 
                            'R', 
                            'Kt_const', 
                            'Smax_const', 
                            'S_Max', 
                            'Kt_sigma', 
                            'Seq', 
                            'One_Over_Nf_sum_of_1_over_Nf',
                            'Life'])

    df = df.round(decimals)
    
    df = df[['Max', 'Min', 'max_line_Num', 'min_line_Num', 'R', 'Kt_const',
       'Smax_const', 'S_Max', 'Kt_sigma', 'Seq', 'Nf', '_1_Over_Nf',
       'One_Over_Nf_sum_of_1_over_Nf', 'Life', 'pct_damage']]
    
    df.columns = change_SN_labels
    
    return df


# In[17]:


def write_spectrum_data_to_file(spec_df, eid, resultType):
    
    with open('Fatigue_output.out','a', encoding="utf-8") as outfile:
        
        outfile.writelines(str(resultType) + ' Elm ' + str(eid) + '\n')
        
        spec_df.to_string(outfile, columns=spec_df.columns.tolist())
        
        outfile.writelines('\n \n \n')


# In[18]:


def write_SN_data_to_file(sn_df, eid, resultType):
    
    with open('Fatigue_output.out','a', encoding="utf-8") as outfile:
        
        outfile.writelines(str(resultType) + ' Elm ' + str(eid) + '\n')
        
        sn_df.to_string(outfile, columns=sn_df.columns.tolist())
        
        outfile.writelines('\nΣ(1/Nf) ' + str(round(sn_df['1/Nf'].sum())))
        
        outfile.writelines('\n' + str(resultType) + ' life : \t' + str(round(sn_df.Nf.tolist()[-1], 1)) +'\t FC (unfactored)')
        
        outfile.writelines('\n \n \n')


# # MAIN START

# In[19]:


premod_h5 = 'G:/PROJECTS/CAEP1073 - EIS PC-12 Radar Installation/wip/FEM/GFEM- M3/M3.1run/1.pre-mod/sol101_fatigue/pc12-47-v1-6-fatigue.h5'

postmod_h5 = 'G:/PROJECTS/CAEP1073 - EIS PC-12 Radar Installation/wip/FEM/GFEM- M3/M3.1run/2.post-mod/sol101_fatigue/pc12-47-v1-6-fatigue.h5'

# Input Element ID
ELEMENT_ID = '' #1118126

factor_1 =  {
    1 :   1.0000,
    2 :   1.0000,
    3 :   1.0000,
    4 :   1.0000,
    5 :   0.4000,
    6 :   1.0000,
    7 :   0.4000,
    8 :   1.0000,
    9 :   1.0000,
    10:   1.0000,
    11:   1.0000,
    12:   1.0000,
    13:   1.0000,
    14:   1.0000,
    15:   1.0000,
    16:   1.0000,
    17:   1.0000,
    18:   1.0000,
    19:   1.0000,
    20:   1.0000,
    21:   1.0000,
    22:   1.0000,
    23:   1.0000,
    24:   1.0000,
    25:   1.0000,
    26:   1.0000}


case_1 =  {
    1 :502,
    2 :502,
    3 :503,
    4 :503,
    5 :503,
    6 :503,
    7 :503,
    8 :503,
    9 :506,
    10:503,
    11:507,
    12:503,
    13:503,
    14:508,
    15:509,
    16:508,
    17:510,
    18:508,
    19:511,
    20:508,
    21:512,
    22:508,
    23:515,
    24:516,
    25:502,
    26:502}


factor_2 =  {
    1 :  0.0000,
    2 :  1.0000,
    3 :  0.0000,
    4 :  0.4482,
    5 :  0.6000,
    6 :  0.4482,
    7 :  0.6000,
    8 :  0.4482,
    9 :  0.4482,
    10:  0.4482,
    11:  0.4482,
    12:  0.4482,
    13:   0.0000,
    14:   0.0000,
    15:   0.0000,
    16:   0.0000,
    17:   0.0000,
    18:   0.0000,
    19:   0.0000,
    20:   0.0000,
    21:   0.0000,
    22:   0.0000,
    23:   0.0000,
    24:   0.0000,
    25:   0.2500,
    26:   0.0000}


case_2 =  {
    1 :'',
    2 :513,
    3 :'',
    4 :523,
    5 :504,
    6 :523,
    7 :505,
    8 :523,
    9 :523,
    10:523,
    11:523,
    12:523,
    13:'',
    14:'',
    15:'',
    16:'',
    17:'',
    18:'',
    19:'',
    20:'',
    21:'',
    22:'',
    23:'',
    24:'',
    25:514,
    26:''}

factor_3 =  {
    1 :   0.0000,
    2 :   0.0000,
    3 :   0.0000,
    4 :   0.0000,
    5 :   0.4482,
    6 :   0.0000,
    7 :   0.4482,
    8 :   0.0000,
    9 :   0.0000,
    10:   0.0000,
    11:   0.0000,
    12:   0.0000,
    13:   0.0000,
    14:   0.0000,
    15:   0.0000,
    16:   0.0000,
    17:   0.0000,
    18:   0.0000,
    19:   0.0000,
    20:   0.0000,
    21:   0.0000,
    22:   0.0000,
    23:   0.0000,
    24:   0.0000,
    25:   0.0000,
    26:   0.0000}


case_3 =  {
    1 :'',
    2 :'',
    3 :'',
    4 :'',
    5 :523,
    6 :'',
    7 :523,
    8 :'',
    9 :'',
    10:'',
    11:'',
    12:'',
    13:'',
    14:'',
    15:'',
    16:'',
    17:'',
    18:'',
    19:'',
    20:'',
    21:'',
    22:'',
    23:'',
    24:'',
    25:'',
    26:''}

descr =  {
    1 :   'Standing on the ground',
    2 :   'Thrust & Torque',
    3 :   'Cruise (1g Level Flight)',
    4 :   'Cruise (1g Level Flight) + Cabin Pressure',
    5 :   'Cruise with Vertical Gust vz = 6 [m/s] + Cabin Pressure',
    6 :   'Cruise (1g Level Flight) + Cabin Pressure',
    7 :   'Cruise with Vertical Gust vz = -6 [m/s] + Cabin Pressure',
    8 :   'Cruise (1g Level Flight) + Cabin Pressure',
    9 :   'Cruise with Lateral Gust vz = 10 [m/s] + Cabin Pressure',
    10:   'Cruise (1g Level Flight) + Cabin Pressure',
    11:   'Cruise with Lateral Gust vz = -10 [m/s] + Cabin Pressure',
    12:   'Cruise (1g Level Flight) + Cabin Pressure',
    13:   'Cruise (1g Level Flight)',
    14:   'Final Approach (1g Level Flight)',
    15:   'Final Approach with Vertical Gust vz = 10 [m/s]',
    16:   'Final Approach (1g Level Flight)',
    17:   'Final Approach with Vertical Gust vz = -10 [m/s]',
    18:   'Final Approach (1g Level Flight)',
    19:   'Final Approach with Lateral Gust vz = 10 [m/s]',
    20:   'Final Approach (1g Level Flight)',
    21:   'Final Approach with Lateral Gust vz = -10 [m/s]',
    22:   'Final Approach (1g Level Flight)',
    23:   'Landing (Spin Up) vz = 6 [ft/s]',
    24:   'Landing (Spring Back) vz = 6 [ft/s]',
    25:   'Braking nx = -0.25g',
    26:   'Standing on the Ground'}
    
df_template = {
    'Factor_1': factor_1,
    'Case_1': case_1,
    'Factor_2': factor_2,
    'Case_2': case_2,
    'Factor_3': factor_3,
    'Case_3': case_3,
    'Description': descr
    
}


# In[20]:


TEMPLATE_DF = pd.DataFrame(df_template)

TEMPLATE_DF.index.name = 'Index'

# TEMPLATE_DF


# In[21]:


# PREMOD_BAR_STRESS_DF = get_bar_stress_df(premod_h5)

# spec_df = get_spectrum_df_1(eid = ELEMENT_ID, template_df = TEMPLATE_DF, stress_df = PREMOD_BAR_STRESS_DF)

# spec_df


# In[22]:


# TEMPLATE_DF = get_template_df(**df_template)

# PREMOD OPERATIONS
PREMOD_BAR_STRESS_DF = get_bar_stress_df(premod_h5)

# premod_spectrum_df = get_spectrum_df(eid = ELEMENT_ID, template_df = TEMPLATE_DF, stress_df = PREMOD_BAR_STRESS_DF)

# premod_rain_table = get_SN_df(premod_spectrum_df)

# POSTMOD OPERATIONS
POSTMOD_BAR_STRESS_DF = get_bar_stress_df(postmod_h5)

# postmod_spectrum_df = get_spectrum_df(eid = ELEMENT_ID, template_df = TEMPLATE_DF, stress_df = POSTMOD_BAR_STRESS_DF)

# postmod_rain_table = get_SN_df(postmod_spectrum_df)

pre_template = {
    'stress_df': PREMOD_BAR_STRESS_DF,
    'template_df': TEMPLATE_DF,
    'eid': ELEMENT_ID,
    'type': 'PRE-MOD'
    
}

post_template = {
    'stress_df': POSTMOD_BAR_STRESS_DF,
    'template_df': TEMPLATE_DF,
    'eid': ELEMENT_ID,
    'type': 'POST-MOD'
    
}


# In[28]:


try:
    os.remove("Fatigue_output.out")
    print('Found: Fatigue_output.out \n Action: Deleted')
except FileNotFoundError:
    #print('File Not Exist')
    pass

premod_bar_eids = PREMOD_BAR_STRESS_DF.EID.unique()

postmod_bar_eids = POSTMOD_BAR_STRESS_DF.EID.unique()

premod_result = []

postmod_result = []

#         premod_result.append((eid, get_premod_bar_element_life(eid = eid, template_df = TEMPLATE_DF, stress_df = PREMOD_BAR_STRESS_DF)))

pre_bar_summary_df = pd.DataFrame({'A' : []})

post_bar_summary_df = pd.DataFrame({'A' : []})

total_eids = len(premod_bar_eids) + len(postmod_bar_eids)

progress = ProgressBar(total_eids, fmt=ProgressBar.FULL)

for eid in premod_bar_eids:
    
    progress.current += 1
    
    progress()
    
    pre_template['eid'] = eid
    try:
        df = custom_manipulations(**pre_template)

        if pre_bar_summary_df.empty == True:
            pre_bar_summary_df = df.copy()
        else:
            pre_bar_summary_df = pd.concat([pre_bar_summary_df, df], axis=0, sort=False)

    except IndexError:
        print('premod failed at EID: {}'.format(eid))
        
        
for eid in postmod_bar_eids:
    
    progress.current += 1
    
    progress()
    
    post_template['eid'] = eid
    
    try:
        df = custom_manipulations(**post_template)
        
        if post_bar_summary_df.empty == True:
            post_bar_summary_df = df.copy()
        else:
            post_bar_summary_df = pd.concat([post_bar_summary_df, df], axis=0, sort=False)                                            
        
    except IndexError:
        print('postmod failed at EID: {}'.format(eid))


# In[29]:


pre_bar_summary_df.drop_duplicates(subset=['EID'], inplace=True)

pre_bar_summary_df = pre_bar_summary_df.reset_index()

post_bar_summary_df.drop_duplicates(subset=['EID'], inplace=True)

post_bar_summary_df = post_bar_summary_df.reset_index()

# pre_bar_summary_df.head()

# post_bar_summary_df.head()

pre_bar_summary_df.set_index('EID', inplace=True)

post_bar_summary_df.set_index('EID', inplace=True)

pre_bar_summary_df = custom_result_table_columns(pre_bar_summary_df)

pre_bar_summary_df = pre_bar_summary_df.add_prefix('PRE-MOD_')

# pre_bar_summary_df.head()

post_bar_summary_df = custom_result_table_columns(post_bar_summary_df)

post_bar_summary_df = post_bar_summary_df.add_prefix('POST-MOD_')

# post_bar_summary_df.head()

result_df = pd.concat([pre_bar_summary_df, post_bar_summary_df], axis=1, sort=False)

result_df['knockdown_factor'] = custom_division(result_df['POST-MOD_Life'], result_df['PRE-MOD_Life'])


# # Write to excel

# In[30]:


with pd.ExcelWriter('Fatigue_output.xlsx') as writer:  # doctest: +SKIP
    
#     premod_spectrum_df.to_excel(writer, sheet_name='premod_spectrum')
    
#     premod_rain_table.to_excel(writer, sheet_name='premod_S-N')
    
#     postmod_spectrum_df.to_excel(writer, sheet_name='postmod_spectrum')
    
#     postmod_rain_table.to_excel(writer, sheet_name='postmod_S-N')
    
    result_df.to_excel(writer, sheet_name='Life_estimation')


# In[31]:


progress.done()

print('ALL DONE WELL!')


# In[33]:


# len(premod_bar_eids)

# len(postmod_bar_eids)

# result_df.shape

# PREMOD_BAR_STRESS_DF.shape

# eids = list(itertools.zip_longest(premod_bar_eids, postmod_bar_eids))

# df = pd.DataFrame(eids, columns=['PRE', 'POST'])
# df.to_excel('eids.xlsx')

