# %% 

#%pip install scikit-learn
#%pip install seaborn

# %%

import splusdata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
import aplpy
import pickle
import astropy.constants as c
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.transforms import Affine2D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

# %%

# conn = splusdata.connect('barbaralunarti','Bah@wwe12')
conn = splusdata.connect('hektor.monteiro','oiii5007')

# %%

loaded_gal_sample = pd.read_pickle('gal_sample.pkl')


# %%

iso_list=[]
for i in range(0,len(loaded_gal_sample)-1):
    with open(f'iso/{loaded_gal_sample["Object"].iloc[i]}.pkl', mode='rb') as file:
        iso_list.append(pickle.load(file))
        
auto_list=[]
for i in range(0,len(loaded_gal_sample)-1):
    with open(f'auto/{loaded_gal_sample["Object"].iloc[i]}.pkl', mode='rb') as file:
        auto_list.append(pickle.load(file))
        
petro_list=[]
for i in range(0,len(loaded_gal_sample)-1):
    with open(f'petro/{loaded_gal_sample["Object"].iloc[i]}.pkl', mode='rb') as file:
        petro_list.append(pickle.load(file))
        
# %%
select_table=27
obj_index=30
print(loaded_gal_sample['Object'][select_table])
iso_keys = [
    'u_iso', 'J0378_iso', 'J0395_iso', 'J0410_iso',
    'J0430_iso', 'g_iso', 'J0515_iso', 'r_iso', 'J0660_iso', 'i_iso',
    'J0861_iso', 'z_iso', 'e_u_iso', 'e_J0378_iso', 'e_J0395_iso',
    'e_J0410_iso', 'e_J0430_iso', 'e_g_iso', 'e_J0515_iso', 'e_r_iso',
    'e_J0660_iso', 'e_i_iso', 'e_J0861_iso', 'e_z_iso'
]
iso_values = {key: iso_list[select_table][obj_index][key] for key in iso_keys}

auto_keys = [
    'u_auto', 'J0378_auto', 'J0395_auto', 'J0410_auto',
    'J0430_auto', 'g_auto', 'J0515_auto', 'r_auto', 'J0660_auto', 'i_auto',
    'J0861_auto', 'z_auto', 'e_u_auto', 'e_J0378_auto', 'e_J0395_auto',
    'e_J0410_auto', 'e_J0430_auto', 'e_g_auto', 'e_J0515_auto', 'e_r_auto',
    'e_J0660_auto', 'e_i_auto', 'e_J0861_auto', 'e_z_auto'
]
auto_values = {key: auto_list[select_table][obj_index][key] for key in auto_keys}

petro_keys = [
    'u_petro', 'J0378_petro', 'J0395_petro', 'J0410_petro',
    'J0430_petro', 'g_petro', 'J0515_petro', 'r_petro', 'J0660_petro', 'i_petro',
    'J0861_petro', 'z_petro', 'e_u_petro', 'e_J0378_petro', 'e_J0395_petro',
    'e_J0410_petro', 'e_J0430_petro', 'e_g_petro', 'e_J0515_petro', 'e_r_petro',
    'e_J0660_petro', 'e_i_petro', 'e_J0861_petro', 'e_z_petro'
]
petro_values = {key: petro_list[select_table][obj_index][key] for key in petro_keys}

# %%

# List comprehension to filter out "e_*" strings
non_e_iso_keys = [key for key in iso_keys if not key.startswith('e_')]
non_e_auto_keys = [key for key in auto_keys if not key.startswith('e_')]
non_e_petro_keys = [key for key in petro_keys if not key.startswith('e_')]

iso_error_values = [iso_values[f'e_{key}'] for key in non_e_iso_keys]
iso_non_error_values = [iso_values[f'{key}'] for key in non_e_iso_keys]

auto_error_values = [auto_values[f'e_{key}'] for key in non_e_auto_keys]
auto_non_error_values = [auto_values[f'{key}'] for key in non_e_auto_keys]

petro_error_values = [petro_values[f'e_{key}'] for key in non_e_petro_keys]
petro_non_error_values = [petro_values[f'{key}'] for key in non_e_petro_keys]

# %%

band_keys = ['u', 'J0378', 'J0395', 'J0410',
    'J0430', 'g', 'J0515', 'r', 'J0660', 'i',
    'J0861', 'z']

plt.figure(figsize=(10, 6))
plt.errorbar(band_keys, iso_non_error_values, yerr=iso_error_values,
             label='ISO', fmt='o', capsize=5, color='green')
plt.errorbar(band_keys, auto_non_error_values, yerr=auto_error_values,
             label='AUTO', fmt='o', capsize=5, color='red')
plt.errorbar(band_keys, petro_non_error_values, yerr=petro_error_values,
             label='PETRO', fmt='o', capsize=5, color='mediumblue')
plt.xlabel("Filter")
plt.ylabel("AB Mag")
plt.title("AM 0330-324")
plt.legend()
plt.xticks(rotation=0, ha='right')
plt.tight_layout()
plt.show()

# %%

band_select=8
x_values = ["ISO","AUTO","PETRO"]
plt.figure(figsize=(5, 6))
plt.errorbar(x_values[0], iso_non_error_values[band_select], yerr=iso_error_values[band_select],
             label=f'ISO ± {iso_error_values[band_select]:.4f}', fmt='o', capsize=5, color='green')
plt.errorbar(x_values[1], auto_non_error_values[band_select], yerr=auto_error_values[band_select],
             label=f'AUTO ± {auto_error_values[band_select]:.4f}', fmt='o', capsize=5, color='red')
plt.errorbar(x_values[2], petro_non_error_values[band_select], yerr=petro_error_values[band_select],
             label=f'PETRO ± {petro_error_values[band_select]:.4f}', fmt='o', capsize=5, color='mediumblue')
plt.xlabel("Bands")
plt.ylabel("Mag")
plt.title("AM 0332-324 - filtro F660")
plt.legend()
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.show()

# %%

filter_select='u'
objs_DT_ID=['iDR4_3_SPLUS-s38s28_0037355',
 'iDR4_3_SPLUS-s42s26_0009603',
 'iDR4_3_SPLUS-s37s33_0029178',
 'iDR4_3_SPLUS-n14s32_0029533',
 'iDR4_3_SPLUS-s19s23_0054919',
 'iDR4_3_SPLUS-s20s23_0027579',
 'iDR4_3_SPLUS-s27s07_0017147',
 'iDR4_3_SPLUS-s26s07_0019433',
 'iDR4_3_SPLUS-s21s10_0013331',
 'iDR4_3_SPLUS-s27s10_0013261',
 'iDR4_3_SPLUS-s24s11_0019810',
 'iDR4_3_SPLUS-s26s14_0016119',
 'iDR4_3_SPLUS-s25s18_0032854',
 'iDR4_3_SPLUS-s27s18_0020395',
 'iDR4_3_SPLUS-s27s19_0008651',
 'iDR4_3_SPLUS-s24s20_0018168',
 'iDR4_3_SPLUS-s25s21_0007448',
 'iDR4_3_SPLUS-s25s24_0039312',
 'iDR4_3_SPLUS-s37s20_0007226',
 'iDR4_3_SPLUS-s34s21_0008123',
 'iDR4_3_SPLUS-s24s26_0024907',
 'iDR4_3_SPLUS-s29s31_0019272',
 'iDR4_3_MC0103_0014596',
 'iDR4_3_SPLUS-s25s34_0007124',
 'iDR4_3_SPLUS-s24s35_0029130',
 'iDR4_3_SPLUS-s25s35_0013478',
 'iDR4_3_SPLUS-s30s34_0033827',
 'iDR4_3_SPLUS-s44s24_0025720',
 'iDR4_3_SPLUS-s28s36_0033655',
 'iDR4_3_SPLUS-s43s27_0018255',
 'iDR4_3_SPLUS-s41s30_0038650',
 'iDR4_3_SPLUS-s45s26_0000595',
 'iDR4_3_SPLUS-s38s32_0024249',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s30s39_0009952',
 'iDR4_3_SPLUS-s34s36_0027270',
 'iDR4_3_SPLUS-s37s34_0035918',
 'iDR4_3_SPLUS-s38s34_0035796',
 'iDR4_3_MC0001_0037018',
 'iDR4_3_SPLUS-s30s41_0001982',
 'iDR4_3_SPLUS-s25s45_0034698',
 'iDR4_3_MC0012_0039189',
 'iDR4_3_SPLUS-s39s34_0021319',
 'iDR4_3_SPLUS-s41s32_0034511',
 'iDR4_3_MC0021_0034624',
 'iDR4_3_SPLUS-s29s42_0019922',
 'iDR4_3_SPLUS-s41s33_0032981',
 'iDR4_3_MC0002_0006559',
 'iDR4_3_SPLUS-s36s39_0020464',
 'iDR4_3_SPLUS-n17s01_0006918',
 'iDR4_3_HYDRA-0104_0011186',
 'iDR4_3_HYDRA-0051_0052680',
 'iDR4_3_HYDRA-0015_0018824',
 'iDR4_3_HYDRA-0106_0049457',
 'iDR4_3_HYDRA-0028_0034942',
 'iDR4_3_HYDRA-0085_0056446',
 'iDR4_3_SPLUS-n20s03_0026145',
 'iDR4_3_SPLUS-n19s22_0026754',
 'iDR4_3_SPLUS-n20s23_0017293',
 'iDR4_3_SPLUS-s30s45_0010034',
 'iDR4_3_SPLUS-s20s23_0040961',
 'iDR4_3_SPLUS-s19s23_0047619',
 'iDR4_3_SPLUS-s23s26_0001674',
 'iDR4_3_SPLUS-s23s26_0041931',
 'iDR4_3_SPLUS-s24s65_0036645',
 'iDR4_3_SPLUS-s45s41_0034849',
 'iDR4_3_SPLUS-s23s46_0035512',
 'iDR4_3_SPLUS-s24s74_0016239',
 'iDR4_3_STRIPE82-0063_0034198',
 'iDR4_3_SPLUS-n13s31_0042282',
 'iDR4_3_SPLUS-n14s32_0041298',
 'iDR4_3_SPLUS-s39s31_0013592',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s21s23_0012649',
 'iDR4_3_HYDRA-0045_0074200',
 'iDR4_3_SPLUS-s34s39_0021318',
 'iDR4_3_SPLUS-s25s35_0015877',
 'iDR4_3_SPLUS-s25s13_0012636',
 'iDR4_3_MC0125_0008151',
 'iDR4_3_MC0097_0015643',
 'iDR4_3_SPLUS-s46s24_0023799',
 'iDR4_3_SPLUS-s46s24_0023407',
 'iDR4_3_SPLUS-s46s25_0034017',
 'iDR4_3_SPLUS-s42s30_0018253',
 'iDR4_3_SPLUS-s42s31_0030929',
 'iDR4_3_SPLUS-s42s32_0018085',
 'iDR4_3_SPLUS-s38s32_0022611',
 'iDR4_3_SPLUS-s38s33_0011808',
 'iDR4_3_SPLUS-s25s13_0017486',
 'iDR4_3_SPLUS-s29s42_0028164',
 'iDR4_3_SPLUS-s26s45_0031867',
 'iDR4_3_HYDRA-0107_0006292',
 'iDR4_3_SPLUS-s24s09_0028206',
 'iDR4_3_SPLUS-s25s10_0026796',
 'iDR4_3_SPLUS-s21s22_0024741',
 'iDR4_3_SPLUS-n15s05_0037835',
 'iDR4_3_SPLUS-n15s19_0049168',
 'iDR4_3_SPLUS-n15s20_0043589',
 'iDR4_3_SPLUS-n17s20_0011455',
 'iDR4_3_SPLUS-n17s21_0060665',
 'iDR4_3_SPLUS-n17s21_0053822',
 'iDR4_3_SPLUS-n16s21_0006990',
 'iDR4_3_SPLUS-n09s38_0057827',
 'iDR4_3_SPLUS-s20s23_0025387',
 'iDR4_3_SPLUS-s45s41_0027231',
 'iDR4_3_SPLUS-s24s09_0027219',
 'iDR4_3_SPLUS-s25s10_0025796',
 'iDR4_3_SPLUS-s20s23_0041616',
 'iDR4_3_SPLUS-s20s23_0029315',
 'iDR4_3_STRIPE82-0005_0021998',
 'iDR4_3_SPLUS-n02s38_0008843',
 'iDR4_3_SPLUS-n01s23_0026692',
 'iDR4_3_SPLUS-n01s37_0009556',
 'iDR4_3_STRIPE82-0105_0023811',
 'iDR4_3_STRIPE82-0059_0016073',
 'iDR4_3_STRIPE82-0104_0005639',
 'iDR4_3_STRIPE82-0014_0039180']

x_values = ["ISO","AUTO","PETRO"]
columns_filter = ['Object', 'u_iso', 'u_auto', 'u_petro', 'e_u_iso', 'e_u_auto', 'e_u_petro']
df_u = pd.DataFrame(columns = columns_filter)
table_to_skip = [1, 4, 5, 7, 8, 10, 14, 15, 21, 22, 23, 24, 26, 29, 31, 34, 38, 41, 42, 43, 45, 48, 49, 55, 57, 59, 61, 62, 63, 65, 68, 71, 73, 74, 80, 81, 83, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 96, 97, 99, 100, 107, 111, 112, 113, 115]

iso_mags=[]
iso_errors=[]
auto_mags=[]
auto_errors=[]
petro_mags=[]
petro_errors=[]

plt.figure(figsize=(10,6))
for table, obj in enumerate(objs_DT_ID):
    if table in table_to_skip:
        continue
    iso_row = iso_list[table][iso_list[table]['ID']==obj][0]
    auto_row = auto_list[table][auto_list[table]['ID']==obj][0]
    petro_row = petro_list[table][petro_list[table]['ID']==obj][0]
    
    mag_values=[]
    error_values=[]
    mag_values.append(iso_row[f'{filter_select}_iso'])
    error_values.append(iso_row[f'e_{filter_select}_iso'])
    mag_values.append(auto_row[f'{filter_select}_auto'])
    error_values.append(auto_row[f'e_{filter_select}_auto'])
    mag_values.append(petro_row[f'{filter_select}_petro'])
    error_values.append(petro_row[f'e_{filter_select}_petro'])
    
    iso_mags.append(iso_row[f'{filter_select}_iso'])
    auto_mags.append(auto_row[f'{filter_select}_auto'])
    petro_mags.append(petro_row[f'{filter_select}_petro'])
    iso_errors.append(iso_row[f'e_{filter_select}_iso'])
    auto_errors.append(auto_row[f'e_{filter_select}_auto'])
    petro_errors.append(petro_row[f'e_{filter_select}_petro'])

    df2 = pd.DataFrame([[loaded_gal_sample["Object"].iloc[table], mag_values[0], mag_values[1], mag_values[2], error_values[0], error_values[1], error_values[2]]],columns=columns_filter)
    df_u = pd.concat([df_u, df2],ignore_index=True)

iso_mean = sum(iso_mags) / len(iso_mags)
auto_mean = sum(auto_mags) / len(auto_mags)
petro_mean = sum(petro_mags) / len(petro_mags)

iso_error_mean = sum(iso_errors) / len(iso_errors)
auto_error_mean = sum(auto_errors) / len(auto_errors)
petro_error_mean = sum(petro_errors) / len(petro_errors)

plt.errorbar(['ISO'], [iso_mean], yerr=[iso_error_mean], fmt='o', capsize=5, color='green', label='ISO')
plt.errorbar(['AUTO'], [auto_mean], yerr=[auto_error_mean], fmt='o', capsize=5, color='red', label='AUTO')
plt.errorbar(['PETRO'], [petro_mean], yerr=[petro_error_mean], fmt='o', capsize=5, color='blue', label='PETRO')

plt.grid(linestyle='--', axis='y')
plt.ylabel("Mag (AB)")
plt.title("Filtro U")
plt.legend()
plt.show()

# %%

columns = ['u_iso', 'u_auto', 'u_petro', 'e_u_iso', 'e_u_auto', 'e_u_petro']
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df_u[columns].quantile(0.25)
Q3 = df_u[columns].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1
outliers = ((df_u[columns] < (Q1 - 1.5 * IQR)) | (df_u[columns] > (Q3 + 1.5 * IQR))).any(axis=1)
outlier_df_u = df_u[outliers]
df_u_without_outliers = df_u[~outliers]

# %%

outlier_df_u.boxplot()

# %%

df_u_without_outliers.boxplot()

# %%

df_u_without_outliers[['u_iso','u_auto','u_petro']].boxplot()

# %%

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_u[columns])
df_u['pca1'] = pca_result[:,0]
df_u['pca2'] = pca_result[:,1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca1', y='pca2', data=df_u, color='purple')
plt.title('Análise de componentes principais (PCA)')
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_u['cluster'] = kmeans.fit_predict(df_u[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='u_iso', y='e_u_iso', color='green', data=df_u)
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_u['cluster'] = kmeans.fit_predict(df_u[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='u_auto', y='e_u_auto', color='red', data=df_u)
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_u['cluster'] = kmeans.fit_predict(df_u[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='u_petro', y='e_u_petro', color='blue', data=df_u)
plt.show()

#%%

g = sns.JointGrid(x="u_iso", y="e_u_iso", data=df_u) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="u_auto", y="e_u_auto", data=df_u) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="u_petro", y="e_u_petro", data=df_u) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

x_label = range(len(df_u['Object']))
fig, ax = plt.subplots(figsize=(15, 24))

plt.errorbar(df_u['u_iso'], x_label, xerr=df_u['e_u_iso'], fmt='o', capsize=5, color='green')
plt.errorbar(df_u['u_auto'], x_label, xerr=df_u['e_u_auto'], fmt='o', capsize=5, color='red')
plt.errorbar(df_u['u_petro'], x_label, xerr=df_u['e_u_petro'], fmt='o', capsize=5, color='mediumblue')

plt.title("Filtro U", fontsize=25)
plt.legend(labels=['ISO', 'AUTO', 'PETRO'], fontsize=18, bbox_to_anchor=(1, 1))
plt.yticks(ticks=x_label, labels=df_u['Object'], rotation=0, ha='right', fontsize=20)
plt.xlabel('Mag (AB)', fontsize=20)
plt.ylabel('Galáxias', rotation=90, ha='right', fontsize=20)
plt.grid(axis='both')
plt.show()

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_u)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_u.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['u_auto'], row['u_iso'], yerr=row['e_u_iso'], xerr=row['e_u_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro U", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(13.5, 20, 0.25))
plt.yticks(np.arange(13.5, 20, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %% 

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_u)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_u.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['u_auto'], row['u_petro'], yerr=row['e_u_petro'], xerr=row['e_u_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro U", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('PETRO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(13.5, 20, 0.25))
plt.yticks(np.arange(13.5, 20, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_u)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_u.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['u_petro'], row['u_iso'], yerr=row['e_u_iso'], xerr=row['e_u_petro'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro U", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('PETRO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(13.5, 20, 0.25))
plt.yticks(np.arange(13.5, 20, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

filter_select='g'
objs_DT_ID=['iDR4_3_SPLUS-s38s28_0037355',
 'iDR4_3_SPLUS-s42s26_0009603',
 'iDR4_3_SPLUS-s37s33_0029178',
 'iDR4_3_SPLUS-n14s32_0029533',
 'iDR4_3_SPLUS-s19s23_0054919',
 'iDR4_3_SPLUS-s20s23_0027579',
 'iDR4_3_SPLUS-s27s07_0017147',
 'iDR4_3_SPLUS-s26s07_0019433',
 'iDR4_3_SPLUS-s21s10_0013331',
 'iDR4_3_SPLUS-s27s10_0013261',
 'iDR4_3_SPLUS-s24s11_0019810',
 'iDR4_3_SPLUS-s26s14_0016119',
 'iDR4_3_SPLUS-s25s18_0032854',
 'iDR4_3_SPLUS-s27s18_0020395',
 'iDR4_3_SPLUS-s27s19_0008651',
 'iDR4_3_SPLUS-s24s20_0018168',
 'iDR4_3_SPLUS-s25s21_0007448',
 'iDR4_3_SPLUS-s25s24_0039312',
 'iDR4_3_SPLUS-s37s20_0007226',
 'iDR4_3_SPLUS-s34s21_0008123',
 'iDR4_3_SPLUS-s24s26_0024907',
 'iDR4_3_SPLUS-s29s31_0019272',
 'iDR4_3_MC0103_0014596',
 'iDR4_3_SPLUS-s25s34_0007124',
 'iDR4_3_SPLUS-s24s35_0029130',
 'iDR4_3_SPLUS-s25s35_0013478',
 'iDR4_3_SPLUS-s30s34_0033827',
 'iDR4_3_SPLUS-s44s24_0025720',
 'iDR4_3_SPLUS-s28s36_0033655',
 'iDR4_3_SPLUS-s43s27_0018255',
 'iDR4_3_SPLUS-s41s30_0038650',
 'iDR4_3_SPLUS-s45s26_0000595',
 'iDR4_3_SPLUS-s38s32_0024249',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s30s39_0009952',
 'iDR4_3_SPLUS-s34s36_0027270',
 'iDR4_3_SPLUS-s37s34_0035918',
 'iDR4_3_SPLUS-s38s34_0035796',
 'iDR4_3_MC0001_0037018',
 'iDR4_3_SPLUS-s30s41_0001982',
 'iDR4_3_SPLUS-s25s45_0034698',
 'iDR4_3_MC0012_0039189',
 'iDR4_3_SPLUS-s39s34_0021319',
 'iDR4_3_SPLUS-s41s32_0034511',
 'iDR4_3_MC0021_0034624',
 'iDR4_3_SPLUS-s29s42_0019922',
 'iDR4_3_SPLUS-s41s33_0032981',
 'iDR4_3_MC0002_0006559',
 'iDR4_3_SPLUS-s36s39_0020464',
 'iDR4_3_SPLUS-n17s01_0006918',
 'iDR4_3_HYDRA-0104_0011186',
 'iDR4_3_HYDRA-0051_0052680',
 'iDR4_3_HYDRA-0015_0018824',
 'iDR4_3_HYDRA-0106_0049457',
 'iDR4_3_HYDRA-0028_0034942',
 'iDR4_3_HYDRA-0085_0056446',
 'iDR4_3_SPLUS-n20s03_0026145',
 'iDR4_3_SPLUS-n19s22_0026754',
 'iDR4_3_SPLUS-n20s23_0017293',
 'iDR4_3_SPLUS-s30s45_0010034',
 'iDR4_3_SPLUS-s20s23_0040961',
 'iDR4_3_SPLUS-s19s23_0047619',
 'iDR4_3_SPLUS-s23s26_0001674',
 'iDR4_3_SPLUS-s23s26_0041931',
 'iDR4_3_SPLUS-s24s65_0036645',
 'iDR4_3_SPLUS-s45s41_0034849',
 'iDR4_3_SPLUS-s23s46_0035512',
 'iDR4_3_SPLUS-s24s74_0016239',
 'iDR4_3_STRIPE82-0063_0034198',
 'iDR4_3_SPLUS-n13s31_0042282',
 'iDR4_3_SPLUS-n14s32_0041298',
 'iDR4_3_SPLUS-s39s31_0013592',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s21s23_0012649',
 'iDR4_3_HYDRA-0045_0074200',
 'iDR4_3_SPLUS-s34s39_0021318',
 'iDR4_3_SPLUS-s25s35_0015877',
 'iDR4_3_SPLUS-s25s13_0012636',
 'iDR4_3_MC0125_0008151',
 'iDR4_3_MC0097_0015643',
 'iDR4_3_SPLUS-s46s24_0023799',
 'iDR4_3_SPLUS-s46s24_0023407',
 'iDR4_3_SPLUS-s46s25_0034017',
 'iDR4_3_SPLUS-s42s30_0018253',
 'iDR4_3_SPLUS-s42s31_0030929',
 'iDR4_3_SPLUS-s42s32_0018085',
 'iDR4_3_SPLUS-s38s32_0022611',
 'iDR4_3_SPLUS-s38s33_0011808',
 'iDR4_3_SPLUS-s25s13_0017486',
 'iDR4_3_SPLUS-s29s42_0028164',
 'iDR4_3_SPLUS-s26s45_0031867',
 'iDR4_3_HYDRA-0107_0006292',
 'iDR4_3_SPLUS-s24s09_0028206',
 'iDR4_3_SPLUS-s25s10_0026796',
 'iDR4_3_SPLUS-s21s22_0024741',
 'iDR4_3_SPLUS-n15s05_0037835',
 'iDR4_3_SPLUS-n15s19_0049168',
 'iDR4_3_SPLUS-n15s20_0043589',
 'iDR4_3_SPLUS-n17s20_0011455',
 'iDR4_3_SPLUS-n17s21_0060665',
 'iDR4_3_SPLUS-n17s21_0053822',
 'iDR4_3_SPLUS-n16s21_0006990',
 'iDR4_3_SPLUS-n09s38_0057827',
 'iDR4_3_SPLUS-s20s23_0025387',
 'iDR4_3_SPLUS-s45s41_0027231',
 'iDR4_3_SPLUS-s24s09_0027219',
 'iDR4_3_SPLUS-s25s10_0025796',
 'iDR4_3_SPLUS-s20s23_0041616',
 'iDR4_3_SPLUS-s20s23_0029315',
 'iDR4_3_STRIPE82-0005_0021998',
 'iDR4_3_SPLUS-n02s38_0008843',
 'iDR4_3_SPLUS-n01s23_0026692',
 'iDR4_3_SPLUS-n01s37_0009556',
 'iDR4_3_STRIPE82-0105_0023811',
 'iDR4_3_STRIPE82-0059_0016073',
 'iDR4_3_STRIPE82-0104_0005639',
 'iDR4_3_STRIPE82-0014_0039180']

x_values = ["ISO","AUTO","PETRO"]
columns_filter = ['Object', 'g_iso', 'g_auto', 'g_petro', 'e_g_iso', 'e_g_auto', 'e_g_petro']
df_g = pd.DataFrame(columns = columns_filter)
table_to_skip = [1, 4, 5, 7, 8, 10, 14, 15, 21, 22, 23, 24, 26, 29, 31, 34, 38, 41, 42, 43, 45, 48, 49, 55, 57, 59, 61, 62, 63, 65, 68, 71, 73, 74, 80, 81, 83, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 96, 97, 99, 100, 107, 111, 112, 113, 115]

iso_mags=[]
iso_errors=[]
auto_mags=[]
auto_errors=[]
petro_mags=[]
petro_errors=[]

plt.figure(figsize=(10,6))
for table, obj in enumerate(objs_DT_ID):
    if table in table_to_skip:
        continue
    iso_row = iso_list[table][iso_list[table]['ID']==obj][0]
    auto_row = auto_list[table][auto_list[table]['ID']==obj][0]
    petro_row = petro_list[table][petro_list[table]['ID']==obj][0]
    
    mag_values=[]
    error_values=[]
    mag_values.append(iso_row[f'{filter_select}_iso'])
    error_values.append(iso_row[f'e_{filter_select}_iso'])
    mag_values.append(auto_row[f'{filter_select}_auto'])
    error_values.append(auto_row[f'e_{filter_select}_auto'])
    mag_values.append(petro_row[f'{filter_select}_petro'])
    error_values.append(petro_row[f'e_{filter_select}_petro'])
    
    iso_mags.append(iso_row[f'{filter_select}_iso'])
    auto_mags.append(auto_row[f'{filter_select}_auto'])
    petro_mags.append(petro_row[f'{filter_select}_petro'])
    iso_errors.append(iso_row[f'e_{filter_select}_iso'])
    auto_errors.append(auto_row[f'e_{filter_select}_auto'])
    petro_errors.append(petro_row[f'e_{filter_select}_petro'])

    df2 = pd.DataFrame([[loaded_gal_sample["Object"].iloc[table], mag_values[0], mag_values[1], mag_values[2], error_values[0], error_values[1], error_values[2]]],columns=columns_filter)
    df_g = pd.concat([df_g, df2],ignore_index=True)

iso_mean = sum(iso_mags) / len(iso_mags)
auto_mean = sum(auto_mags) / len(auto_mags)
petro_mean = sum(petro_mags) / len(petro_mags)

iso_error_mean = sum(iso_errors) / len(iso_errors)
auto_error_mean = sum(auto_errors) / len(auto_errors)
petro_error_mean = sum(petro_errors) / len(petro_errors)

plt.errorbar(['ISO'], [iso_mean], yerr=[iso_error_mean], fmt='o', capsize=5, color='green', label='ISO')
plt.errorbar(['AUTO'], [auto_mean], yerr=[auto_error_mean], fmt='o', capsize=5, color='red', label='AUTO')
plt.errorbar(['PETRO'], [petro_mean], yerr=[petro_error_mean], fmt='o', capsize=5, color='blue', label='PETRO')

plt.grid(linestyle='--', axis='y')
plt.ylabel("Mag (AB)")
plt.title("Filtro G")
plt.legend()
plt.show()

# %%

columns = ['g_iso', 'g_auto', 'g_petro', 'e_g_iso', 'e_g_auto', 'e_g_petro']
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df_g[columns].quantile(0.25)
Q3 = df_g[columns].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1
outliers = ((df_g[columns] < (Q1 - 1.5 * IQR)) | (df_g[columns] > (Q3 + 1.5 * IQR))).any(axis=1)
outlier_df_g = df_g[outliers]
df_g_without_outliers = df_g[~outliers]

# %%

outlier_df_g.boxplot()

# %%

df_g_without_outliers.boxplot()

# %%

df_g_without_outliers[['e_g_iso','e_g_auto','e_g_petro']].boxplot()

# %%

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_g[columns])
df_g['pca1'] = pca_result[:,0]
df_g['pca2'] = pca_result[:,1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca1', y='pca2', data=df_g, color='purple')
plt.title('Análise de componentes principais (PCA)')
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_g['cluster'] = kmeans.fit_predict(df_g[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='g_iso', y='e_g_iso', color='green', data=df_g)
plt.show()

#%%

kmeans = KMeans(n_clusters=3)
df_g['cluster'] = kmeans.fit_predict(df_g[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='g_auto', y='e_g_auto', color='red', data=df_g)
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_g['cluster'] = kmeans.fit_predict(df_g[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='g_petro', y='e_g_petro', color='blue', data=df_g)
plt.show()

#%%

g = sns.JointGrid(x="g_iso", y="e_g_iso", data=df_g) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="g_auto", y="e_g_auto", data=df_g) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="g_petro", y="e_g_petro", data=df_g) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

x_label = range(len(df_g))
fig, ax = plt.subplots(figsize=(24, 6))
trans1 = Affine2D().translate(-0.25, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.25, 0.0) + ax.transData

plt.errorbar(df_g['Object'], df_g['g_iso'], yerr=df_g['e_g_iso'], fmt='o', capsize=5, color='green', transform=trans1)
plt.errorbar(df_g['Object'], df_g['g_auto'], yerr=df_g['e_g_auto'], fmt='o', capsize=5, color='red')
plt.errorbar(df_g['Object'], df_g['g_petro'], yerr=df_g['e_g_petro'], fmt='o', capsize=5, color='mediumblue', transform=trans2)
plt.title("Filtro G")
plt.legend(labels=['ISO', 'AUTO', 'PETRO'])
plt.xticks(rotation=90, ha='right')
plt.ylabel('Mag (AB)')
plt.show

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_g)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_g.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['g_auto'], row['g_iso'], yerr=row['e_g_iso'], xerr=row['e_g_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro G", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(12, 18.25, 0.25))
plt.yticks(np.arange(12, 18.25, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %% 

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_g)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_g.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['g_auto'], row['g_petro'], yerr=row['e_g_petro'], xerr=row['e_g_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro G", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('PETRO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(12, 18.25, 0.25))
plt.yticks(np.arange(12, 18.25, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_g)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_g.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['g_petro'], row['g_iso'], yerr=row['e_g_iso'], xerr=row['e_g_petro'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro G", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('PETRO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(12, 18.25, 0.25))
plt.yticks(np.arange(12, 18.25, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

filter_select='r'
objs_DT_ID=['iDR4_3_SPLUS-s38s28_0037355',
 'iDR4_3_SPLUS-s42s26_0009603',
 'iDR4_3_SPLUS-s37s33_0029178',
 'iDR4_3_SPLUS-n14s32_0029533',
 'iDR4_3_SPLUS-s19s23_0054919',
 'iDR4_3_SPLUS-s20s23_0027579',
 'iDR4_3_SPLUS-s27s07_0017147',
 'iDR4_3_SPLUS-s26s07_0019433',
 'iDR4_3_SPLUS-s21s10_0013331',
 'iDR4_3_SPLUS-s27s10_0013261',
 'iDR4_3_SPLUS-s24s11_0019810',
 'iDR4_3_SPLUS-s26s14_0016119',
 'iDR4_3_SPLUS-s25s18_0032854',
 'iDR4_3_SPLUS-s27s18_0020395',
 'iDR4_3_SPLUS-s27s19_0008651',
 'iDR4_3_SPLUS-s24s20_0018168',
 'iDR4_3_SPLUS-s25s21_0007448',
 'iDR4_3_SPLUS-s25s24_0039312',
 'iDR4_3_SPLUS-s37s20_0007226',
 'iDR4_3_SPLUS-s34s21_0008123',
 'iDR4_3_SPLUS-s24s26_0024907',
 'iDR4_3_SPLUS-s29s31_0019272',
 'iDR4_3_MC0103_0014596',
 'iDR4_3_SPLUS-s25s34_0007124',
 'iDR4_3_SPLUS-s24s35_0029130',
 'iDR4_3_SPLUS-s25s35_0013478',
 'iDR4_3_SPLUS-s30s34_0033827',
 'iDR4_3_SPLUS-s44s24_0025720',
 'iDR4_3_SPLUS-s28s36_0033655',
 'iDR4_3_SPLUS-s43s27_0018255',
 'iDR4_3_SPLUS-s41s30_0038650',
 'iDR4_3_SPLUS-s45s26_0000595',
 'iDR4_3_SPLUS-s38s32_0024249',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s30s39_0009952',
 'iDR4_3_SPLUS-s34s36_0027270',
 'iDR4_3_SPLUS-s37s34_0035918',
 'iDR4_3_SPLUS-s38s34_0035796',
 'iDR4_3_MC0001_0037018',
 'iDR4_3_SPLUS-s30s41_0001982',
 'iDR4_3_SPLUS-s25s45_0034698',
 'iDR4_3_MC0012_0039189',
 'iDR4_3_SPLUS-s39s34_0021319',
 'iDR4_3_SPLUS-s41s32_0034511',
 'iDR4_3_MC0021_0034624',
 'iDR4_3_SPLUS-s29s42_0019922',
 'iDR4_3_SPLUS-s41s33_0032981',
 'iDR4_3_MC0002_0006559',
 'iDR4_3_SPLUS-s36s39_0020464',
 'iDR4_3_SPLUS-n17s01_0006918',
 'iDR4_3_HYDRA-0104_0011186',
 'iDR4_3_HYDRA-0051_0052680',
 'iDR4_3_HYDRA-0015_0018824',
 'iDR4_3_HYDRA-0106_0049457',
 'iDR4_3_HYDRA-0028_0034942',
 'iDR4_3_HYDRA-0085_0056446',
 'iDR4_3_SPLUS-n20s03_0026145',
 'iDR4_3_SPLUS-n19s22_0026754',
 'iDR4_3_SPLUS-n20s23_0017293',
 'iDR4_3_SPLUS-s30s45_0010034',
 'iDR4_3_SPLUS-s20s23_0040961',
 'iDR4_3_SPLUS-s19s23_0047619',
 'iDR4_3_SPLUS-s23s26_0001674',
 'iDR4_3_SPLUS-s23s26_0041931',
 'iDR4_3_SPLUS-s24s65_0036645',
 'iDR4_3_SPLUS-s45s41_0034849',
 'iDR4_3_SPLUS-s23s46_0035512',
 'iDR4_3_SPLUS-s24s74_0016239',
 'iDR4_3_STRIPE82-0063_0034198',
 'iDR4_3_SPLUS-n13s31_0042282',
 'iDR4_3_SPLUS-n14s32_0041298',
 'iDR4_3_SPLUS-s39s31_0013592',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s21s23_0012649',
 'iDR4_3_HYDRA-0045_0074200',
 'iDR4_3_SPLUS-s34s39_0021318',
 'iDR4_3_SPLUS-s25s35_0015877',
 'iDR4_3_SPLUS-s25s13_0012636',
 'iDR4_3_MC0125_0008151',
 'iDR4_3_MC0097_0015643',
 'iDR4_3_SPLUS-s46s24_0023799',
 'iDR4_3_SPLUS-s46s24_0023407',
 'iDR4_3_SPLUS-s46s25_0034017',
 'iDR4_3_SPLUS-s42s30_0018253',
 'iDR4_3_SPLUS-s42s31_0030929',
 'iDR4_3_SPLUS-s42s32_0018085',
 'iDR4_3_SPLUS-s38s32_0022611',
 'iDR4_3_SPLUS-s38s33_0011808',
 'iDR4_3_SPLUS-s25s13_0017486',
 'iDR4_3_SPLUS-s29s42_0028164',
 'iDR4_3_SPLUS-s26s45_0031867',
 'iDR4_3_HYDRA-0107_0006292',
 'iDR4_3_SPLUS-s24s09_0028206',
 'iDR4_3_SPLUS-s25s10_0026796',
 'iDR4_3_SPLUS-s21s22_0024741',
 'iDR4_3_SPLUS-n15s05_0037835',
 'iDR4_3_SPLUS-n15s19_0049168',
 'iDR4_3_SPLUS-n15s20_0043589',
 'iDR4_3_SPLUS-n17s20_0011455',
 'iDR4_3_SPLUS-n17s21_0060665',
 'iDR4_3_SPLUS-n17s21_0053822',
 'iDR4_3_SPLUS-n16s21_0006990',
 'iDR4_3_SPLUS-n09s38_0057827',
 'iDR4_3_SPLUS-s20s23_0025387',
 'iDR4_3_SPLUS-s45s41_0027231',
 'iDR4_3_SPLUS-s24s09_0027219',
 'iDR4_3_SPLUS-s25s10_0025796',
 'iDR4_3_SPLUS-s20s23_0041616',
 'iDR4_3_SPLUS-s20s23_0029315',
 'iDR4_3_STRIPE82-0005_0021998',
 'iDR4_3_SPLUS-n02s38_0008843',
 'iDR4_3_SPLUS-n01s23_0026692',
 'iDR4_3_SPLUS-n01s37_0009556',
 'iDR4_3_STRIPE82-0105_0023811',
 'iDR4_3_STRIPE82-0059_0016073',
 'iDR4_3_STRIPE82-0104_0005639',
 'iDR4_3_STRIPE82-0014_0039180']

x_values = ["ISO","AUTO","PETRO"]
columns_filter = ['Object', 'r_iso', 'r_auto', 'r_petro', 'e_r_iso', 'e_r_auto', 'e_r_petro']
df_r = pd.DataFrame(columns = columns_filter)
table_to_skip = [1, 4, 5, 7, 8, 10, 14, 15, 21, 22, 23, 24, 26, 29, 31, 34, 38, 41, 42, 43, 45, 48, 49, 55, 57, 59, 61, 62, 63, 65, 68, 71, 73, 74, 80, 81, 83, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 96, 97, 99, 100, 107, 111, 112, 115]

iso_mags=[]
iso_errors=[]
auto_mags=[]
auto_errors=[]
petro_mags=[]
petro_errors=[]

plt.figure(figsize=(10,6))
for table, obj in enumerate(objs_DT_ID):
    if table in table_to_skip:
        continue
    iso_row = iso_list[table][iso_list[table]['ID']==obj][0]
    auto_row = auto_list[table][auto_list[table]['ID']==obj][0]
    petro_row = petro_list[table][petro_list[table]['ID']==obj][0]
    
    mag_values=[]
    error_values=[]
    mag_values.append(iso_row[f'{filter_select}_iso'])
    error_values.append(iso_row[f'e_{filter_select}_iso'])
    mag_values.append(auto_row[f'{filter_select}_auto'])
    error_values.append(auto_row[f'e_{filter_select}_auto'])
    mag_values.append(petro_row[f'{filter_select}_petro'])
    error_values.append(petro_row[f'e_{filter_select}_petro'])
    
    iso_mags.append(iso_row[f'{filter_select}_iso'])
    auto_mags.append(auto_row[f'{filter_select}_auto'])
    petro_mags.append(petro_row[f'{filter_select}_petro'])
    iso_errors.append(iso_row[f'e_{filter_select}_iso'])
    auto_errors.append(auto_row[f'e_{filter_select}_auto'])
    petro_errors.append(petro_row[f'e_{filter_select}_petro'])

    df2 = pd.DataFrame([[loaded_gal_sample["Object"].iloc[table], mag_values[0], mag_values[1], mag_values[2], error_values[0], error_values[1], error_values[2]]],columns=columns_filter)
    df_r = pd.concat([df_r, df2],ignore_index=True)

iso_mean = sum(iso_mags) / len(iso_mags)
auto_mean = sum(auto_mags) / len(auto_mags)
petro_mean = sum(petro_mags) / len(petro_mags)

iso_error_mean = sum(iso_errors) / len(iso_errors)
auto_error_mean = sum(auto_errors) / len(auto_errors)
petro_error_mean = sum(petro_errors) / len(petro_errors)

plt.errorbar(['ISO'], [iso_mean], yerr=[iso_error_mean], fmt='o', capsize=5, color='green', label='ISO')
plt.errorbar(['AUTO'], [auto_mean], yerr=[auto_error_mean], fmt='o', capsize=5, color='red', label='AUTO')
plt.errorbar(['PETRO'], [petro_mean], yerr=[petro_error_mean], fmt='o', capsize=5, color='blue', label='PETRO')

plt.grid(linestyle='--', axis='y')
plt.ylabel("Mag (AB)")
plt.title("Filtro R")
plt.legend()
plt.show()

# %%

columns = ['r_iso', 'r_auto', 'r_petro', 'e_r_iso', 'e_r_auto', 'e_r_petro']
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df_r[columns].quantile(0.25)
Q3 = df_r[columns].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1
outliers = ((df_r[columns] < (Q1 - 1.5 * IQR)) | (df_r[columns] > (Q3 + 1.5 * IQR))).any(axis=1)
outlier_df_r = df_r[outliers]
df_r_without_outliers = df_r[~outliers]

# %%

outlier_df_r.boxplot()

# %%

df_r_without_outliers.boxplot()

# %%

df_r_without_outliers[['e_r_iso','e_r_auto','e_r_petro']].boxplot()

# %%

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_r[columns])
df_r['pca1'] = pca_result[:,0]
df_r['pca2'] = pca_result[:,1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca1', y='pca2', data=df_r, color='purple')
plt.title('Análise de componentes principais (PCA)')
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_r['cluster'] = kmeans.fit_predict(df_r[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='r_iso', y='e_r_iso', color='green', data=df_r)
plt.show()

#%%

kmeans = KMeans(n_clusters=3)
df_r['cluster'] = kmeans.fit_predict(df_r[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='r_auto', y='e_r_auto', color='red', data=df_r)
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_r['cluster'] = kmeans.fit_predict(df_r[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='r_petro', y='e_r_petro', color='blue', data=df_r)
plt.show()

#%%

g = sns.JointGrid(x="r_iso", y="e_r_iso", data=df_r) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="r_auto", y="e_r_auto", data=df_r) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="r_petro", y="e_r_petro", data=df_r) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

x_label = range(len(df_r))
fig, ax = plt.subplots(figsize=(24, 6))
trans1 = Affine2D().translate(-0.25, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.25, 0.0) + ax.transData

plt.errorbar(df_r['Object'], df_r['r_iso'], yerr=df_r['e_r_iso'], fmt='o', capsize=5, color='green', transform=trans1)
plt.errorbar(df_r['Object'], df_r['r_auto'], yerr=df_r['e_r_auto'], fmt='o', capsize=5, color='red')
plt.errorbar(df_r['Object'], df_r['r_petro'], yerr=df_r['e_r_petro'], fmt='o', capsize=5, color='mediumblue', transform=trans2)
plt.title("Filtro R")
plt.legend(labels=['ISO', 'AUTO', 'PETRO'])
plt.xticks(rotation=90, ha='right')
plt.ylabel('Mag (AB)')
plt.show

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_r)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_r.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['r_auto'], row['r_iso'], yerr=row['e_r_iso'], xerr=row['e_r_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro R", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(11, 18, 0.25))
plt.yticks(np.arange(11, 18, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %% 

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_r)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_r.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['r_auto'], row['r_petro'], yerr=row['e_r_petro'], xerr=row['e_r_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro R", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('PETRO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(11, 18, 0.25))
plt.yticks(np.arange(11, 18, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_r)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_r.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['r_petro'], row['r_iso'], yerr=row['e_r_iso'], xerr=row['e_r_petro'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro R", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('PETRO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(11, 18, 0.25))
plt.yticks(np.arange(11, 18, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

filter_select='i'
objs_DT_ID=['iDR4_3_SPLUS-s38s28_0037355',
 'iDR4_3_SPLUS-s42s26_0009603',
 'iDR4_3_SPLUS-s37s33_0029178',
 'iDR4_3_SPLUS-n14s32_0029533',
 'iDR4_3_SPLUS-s19s23_0054919',
 'iDR4_3_SPLUS-s20s23_0027579',
 'iDR4_3_SPLUS-s27s07_0017147',
 'iDR4_3_SPLUS-s26s07_0019433',
 'iDR4_3_SPLUS-s21s10_0013331',
 'iDR4_3_SPLUS-s27s10_0013261',
 'iDR4_3_SPLUS-s24s11_0019810',
 'iDR4_3_SPLUS-s26s14_0016119',
 'iDR4_3_SPLUS-s25s18_0032854',
 'iDR4_3_SPLUS-s27s18_0020395',
 'iDR4_3_SPLUS-s27s19_0008651',
 'iDR4_3_SPLUS-s24s20_0018168',
 'iDR4_3_SPLUS-s25s21_0007448',
 'iDR4_3_SPLUS-s25s24_0039312',
 'iDR4_3_SPLUS-s37s20_0007226',
 'iDR4_3_SPLUS-s34s21_0008123',
 'iDR4_3_SPLUS-s24s26_0024907',
 'iDR4_3_SPLUS-s29s31_0019272',
 'iDR4_3_MC0103_0014596',
 'iDR4_3_SPLUS-s25s34_0007124',
 'iDR4_3_SPLUS-s24s35_0029130',
 'iDR4_3_SPLUS-s25s35_0013478',
 'iDR4_3_SPLUS-s30s34_0033827',
 'iDR4_3_SPLUS-s44s24_0025720',
 'iDR4_3_SPLUS-s28s36_0033655',
 'iDR4_3_SPLUS-s43s27_0018255',
 'iDR4_3_SPLUS-s41s30_0038650',
 'iDR4_3_SPLUS-s45s26_0000595',
 'iDR4_3_SPLUS-s38s32_0024249',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s30s39_0009952',
 'iDR4_3_SPLUS-s34s36_0027270',
 'iDR4_3_SPLUS-s37s34_0035918',
 'iDR4_3_SPLUS-s38s34_0035796',
 'iDR4_3_MC0001_0037018',
 'iDR4_3_SPLUS-s30s41_0001982',
 'iDR4_3_SPLUS-s25s45_0034698',
 'iDR4_3_MC0012_0039189',
 'iDR4_3_SPLUS-s39s34_0021319',
 'iDR4_3_SPLUS-s41s32_0034511',
 'iDR4_3_MC0021_0034624',
 'iDR4_3_SPLUS-s29s42_0019922',
 'iDR4_3_SPLUS-s41s33_0032981',
 'iDR4_3_MC0002_0006559',
 'iDR4_3_SPLUS-s36s39_0020464',
 'iDR4_3_SPLUS-n17s01_0006918',
 'iDR4_3_HYDRA-0104_0011186',
 'iDR4_3_HYDRA-0051_0052680',
 'iDR4_3_HYDRA-0015_0018824',
 'iDR4_3_HYDRA-0106_0049457',
 'iDR4_3_HYDRA-0028_0034942',
 'iDR4_3_HYDRA-0085_0056446',
 'iDR4_3_SPLUS-n20s03_0026145',
 'iDR4_3_SPLUS-n19s22_0026754',
 'iDR4_3_SPLUS-n20s23_0017293',
 'iDR4_3_SPLUS-s30s45_0010034',
 'iDR4_3_SPLUS-s20s23_0040961',
 'iDR4_3_SPLUS-s19s23_0047619',
 'iDR4_3_SPLUS-s23s26_0001674',
 'iDR4_3_SPLUS-s23s26_0041931',
 'iDR4_3_SPLUS-s24s65_0036645',
 'iDR4_3_SPLUS-s45s41_0034849',
 'iDR4_3_SPLUS-s23s46_0035512',
 'iDR4_3_SPLUS-s24s74_0016239',
 'iDR4_3_STRIPE82-0063_0034198',
 'iDR4_3_SPLUS-n13s31_0042282',
 'iDR4_3_SPLUS-n14s32_0041298',
 'iDR4_3_SPLUS-s39s31_0013592',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s21s23_0012649',
 'iDR4_3_HYDRA-0045_0074200',
 'iDR4_3_SPLUS-s34s39_0021318',
 'iDR4_3_SPLUS-s25s35_0015877',
 'iDR4_3_SPLUS-s25s13_0012636',
 'iDR4_3_MC0125_0008151',
 'iDR4_3_MC0097_0015643',
 'iDR4_3_SPLUS-s46s24_0023799',
 'iDR4_3_SPLUS-s46s24_0023407',
 'iDR4_3_SPLUS-s46s25_0034017',
 'iDR4_3_SPLUS-s42s30_0018253',
 'iDR4_3_SPLUS-s42s31_0030929',
 'iDR4_3_SPLUS-s42s32_0018085',
 'iDR4_3_SPLUS-s38s32_0022611',
 'iDR4_3_SPLUS-s38s33_0011808',
 'iDR4_3_SPLUS-s25s13_0017486',
 'iDR4_3_SPLUS-s29s42_0028164',
 'iDR4_3_SPLUS-s26s45_0031867',
 'iDR4_3_HYDRA-0107_0006292',
 'iDR4_3_SPLUS-s24s09_0028206',
 'iDR4_3_SPLUS-s25s10_0026796',
 'iDR4_3_SPLUS-s21s22_0024741',
 'iDR4_3_SPLUS-n15s05_0037835',
 'iDR4_3_SPLUS-n15s19_0049168',
 'iDR4_3_SPLUS-n15s20_0043589',
 'iDR4_3_SPLUS-n17s20_0011455',
 'iDR4_3_SPLUS-n17s21_0060665',
 'iDR4_3_SPLUS-n17s21_0053822',
 'iDR4_3_SPLUS-n16s21_0006990',
 'iDR4_3_SPLUS-n09s38_0057827',
 'iDR4_3_SPLUS-s20s23_0025387',
 'iDR4_3_SPLUS-s45s41_0027231',
 'iDR4_3_SPLUS-s24s09_0027219',
 'iDR4_3_SPLUS-s25s10_0025796',
 'iDR4_3_SPLUS-s20s23_0041616',
 'iDR4_3_SPLUS-s20s23_0029315',
 'iDR4_3_STRIPE82-0005_0021998',
 'iDR4_3_SPLUS-n02s38_0008843',
 'iDR4_3_SPLUS-n01s23_0026692',
 'iDR4_3_SPLUS-n01s37_0009556',
 'iDR4_3_STRIPE82-0105_0023811',
 'iDR4_3_STRIPE82-0059_0016073',
 'iDR4_3_STRIPE82-0104_0005639',
 'iDR4_3_STRIPE82-0014_0039180']

x_values = ["ISO","AUTO","PETRO"]
columns_filter = ['Object', 'i_iso', 'i_auto', 'i_petro', 'e_i_iso', 'e_i_auto', 'e_i_petro']
df_i = pd.DataFrame(columns = columns_filter)
table_to_skip = [1, 4, 5, 7, 8, 10, 14, 15, 21, 22, 23, 24, 26, 29, 31, 34, 38, 41, 42, 43, 45, 48, 49, 55, 57, 59, 61, 62, 63, 65, 68, 71, 73, 74, 75, 80, 81, 83, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 96, 97, 99, 100, 107, 111, 112, 115]

iso_mags=[]
iso_errors=[]
auto_mags=[]
auto_errors=[]
petro_mags=[]
petro_errors=[]

plt.figure(figsize=(10,6))
for table, obj in enumerate(objs_DT_ID):
    if table in table_to_skip:
        continue
    iso_row = iso_list[table][iso_list[table]['ID']==obj][0]
    auto_row = auto_list[table][auto_list[table]['ID']==obj][0]
    petro_row = petro_list[table][petro_list[table]['ID']==obj][0]
    
    mag_values=[]
    error_values=[]
    mag_values.append(iso_row[f'{filter_select}_iso'])
    error_values.append(iso_row[f'e_{filter_select}_iso'])
    mag_values.append(auto_row[f'{filter_select}_auto'])
    error_values.append(auto_row[f'e_{filter_select}_auto'])
    mag_values.append(petro_row[f'{filter_select}_petro'])
    error_values.append(petro_row[f'e_{filter_select}_petro'])
    
    iso_mags.append(iso_row[f'{filter_select}_iso'])
    auto_mags.append(auto_row[f'{filter_select}_auto'])
    petro_mags.append(petro_row[f'{filter_select}_petro'])
    iso_errors.append(iso_row[f'e_{filter_select}_iso'])
    auto_errors.append(auto_row[f'e_{filter_select}_auto'])
    petro_errors.append(petro_row[f'e_{filter_select}_petro'])

    df2 = pd.DataFrame([[loaded_gal_sample["Object"].iloc[table], mag_values[0], mag_values[1], mag_values[2], error_values[0], error_values[1], error_values[2]]],columns=columns_filter)
    df_i = pd.concat([df_i, df2],ignore_index=True)

iso_mean = sum(iso_mags) / len(iso_mags)
auto_mean = sum(auto_mags) / len(auto_mags)
petro_mean = sum(petro_mags) / len(petro_mags)

iso_error_mean = sum(iso_errors) / len(iso_errors)
auto_error_mean = sum(auto_errors) / len(auto_errors)
petro_error_mean = sum(petro_errors) / len(petro_errors)

plt.errorbar(['ISO'], [iso_mean], yerr=[iso_error_mean], fmt='o', capsize=5, color='green', label='ISO')
plt.errorbar(['AUTO'], [auto_mean], yerr=[auto_error_mean], fmt='o', capsize=5, color='red', label='AUTO')
plt.errorbar(['PETRO'], [petro_mean], yerr=[petro_error_mean], fmt='o', capsize=5, color='blue', label='PETRO')

plt.grid(linestyle='--', axis='y')
plt.ylabel("Mag (AB)")
plt.title("Filtro I")
plt.legend()
plt.show()

# %%

columns = ['i_iso', 'i_auto', 'i_petro', 'e_i_iso', 'e_i_auto', 'e_i_petro']
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df_i[columns].quantile(0.25)
Q3 = df_i[columns].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1
outliers = ((df_i[columns] < (Q1 - 1.5 * IQR)) | (df_i[columns] > (Q3 + 1.5 * IQR))).any(axis=1)
outlier_df_i = df_i[outliers]
df_i_without_outliers = df_i[~outliers]

# %%

outlier_df_i.boxplot()

# %%

df_i_without_outliers.boxplot()

# %%

df_i_without_outliers[['e_i_iso','e_i_auto','e_i_petro']].boxplot()

# %%

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_i[columns])
df_i['pca1'] = pca_result[:,0]
df_i['pca2'] = pca_result[:,1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca1', y='pca2', data=df_i, color='purple')
plt.title('Análise de componentes principais (PCA)')
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_i['cluster'] = kmeans.fit_predict(df_i[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='i_iso', y='e_i_iso', color='green', data=df_i)
plt.show()

#%%

kmeans = KMeans(n_clusters=3)
df_i['cluster'] = kmeans.fit_predict(df_i[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='i_auto', y='e_i_auto', color='red', data=df_i)
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_i['cluster'] = kmeans.fit_predict(df_i[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='i_petro', y='e_i_petro', color='blue', data=df_i)
plt.show()

#%%

g = sns.JointGrid(x="i_iso", y="e_i_iso", data=df_i) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="i_auto", y="e_i_auto", data=df_i) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="i_petro", y="e_i_petro", data=df_i) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

x_label = range(len(df_i))
fig, ax = plt.subplots(figsize=(24, 6))
trans1 = Affine2D().translate(-0.25, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.25, 0.0) + ax.transData

plt.errorbar(df_i['Object'], df_i['i_iso'], yerr=df_i['e_i_iso'], fmt='o', capsize=5, color='green', transform=trans1)
plt.errorbar(df_i['Object'], df_i['i_auto'], yerr=df_i['e_i_auto'], fmt='o', capsize=5, color='red')
plt.errorbar(df_i['Object'], df_i['i_petro'], yerr=df_i['e_i_petro'], fmt='o', capsize=5, color='mediumblue', transform=trans2)
plt.title("Filtro I")
plt.legend(labels=['ISO', 'AUTO', 'PETRO'])
plt.xticks(rotation=90, ha='right')
plt.ylabel('Mag (AB)')
plt.show

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_i)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_i.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['i_auto'], row['i_iso'], yerr=row['e_i_iso'], xerr=row['e_i_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro I", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(10.5, 17.5, 0.25))
plt.yticks(np.arange(10.5, 17.5, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %% 

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_i)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_i.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['i_auto'], row['i_petro'], yerr=row['e_i_petro'], xerr=row['e_i_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro I", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('PETRO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(10.5, 17.5, 0.25))
plt.yticks(np.arange(10.5, 17.5, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_i)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_i.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['i_petro'], row['i_iso'], yerr=row['e_i_iso'], xerr=row['e_i_petro'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro I", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('PETRO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(10.5, 17.5, 0.25))
plt.yticks(np.arange(10.5, 17.5, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

filter_select='z'
objs_DT_ID=['iDR4_3_SPLUS-s38s28_0037355',
 'iDR4_3_SPLUS-s42s26_0009603',
 'iDR4_3_SPLUS-s37s33_0029178',
 'iDR4_3_SPLUS-n14s32_0029533',
 'iDR4_3_SPLUS-s19s23_0054919',
 'iDR4_3_SPLUS-s20s23_0027579',
 'iDR4_3_SPLUS-s27s07_0017147',
 'iDR4_3_SPLUS-s26s07_0019433',
 'iDR4_3_SPLUS-s21s10_0013331',
 'iDR4_3_SPLUS-s27s10_0013261',
 'iDR4_3_SPLUS-s24s11_0019810',
 'iDR4_3_SPLUS-s26s14_0016119',
 'iDR4_3_SPLUS-s25s18_0032854',
 'iDR4_3_SPLUS-s27s18_0020395',
 'iDR4_3_SPLUS-s27s19_0008651',
 'iDR4_3_SPLUS-s24s20_0018168',
 'iDR4_3_SPLUS-s25s21_0007448',
 'iDR4_3_SPLUS-s25s24_0039312',
 'iDR4_3_SPLUS-s37s20_0007226',
 'iDR4_3_SPLUS-s34s21_0008123',
 'iDR4_3_SPLUS-s24s26_0024907',
 'iDR4_3_SPLUS-s29s31_0019272',
 'iDR4_3_MC0103_0014596',
 'iDR4_3_SPLUS-s25s34_0007124',
 'iDR4_3_SPLUS-s24s35_0029130',
 'iDR4_3_SPLUS-s25s35_0013478',
 'iDR4_3_SPLUS-s30s34_0033827',
 'iDR4_3_SPLUS-s44s24_0025720',
 'iDR4_3_SPLUS-s28s36_0033655',
 'iDR4_3_SPLUS-s43s27_0018255',
 'iDR4_3_SPLUS-s41s30_0038650',
 'iDR4_3_SPLUS-s45s26_0000595',
 'iDR4_3_SPLUS-s38s32_0024249',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s30s39_0009952',
 'iDR4_3_SPLUS-s34s36_0027270',
 'iDR4_3_SPLUS-s37s34_0035918',
 'iDR4_3_SPLUS-s38s34_0035796',
 'iDR4_3_MC0001_0037018',
 'iDR4_3_SPLUS-s30s41_0001982',
 'iDR4_3_SPLUS-s25s45_0034698',
 'iDR4_3_MC0012_0039189',
 'iDR4_3_SPLUS-s39s34_0021319',
 'iDR4_3_SPLUS-s41s32_0034511',
 'iDR4_3_MC0021_0034624',
 'iDR4_3_SPLUS-s29s42_0019922',
 'iDR4_3_SPLUS-s41s33_0032981',
 'iDR4_3_MC0002_0006559',
 'iDR4_3_SPLUS-s36s39_0020464',
 'iDR4_3_SPLUS-n17s01_0006918',
 'iDR4_3_HYDRA-0104_0011186',
 'iDR4_3_HYDRA-0051_0052680',
 'iDR4_3_HYDRA-0015_0018824',
 'iDR4_3_HYDRA-0106_0049457',
 'iDR4_3_HYDRA-0028_0034942',
 'iDR4_3_HYDRA-0085_0056446',
 'iDR4_3_SPLUS-n20s03_0026145',
 'iDR4_3_SPLUS-n19s22_0026754',
 'iDR4_3_SPLUS-n20s23_0017293',
 'iDR4_3_SPLUS-s30s45_0010034',
 'iDR4_3_SPLUS-s20s23_0040961',
 'iDR4_3_SPLUS-s19s23_0047619',
 'iDR4_3_SPLUS-s23s26_0001674',
 'iDR4_3_SPLUS-s23s26_0041931',
 'iDR4_3_SPLUS-s24s65_0036645',
 'iDR4_3_SPLUS-s45s41_0034849',
 'iDR4_3_SPLUS-s23s46_0035512',
 'iDR4_3_SPLUS-s24s74_0016239',
 'iDR4_3_STRIPE82-0063_0034198',
 'iDR4_3_SPLUS-n13s31_0042282',
 'iDR4_3_SPLUS-n14s32_0041298',
 'iDR4_3_SPLUS-s39s31_0013592',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s21s23_0012649',
 'iDR4_3_HYDRA-0045_0074200',
 'iDR4_3_SPLUS-s34s39_0021318',
 'iDR4_3_SPLUS-s25s35_0015877',
 'iDR4_3_SPLUS-s25s13_0012636',
 'iDR4_3_MC0125_0008151',
 'iDR4_3_MC0097_0015643',
 'iDR4_3_SPLUS-s46s24_0023799',
 'iDR4_3_SPLUS-s46s24_0023407',
 'iDR4_3_SPLUS-s46s25_0034017',
 'iDR4_3_SPLUS-s42s30_0018253',
 'iDR4_3_SPLUS-s42s31_0030929',
 'iDR4_3_SPLUS-s42s32_0018085',
 'iDR4_3_SPLUS-s38s32_0022611',
 'iDR4_3_SPLUS-s38s33_0011808',
 'iDR4_3_SPLUS-s25s13_0017486',
 'iDR4_3_SPLUS-s29s42_0028164',
 'iDR4_3_SPLUS-s26s45_0031867',
 'iDR4_3_HYDRA-0107_0006292',
 'iDR4_3_SPLUS-s24s09_0028206',
 'iDR4_3_SPLUS-s25s10_0026796',
 'iDR4_3_SPLUS-s21s22_0024741',
 'iDR4_3_SPLUS-n15s05_0037835',
 'iDR4_3_SPLUS-n15s19_0049168',
 'iDR4_3_SPLUS-n15s20_0043589',
 'iDR4_3_SPLUS-n17s20_0011455',
 'iDR4_3_SPLUS-n17s21_0060665',
 'iDR4_3_SPLUS-n17s21_0053822',
 'iDR4_3_SPLUS-n16s21_0006990',
 'iDR4_3_SPLUS-n09s38_0057827',
 'iDR4_3_SPLUS-s20s23_0025387',
 'iDR4_3_SPLUS-s45s41_0027231',
 'iDR4_3_SPLUS-s24s09_0027219',
 'iDR4_3_SPLUS-s25s10_0025796',
 'iDR4_3_SPLUS-s20s23_0041616',
 'iDR4_3_SPLUS-s20s23_0029315',
 'iDR4_3_STRIPE82-0005_0021998',
 'iDR4_3_SPLUS-n02s38_0008843',
 'iDR4_3_SPLUS-n01s23_0026692',
 'iDR4_3_SPLUS-n01s37_0009556',
 'iDR4_3_STRIPE82-0105_0023811',
 'iDR4_3_STRIPE82-0059_0016073',
 'iDR4_3_STRIPE82-0104_0005639',
 'iDR4_3_STRIPE82-0014_0039180']

x_values = ["ISO","AUTO","PETRO"]
columns_filter = ['Object', 'z_iso', 'z_auto', 'z_petro', 'e_z_iso', 'e_z_auto', 'e_z_petro']
df_z = pd.DataFrame(columns = columns_filter)
table_to_skip = [1, 4, 5, 7, 8, 10, 14, 15, 21, 22, 23, 24, 26, 29, 31, 34, 38, 41, 42, 43, 45, 48, 49, 55, 57, 59, 61, 62, 63, 65, 68, 71, 73, 74, 75, 80, 81, 83, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 96, 97, 99, 100, 107, 111, 112, 115]

iso_mags=[]
iso_errors=[]
auto_mags=[]
auto_errors=[]
petro_mags=[]
petro_errors=[]

plt.figure(figsize=(10,6))
for table, obj in enumerate(objs_DT_ID):
    if table in table_to_skip:
        continue
    iso_row = iso_list[table][iso_list[table]['ID']==obj][0]
    auto_row = auto_list[table][auto_list[table]['ID']==obj][0]
    petro_row = petro_list[table][petro_list[table]['ID']==obj][0]
    
    mag_values=[]
    error_values=[]
    mag_values.append(iso_row[f'{filter_select}_iso'])
    error_values.append(iso_row[f'e_{filter_select}_iso'])
    mag_values.append(auto_row[f'{filter_select}_auto'])
    error_values.append(auto_row[f'e_{filter_select}_auto'])
    mag_values.append(petro_row[f'{filter_select}_petro'])
    error_values.append(petro_row[f'e_{filter_select}_petro'])
    
    iso_mags.append(iso_row[f'{filter_select}_iso'])
    auto_mags.append(auto_row[f'{filter_select}_auto'])
    petro_mags.append(petro_row[f'{filter_select}_petro'])
    iso_errors.append(iso_row[f'e_{filter_select}_iso'])
    auto_errors.append(auto_row[f'e_{filter_select}_auto'])
    petro_errors.append(petro_row[f'e_{filter_select}_petro'])

    df2 = pd.DataFrame([[loaded_gal_sample["Object"].iloc[table], mag_values[0], mag_values[1], mag_values[2], error_values[0], error_values[1], error_values[2]]],columns=columns_filter)
    df_z = pd.concat([df_z, df2],ignore_index=True)

iso_mean = sum(iso_mags) / len(iso_mags)
auto_mean = sum(auto_mags) / len(auto_mags)
petro_mean = sum(petro_mags) / len(petro_mags)

iso_error_mean = sum(iso_errors) / len(iso_errors)
auto_error_mean = sum(auto_errors) / len(auto_errors)
petro_error_mean = sum(petro_errors) / len(petro_errors)

plt.errorbar(['ISO'], [iso_mean], yerr=[iso_error_mean], fmt='o', capsize=5, color='green', label='ISO')
plt.errorbar(['AUTO'], [auto_mean], yerr=[auto_error_mean], fmt='o', capsize=5, color='red', label='AUTO')
plt.errorbar(['PETRO'], [petro_mean], yerr=[petro_error_mean], fmt='o', capsize=5, color='blue', label='PETRO')

plt.grid(linestyle='--', axis='y')
plt.ylabel("Mag (AB)")
plt.title("Filtro Z")
plt.legend()
plt.show()

# %%

columns = ['z_iso', 'z_auto', 'z_petro', 'e_z_iso', 'e_z_auto', 'e_z_petro']
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df_z[columns].quantile(0.25)
Q3 = df_z[columns].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1
outliers = ((df_z[columns] < (Q1 - 1.5 * IQR)) | (df_z[columns] > (Q3 + 1.5 * IQR))).any(axis=1)
outlier_df_z = df_z[outliers]
df_z_without_outliers = df_z[~outliers]

# %%

outlier_df_z.boxplot()

# %%

df_z_without_outliers.boxplot()

# %%

df_z_without_outliers[['e_z_iso','e_z_auto','e_z_petro']].boxplot()

# %%

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_z[columns])
df_z['pca1'] = pca_result[:,0]
df_z['pca2'] = pca_result[:,1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca1', y='pca2', data=df_z, color='purple')
plt.title('Análise de componentes principais (PCA)')
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_z['cluster'] = kmeans.fit_predict(df_z[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='z_iso', y='e_z_iso', color='green', data=df_z)
plt.show()

#%%

kmeans = KMeans(n_clusters=3)
df_z['cluster'] = kmeans.fit_predict(df_z[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='z_auto', y='e_z_auto', color='red', data=df_z)
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_z['cluster'] = kmeans.fit_predict(df_z[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='z_petro', y='e_z_petro', color='blue', data=df_z)
plt.show()

#%%

g = sns.JointGrid(x="z_iso", y="e_z_iso", data=df_z) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="z_auto", y="e_z_auto", data=df_z) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="z_petro", y="e_z_petro", data=df_z) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

x_label = range(len(df_z))
fig, ax = plt.subplots(figsize=(24, 6))
trans1 = Affine2D().translate(-0.25, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.25, 0.0) + ax.transData

plt.errorbar(df_z['Object'], df_z['z_iso'], yerr=df_z['e_z_iso'], fmt='o', capsize=5, color='green', transform=trans1)
plt.errorbar(df_z['Object'], df_z['z_auto'], yerr=df_z['e_z_auto'], fmt='o', capsize=5, color='red')
plt.errorbar(df_z['Object'], df_z['z_petro'], yerr=df_z['e_z_petro'], fmt='o', capsize=5, color='mediumblue', transform=trans2)
plt.title("Filtro Z")
plt.legend(labels=['ISO', 'AUTO', 'PETRO'])
plt.xticks(rotation=90, ha='right')
plt.ylabel('Mag (AB)')
plt.show

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_z)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_z.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['z_auto'], row['z_iso'], yerr=row['e_z_iso'], xerr=row['e_z_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro Z", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(10.5, 17, 0.25))
plt.yticks(np.arange(10.5, 17, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %% 

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_z)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_z.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['z_auto'], row['z_petro'], yerr=row['e_z_petro'], xerr=row['e_z_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro Z", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('PETRO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(10.5, 17, 0.25))
plt.yticks(np.arange(10.5, 17, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_z)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_z.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['z_petro'], row['z_iso'], yerr=row['e_z_iso'], xerr=row['e_z_petro'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro Z", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('PETRO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(10.5, 17.5, 0.25))
plt.yticks(np.arange(10.5, 17.5, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

filter_select='J0378'
objs_DT_ID=['iDR4_3_SPLUS-s38s28_0037355',
 'iDR4_3_SPLUS-s42s26_0009603',
 'iDR4_3_SPLUS-s37s33_0029178',
 'iDR4_3_SPLUS-n14s32_0029533',
 'iDR4_3_SPLUS-s19s23_0054919',
 'iDR4_3_SPLUS-s20s23_0027579',
 'iDR4_3_SPLUS-s27s07_0017147',
 'iDR4_3_SPLUS-s26s07_0019433',
 'iDR4_3_SPLUS-s21s10_0013331',
 'iDR4_3_SPLUS-s27s10_0013261',
 'iDR4_3_SPLUS-s24s11_0019810',
 'iDR4_3_SPLUS-s26s14_0016119',
 'iDR4_3_SPLUS-s25s18_0032854',
 'iDR4_3_SPLUS-s27s18_0020395',
 'iDR4_3_SPLUS-s27s19_0008651',
 'iDR4_3_SPLUS-s24s20_0018168',
 'iDR4_3_SPLUS-s25s21_0007448',
 'iDR4_3_SPLUS-s25s24_0039312',
 'iDR4_3_SPLUS-s37s20_0007226',
 'iDR4_3_SPLUS-s34s21_0008123',
 'iDR4_3_SPLUS-s24s26_0024907',
 'iDR4_3_SPLUS-s29s31_0019272',
 'iDR4_3_MC0103_0014596',
 'iDR4_3_SPLUS-s25s34_0007124',
 'iDR4_3_SPLUS-s24s35_0029130',
 'iDR4_3_SPLUS-s25s35_0013478',
 'iDR4_3_SPLUS-s30s34_0033827',
 'iDR4_3_SPLUS-s44s24_0025720',
 'iDR4_3_SPLUS-s28s36_0033655',
 'iDR4_3_SPLUS-s43s27_0018255',
 'iDR4_3_SPLUS-s41s30_0038650',
 'iDR4_3_SPLUS-s45s26_0000595',
 'iDR4_3_SPLUS-s38s32_0024249',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s30s39_0009952',
 'iDR4_3_SPLUS-s34s36_0027270',
 'iDR4_3_SPLUS-s37s34_0035918',
 'iDR4_3_SPLUS-s38s34_0035796',
 'iDR4_3_MC0001_0037018',
 'iDR4_3_SPLUS-s30s41_0001982',
 'iDR4_3_SPLUS-s25s45_0034698',
 'iDR4_3_MC0012_0039189',
 'iDR4_3_SPLUS-s39s34_0021319',
 'iDR4_3_SPLUS-s41s32_0034511',
 'iDR4_3_MC0021_0034624',
 'iDR4_3_SPLUS-s29s42_0019922',
 'iDR4_3_SPLUS-s41s33_0032981',
 'iDR4_3_MC0002_0006559',
 'iDR4_3_SPLUS-s36s39_0020464',
 'iDR4_3_SPLUS-n17s01_0006918',
 'iDR4_3_HYDRA-0104_0011186',
 'iDR4_3_HYDRA-0051_0052680',
 'iDR4_3_HYDRA-0015_0018824',
 'iDR4_3_HYDRA-0106_0049457',
 'iDR4_3_HYDRA-0028_0034942',
 'iDR4_3_HYDRA-0085_0056446',
 'iDR4_3_SPLUS-n20s03_0026145',
 'iDR4_3_SPLUS-n19s22_0026754',
 'iDR4_3_SPLUS-n20s23_0017293',
 'iDR4_3_SPLUS-s30s45_0010034',
 'iDR4_3_SPLUS-s20s23_0040961',
 'iDR4_3_SPLUS-s19s23_0047619',
 'iDR4_3_SPLUS-s23s26_0001674',
 'iDR4_3_SPLUS-s23s26_0041931',
 'iDR4_3_SPLUS-s24s65_0036645',
 'iDR4_3_SPLUS-s45s41_0034849',
 'iDR4_3_SPLUS-s23s46_0035512',
 'iDR4_3_SPLUS-s24s74_0016239',
 'iDR4_3_STRIPE82-0063_0034198',
 'iDR4_3_SPLUS-n13s31_0042282',
 'iDR4_3_SPLUS-n14s32_0041298',
 'iDR4_3_SPLUS-s39s31_0013592',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s21s23_0012649',
 'iDR4_3_HYDRA-0045_0074200',
 'iDR4_3_SPLUS-s34s39_0021318',
 'iDR4_3_SPLUS-s25s35_0015877',
 'iDR4_3_SPLUS-s25s13_0012636',
 'iDR4_3_MC0125_0008151',
 'iDR4_3_MC0097_0015643',
 'iDR4_3_SPLUS-s46s24_0023799',
 'iDR4_3_SPLUS-s46s24_0023407',
 'iDR4_3_SPLUS-s46s25_0034017',
 'iDR4_3_SPLUS-s42s30_0018253',
 'iDR4_3_SPLUS-s42s31_0030929',
 'iDR4_3_SPLUS-s42s32_0018085',
 'iDR4_3_SPLUS-s38s32_0022611',
 'iDR4_3_SPLUS-s38s33_0011808',
 'iDR4_3_SPLUS-s25s13_0017486',
 'iDR4_3_SPLUS-s29s42_0028164',
 'iDR4_3_SPLUS-s26s45_0031867',
 'iDR4_3_HYDRA-0107_0006292',
 'iDR4_3_SPLUS-s24s09_0028206',
 'iDR4_3_SPLUS-s25s10_0026796',
 'iDR4_3_SPLUS-s21s22_0024741',
 'iDR4_3_SPLUS-n15s05_0037835',
 'iDR4_3_SPLUS-n15s19_0049168',
 'iDR4_3_SPLUS-n15s20_0043589',
 'iDR4_3_SPLUS-n17s20_0011455',
 'iDR4_3_SPLUS-n17s21_0060665',
 'iDR4_3_SPLUS-n17s21_0053822',
 'iDR4_3_SPLUS-n16s21_0006990',
 'iDR4_3_SPLUS-n09s38_0057827',
 'iDR4_3_SPLUS-s20s23_0025387',
 'iDR4_3_SPLUS-s45s41_0027231',
 'iDR4_3_SPLUS-s24s09_0027219',
 'iDR4_3_SPLUS-s25s10_0025796',
 'iDR4_3_SPLUS-s20s23_0041616',
 'iDR4_3_SPLUS-s20s23_0029315',
 'iDR4_3_STRIPE82-0005_0021998',
 'iDR4_3_SPLUS-n02s38_0008843',
 'iDR4_3_SPLUS-n01s23_0026692',
 'iDR4_3_SPLUS-n01s37_0009556',
 'iDR4_3_STRIPE82-0105_0023811',
 'iDR4_3_STRIPE82-0059_0016073',
 'iDR4_3_STRIPE82-0104_0005639',
 'iDR4_3_STRIPE82-0014_0039180']

x_values = ["ISO","AUTO","PETRO"]
columns_filter = ['Object', 'J0378_iso', 'J0378_auto', 'J0378_petro', 'e_J0378_iso', 'e_J0378_auto', 'e_J0378_petro']
df_J0378 = pd.DataFrame(columns = columns_filter)
table_to_skip = [1, 4, 5, 7, 8, 10, 14, 15, 21, 22, 23, 24, 26, 29, 31, 34, 38, 41, 42, 43, 44, 45, 48, 49, 55, 57, 59, 61, 62, 63, 65, 68, 71, 73, 74, 80, 81, 83, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 96, 97, 99, 100, 107, 111, 112, 115]

iso_mags=[]
iso_errors=[]
auto_mags=[]
auto_errors=[]
petro_mags=[]
petro_errors=[]

plt.figure(figsize=(10,6))
for table, obj in enumerate(objs_DT_ID):
    if table in table_to_skip:
        continue
    iso_row = iso_list[table][iso_list[table]['ID']==obj][0]
    auto_row = auto_list[table][auto_list[table]['ID']==obj][0]
    petro_row = petro_list[table][petro_list[table]['ID']==obj][0]
    
    mag_values=[]
    error_values=[]
    mag_values.append(iso_row[f'{filter_select}_iso'])
    error_values.append(iso_row[f'e_{filter_select}_iso'])
    mag_values.append(auto_row[f'{filter_select}_auto'])
    error_values.append(auto_row[f'e_{filter_select}_auto'])
    mag_values.append(petro_row[f'{filter_select}_petro'])
    error_values.append(petro_row[f'e_{filter_select}_petro'])
    
    iso_mags.append(iso_row[f'{filter_select}_iso'])
    auto_mags.append(auto_row[f'{filter_select}_auto'])
    petro_mags.append(petro_row[f'{filter_select}_petro'])
    iso_errors.append(iso_row[f'e_{filter_select}_iso'])
    auto_errors.append(auto_row[f'e_{filter_select}_auto'])
    petro_errors.append(petro_row[f'e_{filter_select}_petro'])

    df2 = pd.DataFrame([[loaded_gal_sample["Object"].iloc[table], mag_values[0], mag_values[1], mag_values[2], error_values[0], error_values[1], error_values[2]]],columns=columns_filter)
    df_J0378 = pd.concat([df_J0378, df2],ignore_index=True)

iso_mean = sum(iso_mags) / len(iso_mags)
auto_mean = sum(auto_mags) / len(auto_mags)
petro_mean = sum(petro_mags) / len(petro_mags)

iso_error_mean = sum(iso_errors) / len(iso_errors)
auto_error_mean = sum(auto_errors) / len(auto_errors)
petro_error_mean = sum(petro_errors) / len(petro_errors)

plt.errorbar(['ISO'], [iso_mean], yerr=[iso_error_mean], fmt='o', capsize=5, color='green', label='ISO')
plt.errorbar(['AUTO'], [auto_mean], yerr=[auto_error_mean], fmt='o', capsize=5, color='red', label='AUTO')
plt.errorbar(['PETRO'], [petro_mean], yerr=[petro_error_mean], fmt='o', capsize=5, color='blue', label='PETRO')

plt.grid(linestyle='--', axis='y')
plt.ylabel("Mag (AB)")
plt.title("Filtro J0378 - [OII]")
plt.legend()
plt.show()

# %%

columns = ['J0378_iso', 'J0378_auto', 'J0378_petro', 'e_J0378_iso', 'e_J0378_auto', 'e_J0378_petro']
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df_J0378[columns].quantile(0.25)
Q3 = df_J0378[columns].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1
outliers = ((df_J0378[columns] < (Q1 - 1.5 * IQR)) | (df_J0378[columns] > (Q3 + 1.5 * IQR))).any(axis=1)
outlier_df_J0378 = df_J0378[outliers]
df_J0378_without_outliers = df_J0378[~outliers]

# %%

outlier_df_J0378.boxplot()

# %%

df_J0378_without_outliers.boxplot()

# %%

df_J0378_without_outliers[['e_J0378_iso','e_J0378_auto','e_J0378_petro']].boxplot()

# %%

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_J0378[columns])
df_J0378['pca1'] = pca_result[:,0]
df_J0378['pca2'] = pca_result[:,1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca1', y='pca2', data=df_J0378, color='purple')
plt.title('Análise de componentes principais (PCA)')
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_J0378['cluster'] = kmeans.fit_predict(df_J0378[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0378_iso', y='e_J0378_iso', color='green', data=df_J0378)
plt.show()

#%%

kmeans = KMeans(n_clusters=3)
df_J0378['cluster'] = kmeans.fit_predict(df_J0378[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0378_auto', y='e_J0378_auto', color='red', data=df_J0378)
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_J0378['cluster'] = kmeans.fit_predict(df_J0378[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0378_petro', y='e_J0378_petro', color='blue', data=df_J0378)
plt.show()

#%%

g = sns.JointGrid(x="J0378_iso", y="e_J0378_iso", data=df_J0378) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="J0378_auto", y="e_J0378_auto", data=df_J0378) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="J0378_petro", y="e_J0378_petro", data=df_J0378) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

x_label = range(len(df_J0378))
fig, ax = plt.subplots(figsize=(24, 6))
trans1 = Affine2D().translate(-0.25, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.25, 0.0) + ax.transData

plt.errorbar(df_J0378['Object'], df_J0378['J0378_iso'], yerr=df_J0378['e_J0378_iso'], fmt='o', capsize=5, color='green', transform=trans1)
plt.errorbar(df_J0378['Object'], df_J0378['J0378_auto'], yerr=df_J0378['e_J0378_auto'], fmt='o', capsize=5, color='red')
plt.errorbar(df_J0378['Object'], df_J0378['J0378_petro'], yerr=df_J0378['e_J0378_petro'], fmt='o', capsize=5, color='mediumblue', transform=trans2)
plt.title("Filtro J0378 - [OII]")
plt.legend(labels=['ISO', 'AUTO', 'PETRO'])
plt.xticks(rotation=90, ha='right')
plt.ylabel('Mag (AB)')
plt.show

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0378)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0378.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0378_auto'], row['J0378_iso'], yerr=row['e_J0378_iso'], xerr=row['e_J0378_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0378 - [OII]", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(13, 19.5, 0.25))
plt.yticks(np.arange(13, 19.5, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %% 

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0378)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0378.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0378_auto'], row['J0378_petro'], yerr=row['e_J0378_petro'], xerr=row['e_J0378_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0378 - [OII]", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('PETRO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(13, 19.5, 0.25))
plt.yticks(np.arange(13, 19.5, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0378)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0378.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0378_petro'], row['J0378_iso'], yerr=row['e_J0378_iso'], xerr=row['e_J0378_petro'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0378 - [OII]", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('PETRO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(13, 20, 0.25))
plt.yticks(np.arange(13, 20, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

filter_select='J0395'
objs_DT_ID=['iDR4_3_SPLUS-s38s28_0037355',
 'iDR4_3_SPLUS-s42s26_0009603',
 'iDR4_3_SPLUS-s37s33_0029178',
 'iDR4_3_SPLUS-n14s32_0029533',
 'iDR4_3_SPLUS-s19s23_0054919',
 'iDR4_3_SPLUS-s20s23_0027579',
 'iDR4_3_SPLUS-s27s07_0017147',
 'iDR4_3_SPLUS-s26s07_0019433',
 'iDR4_3_SPLUS-s21s10_0013331',
 'iDR4_3_SPLUS-s27s10_0013261',
 'iDR4_3_SPLUS-s24s11_0019810',
 'iDR4_3_SPLUS-s26s14_0016119',
 'iDR4_3_SPLUS-s25s18_0032854',
 'iDR4_3_SPLUS-s27s18_0020395',
 'iDR4_3_SPLUS-s27s19_0008651',
 'iDR4_3_SPLUS-s24s20_0018168',
 'iDR4_3_SPLUS-s25s21_0007448',
 'iDR4_3_SPLUS-s25s24_0039312',
 'iDR4_3_SPLUS-s37s20_0007226',
 'iDR4_3_SPLUS-s34s21_0008123',
 'iDR4_3_SPLUS-s24s26_0024907',
 'iDR4_3_SPLUS-s29s31_0019272',
 'iDR4_3_MC0103_0014596',
 'iDR4_3_SPLUS-s25s34_0007124',
 'iDR4_3_SPLUS-s24s35_0029130',
 'iDR4_3_SPLUS-s25s35_0013478',
 'iDR4_3_SPLUS-s30s34_0033827',
 'iDR4_3_SPLUS-s44s24_0025720',
 'iDR4_3_SPLUS-s28s36_0033655',
 'iDR4_3_SPLUS-s43s27_0018255',
 'iDR4_3_SPLUS-s41s30_0038650',
 'iDR4_3_SPLUS-s45s26_0000595',
 'iDR4_3_SPLUS-s38s32_0024249',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s30s39_0009952',
 'iDR4_3_SPLUS-s34s36_0027270',
 'iDR4_3_SPLUS-s37s34_0035918',
 'iDR4_3_SPLUS-s38s34_0035796',
 'iDR4_3_MC0001_0037018',
 'iDR4_3_SPLUS-s30s41_0001982',
 'iDR4_3_SPLUS-s25s45_0034698',
 'iDR4_3_MC0012_0039189',
 'iDR4_3_SPLUS-s39s34_0021319',
 'iDR4_3_SPLUS-s41s32_0034511',
 'iDR4_3_MC0021_0034624',
 'iDR4_3_SPLUS-s29s42_0019922',
 'iDR4_3_SPLUS-s41s33_0032981',
 'iDR4_3_MC0002_0006559',
 'iDR4_3_SPLUS-s36s39_0020464',
 'iDR4_3_SPLUS-n17s01_0006918',
 'iDR4_3_HYDRA-0104_0011186',
 'iDR4_3_HYDRA-0051_0052680',
 'iDR4_3_HYDRA-0015_0018824',
 'iDR4_3_HYDRA-0106_0049457',
 'iDR4_3_HYDRA-0028_0034942',
 'iDR4_3_HYDRA-0085_0056446',
 'iDR4_3_SPLUS-n20s03_0026145',
 'iDR4_3_SPLUS-n19s22_0026754',
 'iDR4_3_SPLUS-n20s23_0017293',
 'iDR4_3_SPLUS-s30s45_0010034',
 'iDR4_3_SPLUS-s20s23_0040961',
 'iDR4_3_SPLUS-s19s23_0047619',
 'iDR4_3_SPLUS-s23s26_0001674',
 'iDR4_3_SPLUS-s23s26_0041931',
 'iDR4_3_SPLUS-s24s65_0036645',
 'iDR4_3_SPLUS-s45s41_0034849',
 'iDR4_3_SPLUS-s23s46_0035512',
 'iDR4_3_SPLUS-s24s74_0016239',
 'iDR4_3_STRIPE82-0063_0034198',
 'iDR4_3_SPLUS-n13s31_0042282',
 'iDR4_3_SPLUS-n14s32_0041298',
 'iDR4_3_SPLUS-s39s31_0013592',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s21s23_0012649',
 'iDR4_3_HYDRA-0045_0074200',
 'iDR4_3_SPLUS-s34s39_0021318',
 'iDR4_3_SPLUS-s25s35_0015877',
 'iDR4_3_SPLUS-s25s13_0012636',
 'iDR4_3_MC0125_0008151',
 'iDR4_3_MC0097_0015643',
 'iDR4_3_SPLUS-s46s24_0023799',
 'iDR4_3_SPLUS-s46s24_0023407',
 'iDR4_3_SPLUS-s46s25_0034017',
 'iDR4_3_SPLUS-s42s30_0018253',
 'iDR4_3_SPLUS-s42s31_0030929',
 'iDR4_3_SPLUS-s42s32_0018085',
 'iDR4_3_SPLUS-s38s32_0022611',
 'iDR4_3_SPLUS-s38s33_0011808',
 'iDR4_3_SPLUS-s25s13_0017486',
 'iDR4_3_SPLUS-s29s42_0028164',
 'iDR4_3_SPLUS-s26s45_0031867',
 'iDR4_3_HYDRA-0107_0006292',
 'iDR4_3_SPLUS-s24s09_0028206',
 'iDR4_3_SPLUS-s25s10_0026796',
 'iDR4_3_SPLUS-s21s22_0024741',
 'iDR4_3_SPLUS-n15s05_0037835',
 'iDR4_3_SPLUS-n15s19_0049168',
 'iDR4_3_SPLUS-n15s20_0043589',
 'iDR4_3_SPLUS-n17s20_0011455',
 'iDR4_3_SPLUS-n17s21_0060665',
 'iDR4_3_SPLUS-n17s21_0053822',
 'iDR4_3_SPLUS-n16s21_0006990',
 'iDR4_3_SPLUS-n09s38_0057827',
 'iDR4_3_SPLUS-s20s23_0025387',
 'iDR4_3_SPLUS-s45s41_0027231',
 'iDR4_3_SPLUS-s24s09_0027219',
 'iDR4_3_SPLUS-s25s10_0025796',
 'iDR4_3_SPLUS-s20s23_0041616',
 'iDR4_3_SPLUS-s20s23_0029315',
 'iDR4_3_STRIPE82-0005_0021998',
 'iDR4_3_SPLUS-n02s38_0008843',
 'iDR4_3_SPLUS-n01s23_0026692',
 'iDR4_3_SPLUS-n01s37_0009556',
 'iDR4_3_STRIPE82-0105_0023811',
 'iDR4_3_STRIPE82-0059_0016073',
 'iDR4_3_STRIPE82-0104_0005639',
 'iDR4_3_STRIPE82-0014_0039180']

x_values = ["ISO","AUTO","PETRO"]
columns_filter = ['Object', 'J0395_iso', 'J0395_auto', 'J0395_petro', 'e_J0395_iso', 'e_J0395_auto', 'e_J0395_petro']
df_J0395 = pd.DataFrame(columns = columns_filter)
table_to_skip = [1, 4, 5, 7, 8, 10, 14, 15, 21, 22, 23, 24, 26, 29, 31, 34, 38, 41, 42, 43, 44, 45, 48, 49, 55, 57, 59, 61, 62, 63, 64, 65, 68, 71, 73, 74, 80, 81, 83, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 96, 97, 99, 100, 107, 111, 112, 115]

iso_mags=[]
iso_errors=[]
auto_mags=[]
auto_errors=[]
petro_mags=[]
petro_errors=[]

plt.figure(figsize=(10,6))
for table, obj in enumerate(objs_DT_ID):
    if table in table_to_skip:
        continue
    iso_row = iso_list[table][iso_list[table]['ID']==obj][0]
    auto_row = auto_list[table][auto_list[table]['ID']==obj][0]
    petro_row = petro_list[table][petro_list[table]['ID']==obj][0]
    
    mag_values=[]
    error_values=[]
    mag_values.append(iso_row[f'{filter_select}_iso'])
    error_values.append(iso_row[f'e_{filter_select}_iso'])
    mag_values.append(auto_row[f'{filter_select}_auto'])
    error_values.append(auto_row[f'e_{filter_select}_auto'])
    mag_values.append(petro_row[f'{filter_select}_petro'])
    error_values.append(petro_row[f'e_{filter_select}_petro'])
    
    iso_mags.append(iso_row[f'{filter_select}_iso'])
    auto_mags.append(auto_row[f'{filter_select}_auto'])
    petro_mags.append(petro_row[f'{filter_select}_petro'])
    iso_errors.append(iso_row[f'e_{filter_select}_iso'])
    auto_errors.append(auto_row[f'e_{filter_select}_auto'])
    petro_errors.append(petro_row[f'e_{filter_select}_petro'])

    df2 = pd.DataFrame([[loaded_gal_sample["Object"].iloc[table], mag_values[0], mag_values[1], mag_values[2], error_values[0], error_values[1], error_values[2]]],columns=columns_filter)
    df_J0395 = pd.concat([df_J0395, df2],ignore_index=True)

iso_mean = sum(iso_mags) / len(iso_mags)
auto_mean = sum(auto_mags) / len(auto_mags)
petro_mean = sum(petro_mags) / len(petro_mags)

iso_error_mean = sum(iso_errors) / len(iso_errors)
auto_error_mean = sum(auto_errors) / len(auto_errors)
petro_error_mean = sum(petro_errors) / len(petro_errors)

plt.errorbar(['ISO'], [iso_mean], yerr=[iso_error_mean], fmt='o', capsize=5, color='green', label='ISO')
plt.errorbar(['AUTO'], [auto_mean], yerr=[auto_error_mean], fmt='o', capsize=5, color='red', label='AUTO')
plt.errorbar(['PETRO'], [petro_mean], yerr=[petro_error_mean], fmt='o', capsize=5, color='blue', label='PETRO')

plt.grid(linestyle='--', axis='y')
plt.ylabel("Mag (AB)")
plt.title("Filtro J0395 - Ca H+K")
plt.legend()
plt.show()

# %%

columns = ['J0395_iso', 'J0395_auto', 'J0395_petro', 'e_J0395_iso', 'e_J0395_auto', 'e_J0395_petro']
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df_J0395[columns].quantile(0.25)
Q3 = df_J0395[columns].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1
outliers = ((df_J0395[columns] < (Q1 - 1.5 * IQR)) | (df_J0395[columns] > (Q3 + 1.5 * IQR))).any(axis=1)
outlier_df_J0395 = df_J0395[outliers]
df_J0395_without_outliers = df_J0395[~outliers]

# %%

outlier_df_J0395.boxplot()

# %%

df_J0395_without_outliers.boxplot()

# %%

df_J0395_without_outliers[['e_J0395_iso','e_J0395_auto','e_J0395_petro']].boxplot()

# %%

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_J0395[columns])
df_J0395['pca1'] = pca_result[:,0]
df_J0395['pca2'] = pca_result[:,1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca1', y='pca2', data=df_J0395, color='purple')
plt.title('Análise de componentes principais (PCA)')
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_J0395['cluster'] = kmeans.fit_predict(df_J0395[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0395_iso', y='e_J0395_iso', color='green', data=df_J0395)
plt.show()

#%%

kmeans = KMeans(n_clusters=3)
df_J0395['cluster'] = kmeans.fit_predict(df_J0395[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0395_auto', y='e_J0395_auto', color='red', data=df_J0395)
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_J0395['cluster'] = kmeans.fit_predict(df_J0395[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0395_petro', y='e_J0395_petro', color='blue', data=df_J0395)
plt.show()

#%%

g = sns.JointGrid(x="J0395_iso", y="e_J0395_iso", data=df_J0395) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="J0395_auto", y="e_J0395_auto", data=df_J0395) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="J0395_petro", y="e_J0395_petro", data=df_J0395) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

x_label = range(len(df_J0395))
fig, ax = plt.subplots(figsize=(24, 6))
trans1 = Affine2D().translate(-0.25, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.25, 0.0) + ax.transData

plt.errorbar(df_J0395['Object'], df_J0395['J0395_iso'], yerr=df_J0395['e_J0395_iso'], fmt='o', capsize=5, color='green', transform=trans1)
plt.errorbar(df_J0395['Object'], df_J0395['J0395_auto'], yerr=df_J0395['e_J0395_auto'], fmt='o', capsize=5, color='red')
plt.errorbar(df_J0395['Object'], df_J0395['J0395_petro'], yerr=df_J0395['e_J0395_petro'], fmt='o', capsize=5, color='mediumblue', transform=trans2)
plt.title("Filtro J0395 - Ca H+K")
plt.legend(labels=['ISO', 'AUTO', 'PETRO'])
plt.xticks(rotation=90, ha='right')
plt.ylabel('Mag (AB)')
plt.show

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0395)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0395.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0395_auto'], row['J0395_iso'], yerr=row['e_J0395_iso'], xerr=row['e_J0395_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0395 - Ca H+K", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(13, 20, 0.25))
plt.yticks(np.arange(13, 20, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %% 

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0395)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0395.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0395_auto'], row['J0395_petro'], yerr=row['e_J0395_petro'], xerr=row['e_J0395_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0395 - Ca H+K", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('PETRO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(13, 20, 0.25))
plt.yticks(np.arange(13, 20, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0395)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0395.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0395_petro'], row['J0395_iso'], yerr=row['e_J0395_iso'], xerr=row['e_J0395_petro'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0395 - Ca H+K", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('PETRO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(13, 21, 0.25))
plt.yticks(np.arange(13, 21, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

filter_select='J0410'
objs_DT_ID=['iDR4_3_SPLUS-s38s28_0037355',
 'iDR4_3_SPLUS-s42s26_0009603',
 'iDR4_3_SPLUS-s37s33_0029178',
 'iDR4_3_SPLUS-n14s32_0029533',
 'iDR4_3_SPLUS-s19s23_0054919',
 'iDR4_3_SPLUS-s20s23_0027579',
 'iDR4_3_SPLUS-s27s07_0017147',
 'iDR4_3_SPLUS-s26s07_0019433',
 'iDR4_3_SPLUS-s21s10_0013331',
 'iDR4_3_SPLUS-s27s10_0013261',
 'iDR4_3_SPLUS-s24s11_0019810',
 'iDR4_3_SPLUS-s26s14_0016119',
 'iDR4_3_SPLUS-s25s18_0032854',
 'iDR4_3_SPLUS-s27s18_0020395',
 'iDR4_3_SPLUS-s27s19_0008651',
 'iDR4_3_SPLUS-s24s20_0018168',
 'iDR4_3_SPLUS-s25s21_0007448',
 'iDR4_3_SPLUS-s25s24_0039312',
 'iDR4_3_SPLUS-s37s20_0007226',
 'iDR4_3_SPLUS-s34s21_0008123',
 'iDR4_3_SPLUS-s24s26_0024907',
 'iDR4_3_SPLUS-s29s31_0019272',
 'iDR4_3_MC0103_0014596',
 'iDR4_3_SPLUS-s25s34_0007124',
 'iDR4_3_SPLUS-s24s35_0029130',
 'iDR4_3_SPLUS-s25s35_0013478',
 'iDR4_3_SPLUS-s30s34_0033827',
 'iDR4_3_SPLUS-s44s24_0025720',
 'iDR4_3_SPLUS-s28s36_0033655',
 'iDR4_3_SPLUS-s43s27_0018255',
 'iDR4_3_SPLUS-s41s30_0038650',
 'iDR4_3_SPLUS-s45s26_0000595',
 'iDR4_3_SPLUS-s38s32_0024249',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s30s39_0009952',
 'iDR4_3_SPLUS-s34s36_0027270',
 'iDR4_3_SPLUS-s37s34_0035918',
 'iDR4_3_SPLUS-s38s34_0035796',
 'iDR4_3_MC0001_0037018',
 'iDR4_3_SPLUS-s30s41_0001982',
 'iDR4_3_SPLUS-s25s45_0034698',
 'iDR4_3_MC0012_0039189',
 'iDR4_3_SPLUS-s39s34_0021319',
 'iDR4_3_SPLUS-s41s32_0034511',
 'iDR4_3_MC0021_0034624',
 'iDR4_3_SPLUS-s29s42_0019922',
 'iDR4_3_SPLUS-s41s33_0032981',
 'iDR4_3_MC0002_0006559',
 'iDR4_3_SPLUS-s36s39_0020464',
 'iDR4_3_SPLUS-n17s01_0006918',
 'iDR4_3_HYDRA-0104_0011186',
 'iDR4_3_HYDRA-0051_0052680',
 'iDR4_3_HYDRA-0015_0018824',
 'iDR4_3_HYDRA-0106_0049457',
 'iDR4_3_HYDRA-0028_0034942',
 'iDR4_3_HYDRA-0085_0056446',
 'iDR4_3_SPLUS-n20s03_0026145',
 'iDR4_3_SPLUS-n19s22_0026754',
 'iDR4_3_SPLUS-n20s23_0017293',
 'iDR4_3_SPLUS-s30s45_0010034',
 'iDR4_3_SPLUS-s20s23_0040961',
 'iDR4_3_SPLUS-s19s23_0047619',
 'iDR4_3_SPLUS-s23s26_0001674',
 'iDR4_3_SPLUS-s23s26_0041931',
 'iDR4_3_SPLUS-s24s65_0036645',
 'iDR4_3_SPLUS-s45s41_0034849',
 'iDR4_3_SPLUS-s23s46_0035512',
 'iDR4_3_SPLUS-s24s74_0016239',
 'iDR4_3_STRIPE82-0063_0034198',
 'iDR4_3_SPLUS-n13s31_0042282',
 'iDR4_3_SPLUS-n14s32_0041298',
 'iDR4_3_SPLUS-s39s31_0013592',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s21s23_0012649',
 'iDR4_3_HYDRA-0045_0074200',
 'iDR4_3_SPLUS-s34s39_0021318',
 'iDR4_3_SPLUS-s25s35_0015877',
 'iDR4_3_SPLUS-s25s13_0012636',
 'iDR4_3_MC0125_0008151',
 'iDR4_3_MC0097_0015643',
 'iDR4_3_SPLUS-s46s24_0023799',
 'iDR4_3_SPLUS-s46s24_0023407',
 'iDR4_3_SPLUS-s46s25_0034017',
 'iDR4_3_SPLUS-s42s30_0018253',
 'iDR4_3_SPLUS-s42s31_0030929',
 'iDR4_3_SPLUS-s42s32_0018085',
 'iDR4_3_SPLUS-s38s32_0022611',
 'iDR4_3_SPLUS-s38s33_0011808',
 'iDR4_3_SPLUS-s25s13_0017486',
 'iDR4_3_SPLUS-s29s42_0028164',
 'iDR4_3_SPLUS-s26s45_0031867',
 'iDR4_3_HYDRA-0107_0006292',
 'iDR4_3_SPLUS-s24s09_0028206',
 'iDR4_3_SPLUS-s25s10_0026796',
 'iDR4_3_SPLUS-s21s22_0024741',
 'iDR4_3_SPLUS-n15s05_0037835',
 'iDR4_3_SPLUS-n15s19_0049168',
 'iDR4_3_SPLUS-n15s20_0043589',
 'iDR4_3_SPLUS-n17s20_0011455',
 'iDR4_3_SPLUS-n17s21_0060665',
 'iDR4_3_SPLUS-n17s21_0053822',
 'iDR4_3_SPLUS-n16s21_0006990',
 'iDR4_3_SPLUS-n09s38_0057827',
 'iDR4_3_SPLUS-s20s23_0025387',
 'iDR4_3_SPLUS-s45s41_0027231',
 'iDR4_3_SPLUS-s24s09_0027219',
 'iDR4_3_SPLUS-s25s10_0025796',
 'iDR4_3_SPLUS-s20s23_0041616',
 'iDR4_3_SPLUS-s20s23_0029315',
 'iDR4_3_STRIPE82-0005_0021998',
 'iDR4_3_SPLUS-n02s38_0008843',
 'iDR4_3_SPLUS-n01s23_0026692',
 'iDR4_3_SPLUS-n01s37_0009556',
 'iDR4_3_STRIPE82-0105_0023811',
 'iDR4_3_STRIPE82-0059_0016073',
 'iDR4_3_STRIPE82-0104_0005639',
 'iDR4_3_STRIPE82-0014_0039180']

x_values = ["ISO","AUTO","PETRO"]
columns_filter = ['Object', 'J0410_iso', 'J0410_auto', 'J0410_petro', 'e_J0410_iso', 'e_J0410_auto', 'e_J0410_petro']
df_J0410 = pd.DataFrame(columns = columns_filter)
table_to_skip = [1, 4, 5, 7, 8, 10, 14, 15, 21, 22, 23, 24, 26, 29, 31, 34, 38, 41, 42, 43, 44, 45, 48, 49, 55, 57, 59, 61, 62, 63, 65, 68, 71, 73, 74, 80, 81, 83, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 96, 97, 99, 100, 107, 111, 112, 115]

iso_mags=[]
iso_errors=[]
auto_mags=[]
auto_errors=[]
petro_mags=[]
petro_errors=[]

plt.figure(figsize=(10,6))
for table, obj in enumerate(objs_DT_ID):
    if table in table_to_skip:
        continue
    iso_row = iso_list[table][iso_list[table]['ID']==obj][0]
    auto_row = auto_list[table][auto_list[table]['ID']==obj][0]
    petro_row = petro_list[table][petro_list[table]['ID']==obj][0]
    
    mag_values=[]
    error_values=[]
    mag_values.append(iso_row[f'{filter_select}_iso'])
    error_values.append(iso_row[f'e_{filter_select}_iso'])
    mag_values.append(auto_row[f'{filter_select}_auto'])
    error_values.append(auto_row[f'e_{filter_select}_auto'])
    mag_values.append(petro_row[f'{filter_select}_petro'])
    error_values.append(petro_row[f'e_{filter_select}_petro'])
    
    iso_mags.append(iso_row[f'{filter_select}_iso'])
    auto_mags.append(auto_row[f'{filter_select}_auto'])
    petro_mags.append(petro_row[f'{filter_select}_petro'])
    iso_errors.append(iso_row[f'e_{filter_select}_iso'])
    auto_errors.append(auto_row[f'e_{filter_select}_auto'])
    petro_errors.append(petro_row[f'e_{filter_select}_petro'])

    df2 = pd.DataFrame([[loaded_gal_sample["Object"].iloc[table], mag_values[0], mag_values[1], mag_values[2], error_values[0], error_values[1], error_values[2]]],columns=columns_filter)
    df_J0410 = pd.concat([df_J0410, df2],ignore_index=True)

iso_mean = sum(iso_mags) / len(iso_mags)
auto_mean = sum(auto_mags) / len(auto_mags)
petro_mean = sum(petro_mags) / len(petro_mags)

iso_error_mean = sum(iso_errors) / len(iso_errors)
auto_error_mean = sum(auto_errors) / len(auto_errors)
petro_error_mean = sum(petro_errors) / len(petro_errors)

plt.errorbar(['ISO'], [iso_mean], yerr=[iso_error_mean], fmt='o', capsize=5, color='green', label='ISO')
plt.errorbar(['AUTO'], [auto_mean], yerr=[auto_error_mean], fmt='o', capsize=5, color='red', label='AUTO')
plt.errorbar(['PETRO'], [petro_mean], yerr=[petro_error_mean], fmt='o', capsize=5, color='blue', label='PETRO')

plt.grid(linestyle='--', axis='y')
plt.ylabel("Mag (AB)")
plt.title("Filtro J0410 - Hδ")
plt.legend()
plt.show()

# %%

columns = ['J0410_iso', 'J0410_auto', 'J0410_petro', 'e_J0410_iso', 'e_J0410_auto', 'e_J0410_petro']
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df_J0410[columns].quantile(0.25)
Q3 = df_J0410[columns].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1
outliers = ((df_J0410[columns] < (Q1 - 1.5 * IQR)) | (df_J0410[columns] > (Q3 + 1.5 * IQR))).any(axis=1)
outlier_df_J0410 = df_J0410[outliers]
df_J0410_without_outliers = df_J0410[~outliers]

# %%

outlier_df_J0410.boxplot()

# %%

df_J0410_without_outliers.boxplot()

# %%

df_J0410_without_outliers[['e_J0410_iso','e_J0410_auto','e_J0410_petro']].boxplot()

# %%

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_J0410[columns])
df_J0410['pca1'] = pca_result[:,0]
df_J0410['pca2'] = pca_result[:,1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca1', y='pca2', data=df_J0410, color='purple')
plt.title('Análise de componentes principais (PCA)')
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_J0410['cluster'] = kmeans.fit_predict(df_J0410[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0410_iso', y='e_J0410_iso', color='green', data=df_J0410)
plt.show()

#%%

kmeans = KMeans(n_clusters=3)
df_J0410['cluster'] = kmeans.fit_predict(df_J0410[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0410_auto', y='e_J0410_auto', color='red', data=df_J0410)
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_J0410['cluster'] = kmeans.fit_predict(df_J0410[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0410_petro', y='e_J0410_petro', color='blue', data=df_J0410)
plt.show()

#%%

g = sns.JointGrid(x="J0410_iso", y="e_J0410_iso", data=df_J0410) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="J0410_auto", y="e_J0410_auto", data=df_J0410) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="J0410_petro", y="e_J0410_petro", data=df_J0410) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

x_label = range(len(df_J0410))
fig, ax = plt.subplots(figsize=(24, 6))
trans1 = Affine2D().translate(-0.25, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.25, 0.0) + ax.transData

plt.errorbar(df_J0410['Object'], df_J0410['J0410_iso'], yerr=df_J0410['e_J0410_iso'], fmt='o', capsize=5, color='green', transform=trans1)
plt.errorbar(df_J0410['Object'], df_J0410['J0410_auto'], yerr=df_J0410['e_J0410_auto'], fmt='o', capsize=5, color='red')
plt.errorbar(df_J0410['Object'], df_J0410['J0410_petro'], yerr=df_J0410['e_J0410_petro'], fmt='o', capsize=5, color='mediumblue', transform=trans2)
plt.title("Filtro J0410 - Hδ")
plt.legend(labels=['ISO', 'AUTO', 'PETRO'])
plt.xticks(rotation=90, ha='right')
plt.ylabel('Mag (AB)')
plt.show

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0410)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0410.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0410_auto'], row['J0410_iso'], yerr=row['e_J0410_iso'], xerr=row['e_J0410_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0410 - Hδ", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(12.5, 20, 0.25))
plt.yticks(np.arange(12.5, 20, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %% 

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0410)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0410.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0410_auto'], row['J0410_petro'], yerr=row['e_J0410_petro'], xerr=row['e_J0410_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0410 - Hδ", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('PETRO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(12.5, 20, 0.25))
plt.yticks(np.arange(12.5, 20, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0410)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0410.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0410_petro'], row['J0410_iso'], yerr=row['e_J0410_iso'], xerr=row['e_J0410_petro'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0410 - Hδ", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('PETRO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(12.5, 19.75, 0.25))
plt.yticks(np.arange(12.5, 19.75, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

filter_select='J0430'
objs_DT_ID=['iDR4_3_SPLUS-s38s28_0037355',
 'iDR4_3_SPLUS-s42s26_0009603',
 'iDR4_3_SPLUS-s37s33_0029178',
 'iDR4_3_SPLUS-n14s32_0029533',
 'iDR4_3_SPLUS-s19s23_0054919',
 'iDR4_3_SPLUS-s20s23_0027579',
 'iDR4_3_SPLUS-s27s07_0017147',
 'iDR4_3_SPLUS-s26s07_0019433',
 'iDR4_3_SPLUS-s21s10_0013331',
 'iDR4_3_SPLUS-s27s10_0013261',
 'iDR4_3_SPLUS-s24s11_0019810',
 'iDR4_3_SPLUS-s26s14_0016119',
 'iDR4_3_SPLUS-s25s18_0032854',
 'iDR4_3_SPLUS-s27s18_0020395',
 'iDR4_3_SPLUS-s27s19_0008651',
 'iDR4_3_SPLUS-s24s20_0018168',
 'iDR4_3_SPLUS-s25s21_0007448',
 'iDR4_3_SPLUS-s25s24_0039312',
 'iDR4_3_SPLUS-s37s20_0007226',
 'iDR4_3_SPLUS-s34s21_0008123',
 'iDR4_3_SPLUS-s24s26_0024907',
 'iDR4_3_SPLUS-s29s31_0019272',
 'iDR4_3_MC0103_0014596',
 'iDR4_3_SPLUS-s25s34_0007124',
 'iDR4_3_SPLUS-s24s35_0029130',
 'iDR4_3_SPLUS-s25s35_0013478',
 'iDR4_3_SPLUS-s30s34_0033827',
 'iDR4_3_SPLUS-s44s24_0025720',
 'iDR4_3_SPLUS-s28s36_0033655',
 'iDR4_3_SPLUS-s43s27_0018255',
 'iDR4_3_SPLUS-s41s30_0038650',
 'iDR4_3_SPLUS-s45s26_0000595',
 'iDR4_3_SPLUS-s38s32_0024249',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s30s39_0009952',
 'iDR4_3_SPLUS-s34s36_0027270',
 'iDR4_3_SPLUS-s37s34_0035918',
 'iDR4_3_SPLUS-s38s34_0035796',
 'iDR4_3_MC0001_0037018',
 'iDR4_3_SPLUS-s30s41_0001982',
 'iDR4_3_SPLUS-s25s45_0034698',
 'iDR4_3_MC0012_0039189',
 'iDR4_3_SPLUS-s39s34_0021319',
 'iDR4_3_SPLUS-s41s32_0034511',
 'iDR4_3_MC0021_0034624',
 'iDR4_3_SPLUS-s29s42_0019922',
 'iDR4_3_SPLUS-s41s33_0032981',
 'iDR4_3_MC0002_0006559',
 'iDR4_3_SPLUS-s36s39_0020464',
 'iDR4_3_SPLUS-n17s01_0006918',
 'iDR4_3_HYDRA-0104_0011186',
 'iDR4_3_HYDRA-0051_0052680',
 'iDR4_3_HYDRA-0015_0018824',
 'iDR4_3_HYDRA-0106_0049457',
 'iDR4_3_HYDRA-0028_0034942',
 'iDR4_3_HYDRA-0085_0056446',
 'iDR4_3_SPLUS-n20s03_0026145',
 'iDR4_3_SPLUS-n19s22_0026754',
 'iDR4_3_SPLUS-n20s23_0017293',
 'iDR4_3_SPLUS-s30s45_0010034',
 'iDR4_3_SPLUS-s20s23_0040961',
 'iDR4_3_SPLUS-s19s23_0047619',
 'iDR4_3_SPLUS-s23s26_0001674',
 'iDR4_3_SPLUS-s23s26_0041931',
 'iDR4_3_SPLUS-s24s65_0036645',
 'iDR4_3_SPLUS-s45s41_0034849',
 'iDR4_3_SPLUS-s23s46_0035512',
 'iDR4_3_SPLUS-s24s74_0016239',
 'iDR4_3_STRIPE82-0063_0034198',
 'iDR4_3_SPLUS-n13s31_0042282',
 'iDR4_3_SPLUS-n14s32_0041298',
 'iDR4_3_SPLUS-s39s31_0013592',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s21s23_0012649',
 'iDR4_3_HYDRA-0045_0074200',
 'iDR4_3_SPLUS-s34s39_0021318',
 'iDR4_3_SPLUS-s25s35_0015877',
 'iDR4_3_SPLUS-s25s13_0012636',
 'iDR4_3_MC0125_0008151',
 'iDR4_3_MC0097_0015643',
 'iDR4_3_SPLUS-s46s24_0023799',
 'iDR4_3_SPLUS-s46s24_0023407',
 'iDR4_3_SPLUS-s46s25_0034017',
 'iDR4_3_SPLUS-s42s30_0018253',
 'iDR4_3_SPLUS-s42s31_0030929',
 'iDR4_3_SPLUS-s42s32_0018085',
 'iDR4_3_SPLUS-s38s32_0022611',
 'iDR4_3_SPLUS-s38s33_0011808',
 'iDR4_3_SPLUS-s25s13_0017486',
 'iDR4_3_SPLUS-s29s42_0028164',
 'iDR4_3_SPLUS-s26s45_0031867',
 'iDR4_3_HYDRA-0107_0006292',
 'iDR4_3_SPLUS-s24s09_0028206',
 'iDR4_3_SPLUS-s25s10_0026796',
 'iDR4_3_SPLUS-s21s22_0024741',
 'iDR4_3_SPLUS-n15s05_0037835',
 'iDR4_3_SPLUS-n15s19_0049168',
 'iDR4_3_SPLUS-n15s20_0043589',
 'iDR4_3_SPLUS-n17s20_0011455',
 'iDR4_3_SPLUS-n17s21_0060665',
 'iDR4_3_SPLUS-n17s21_0053822',
 'iDR4_3_SPLUS-n16s21_0006990',
 'iDR4_3_SPLUS-n09s38_0057827',
 'iDR4_3_SPLUS-s20s23_0025387',
 'iDR4_3_SPLUS-s45s41_0027231',
 'iDR4_3_SPLUS-s24s09_0027219',
 'iDR4_3_SPLUS-s25s10_0025796',
 'iDR4_3_SPLUS-s20s23_0041616',
 'iDR4_3_SPLUS-s20s23_0029315',
 'iDR4_3_STRIPE82-0005_0021998',
 'iDR4_3_SPLUS-n02s38_0008843',
 'iDR4_3_SPLUS-n01s23_0026692',
 'iDR4_3_SPLUS-n01s37_0009556',
 'iDR4_3_STRIPE82-0105_0023811',
 'iDR4_3_STRIPE82-0059_0016073',
 'iDR4_3_STRIPE82-0104_0005639',
 'iDR4_3_STRIPE82-0014_0039180']

x_values = ["ISO","AUTO","PETRO"]
columns_filter = ['Object', 'J0430_iso', 'J0430_auto', 'J0430_petro', 'e_J0430_iso', 'e_J0430_auto', 'e_J0430_petro']
df_J0430 = pd.DataFrame(columns = columns_filter)
table_to_skip = [1, 4, 5, 7, 8, 10, 14, 15, 21, 22, 23, 24, 26, 29, 31, 34, 38, 41, 42, 43, 45, 48, 49, 55, 57, 59, 61, 62, 63, 65, 68, 71, 73, 74, 80, 81, 83, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 96, 97, 99, 100, 107, 111, 112, 115]

iso_mags=[]
iso_errors=[]
auto_mags=[]
auto_errors=[]
petro_mags=[]
petro_errors=[]

plt.figure(figsize=(10,6))
for table, obj in enumerate(objs_DT_ID):
    if table in table_to_skip:
        continue
    iso_row = iso_list[table][iso_list[table]['ID']==obj][0]
    auto_row = auto_list[table][auto_list[table]['ID']==obj][0]
    petro_row = petro_list[table][petro_list[table]['ID']==obj][0]
    
    mag_values=[]
    error_values=[]
    mag_values.append(iso_row[f'{filter_select}_iso'])
    error_values.append(iso_row[f'e_{filter_select}_iso'])
    mag_values.append(auto_row[f'{filter_select}_auto'])
    error_values.append(auto_row[f'e_{filter_select}_auto'])
    mag_values.append(petro_row[f'{filter_select}_petro'])
    error_values.append(petro_row[f'e_{filter_select}_petro'])
    
    iso_mags.append(iso_row[f'{filter_select}_iso'])
    auto_mags.append(auto_row[f'{filter_select}_auto'])
    petro_mags.append(petro_row[f'{filter_select}_petro'])
    iso_errors.append(iso_row[f'e_{filter_select}_iso'])
    auto_errors.append(auto_row[f'e_{filter_select}_auto'])
    petro_errors.append(petro_row[f'e_{filter_select}_petro'])

    df2 = pd.DataFrame([[loaded_gal_sample["Object"].iloc[table], mag_values[0], mag_values[1], mag_values[2], error_values[0], error_values[1], error_values[2]]],columns=columns_filter)
    df_J0430 = pd.concat([df_J0430, df2],ignore_index=True)

iso_mean = sum(iso_mags) / len(iso_mags)
auto_mean = sum(auto_mags) / len(auto_mags)
petro_mean = sum(petro_mags) / len(petro_mags)

iso_error_mean = sum(iso_errors) / len(iso_errors)
auto_error_mean = sum(auto_errors) / len(auto_errors)
petro_error_mean = sum(petro_errors) / len(petro_errors)

plt.errorbar(['ISO'], [iso_mean], yerr=[iso_error_mean], fmt='o', capsize=5, color='green', label='ISO')
plt.errorbar(['AUTO'], [auto_mean], yerr=[auto_error_mean], fmt='o', capsize=5, color='red', label='AUTO')
plt.errorbar(['PETRO'], [petro_mean], yerr=[petro_error_mean], fmt='o', capsize=5, color='blue', label='PETRO')

plt.grid(linestyle='--', axis='y')
plt.ylabel("Mag (AB)")
plt.title("Filtro J0430 - G-band")
plt.legend()
plt.show()

# %%

columns = ['J0430_iso', 'J0430_auto', 'J0430_petro', 'e_J0430_iso', 'e_J0430_auto', 'e_J0430_petro']
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df_J0430[columns].quantile(0.25)
Q3 = df_J0430[columns].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1
outliers = ((df_J0430[columns] < (Q1 - 1.5 * IQR)) | (df_J0430[columns] > (Q3 + 1.5 * IQR))).any(axis=1)
outlier_df_J0430 = df_J0430[outliers]
df_J0430_without_outliers = df_J0430[~outliers]

# %%

outlier_df_J0430.boxplot()

# %%

df_J0430_without_outliers.boxplot()

# %%

df_J0430_without_outliers[['e_J0430_iso','e_J0430_auto','e_J0430_petro']].boxplot()

# %%

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_J0430[columns])
df_J0430['pca1'] = pca_result[:,0]
df_J0430['pca2'] = pca_result[:,1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca1', y='pca2', data=df_J0430, color='purple')
plt.title('Análise de componentes principais (PCA)')
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_J0430['cluster'] = kmeans.fit_predict(df_J0430[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0430_iso', y='e_J0430_iso', color='green', data=df_J0430)
plt.show()

#%%

kmeans = KMeans(n_clusters=3)
df_J0430['cluster'] = kmeans.fit_predict(df_J0430[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0430_auto', y='e_J0430_auto', color='red', data=df_J0430)
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_J0430['cluster'] = kmeans.fit_predict(df_J0430[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0430_petro', y='e_J0430_petro', color='blue', data=df_J0430)
plt.show()

#%%

g = sns.JointGrid(x="J0430_iso", y="e_J0430_iso", data=df_J0430) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="J0430_auto", y="e_J0430_auto", data=df_J0430) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="J0430_petro", y="e_J0430_petro", data=df_J0430) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

x_label = range(len(df_J0430))
fig, ax = plt.subplots(figsize=(24, 6))
trans1 = Affine2D().translate(-0.25, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.25, 0.0) + ax.transData

plt.errorbar(df_J0430['Object'], df_J0430['J0430_iso'], yerr=df_J0430['e_J0430_iso'], fmt='o', capsize=5, color='green', transform=trans1)
plt.errorbar(df_J0430['Object'], df_J0430['J0430_auto'], yerr=df_J0430['e_J0430_auto'], fmt='o', capsize=5, color='red')
plt.errorbar(df_J0430['Object'], df_J0430['J0430_petro'], yerr=df_J0430['e_J0430_petro'], fmt='o', capsize=5, color='mediumblue', transform=trans2)
plt.title("Filtro J0430 - G-band")
plt.legend(labels=['ISO', 'AUTO', 'PETRO'])
plt.xticks(rotation=90, ha='right')
plt.ylabel('Mag (AB)')
plt.show

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0430)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0430.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0430_auto'], row['J0430_iso'], yerr=row['e_J0430_iso'], xerr=row['e_J0430_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0430 - G-band", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(12.5, 19.25, 0.25))
plt.yticks(np.arange(12.5, 19.25, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %% 

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0430)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0430.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0430_auto'], row['J0430_petro'], yerr=row['e_J0430_petro'], xerr=row['e_J0430_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0430 - G-band", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('PETRO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(12.5, 19.25, 0.25))
plt.yticks(np.arange(12.5, 19.25, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0430)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0430.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0430_petro'], row['J0430_iso'], yerr=row['e_J0430_iso'], xerr=row['e_J0430_petro'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0430 - G-band", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('PETRO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(12.5, 19.75, 0.25))
plt.yticks(np.arange(12.5, 19.75, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

filter_select='J0515'
objs_DT_ID=['iDR4_3_SPLUS-s38s28_0037355',
 'iDR4_3_SPLUS-s42s26_0009603',
 'iDR4_3_SPLUS-s37s33_0029178',
 'iDR4_3_SPLUS-n14s32_0029533',
 'iDR4_3_SPLUS-s19s23_0054919',
 'iDR4_3_SPLUS-s20s23_0027579',
 'iDR4_3_SPLUS-s27s07_0017147',
 'iDR4_3_SPLUS-s26s07_0019433',
 'iDR4_3_SPLUS-s21s10_0013331',
 'iDR4_3_SPLUS-s27s10_0013261',
 'iDR4_3_SPLUS-s24s11_0019810',
 'iDR4_3_SPLUS-s26s14_0016119',
 'iDR4_3_SPLUS-s25s18_0032854',
 'iDR4_3_SPLUS-s27s18_0020395',
 'iDR4_3_SPLUS-s27s19_0008651',
 'iDR4_3_SPLUS-s24s20_0018168',
 'iDR4_3_SPLUS-s25s21_0007448',
 'iDR4_3_SPLUS-s25s24_0039312',
 'iDR4_3_SPLUS-s37s20_0007226',
 'iDR4_3_SPLUS-s34s21_0008123',
 'iDR4_3_SPLUS-s24s26_0024907',
 'iDR4_3_SPLUS-s29s31_0019272',
 'iDR4_3_MC0103_0014596',
 'iDR4_3_SPLUS-s25s34_0007124',
 'iDR4_3_SPLUS-s24s35_0029130',
 'iDR4_3_SPLUS-s25s35_0013478',
 'iDR4_3_SPLUS-s30s34_0033827',
 'iDR4_3_SPLUS-s44s24_0025720',
 'iDR4_3_SPLUS-s28s36_0033655',
 'iDR4_3_SPLUS-s43s27_0018255',
 'iDR4_3_SPLUS-s41s30_0038650',
 'iDR4_3_SPLUS-s45s26_0000595',
 'iDR4_3_SPLUS-s38s32_0024249',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s30s39_0009952',
 'iDR4_3_SPLUS-s34s36_0027270',
 'iDR4_3_SPLUS-s37s34_0035918',
 'iDR4_3_SPLUS-s38s34_0035796',
 'iDR4_3_MC0001_0037018',
 'iDR4_3_SPLUS-s30s41_0001982',
 'iDR4_3_SPLUS-s25s45_0034698',
 'iDR4_3_MC0012_0039189',
 'iDR4_3_SPLUS-s39s34_0021319',
 'iDR4_3_SPLUS-s41s32_0034511',
 'iDR4_3_MC0021_0034624',
 'iDR4_3_SPLUS-s29s42_0019922',
 'iDR4_3_SPLUS-s41s33_0032981',
 'iDR4_3_MC0002_0006559',
 'iDR4_3_SPLUS-s36s39_0020464',
 'iDR4_3_SPLUS-n17s01_0006918',
 'iDR4_3_HYDRA-0104_0011186',
 'iDR4_3_HYDRA-0051_0052680',
 'iDR4_3_HYDRA-0015_0018824',
 'iDR4_3_HYDRA-0106_0049457',
 'iDR4_3_HYDRA-0028_0034942',
 'iDR4_3_HYDRA-0085_0056446',
 'iDR4_3_SPLUS-n20s03_0026145',
 'iDR4_3_SPLUS-n19s22_0026754',
 'iDR4_3_SPLUS-n20s23_0017293',
 'iDR4_3_SPLUS-s30s45_0010034',
 'iDR4_3_SPLUS-s20s23_0040961',
 'iDR4_3_SPLUS-s19s23_0047619',
 'iDR4_3_SPLUS-s23s26_0001674',
 'iDR4_3_SPLUS-s23s26_0041931',
 'iDR4_3_SPLUS-s24s65_0036645',
 'iDR4_3_SPLUS-s45s41_0034849',
 'iDR4_3_SPLUS-s23s46_0035512',
 'iDR4_3_SPLUS-s24s74_0016239',
 'iDR4_3_STRIPE82-0063_0034198',
 'iDR4_3_SPLUS-n13s31_0042282',
 'iDR4_3_SPLUS-n14s32_0041298',
 'iDR4_3_SPLUS-s39s31_0013592',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s21s23_0012649',
 'iDR4_3_HYDRA-0045_0074200',
 'iDR4_3_SPLUS-s34s39_0021318',
 'iDR4_3_SPLUS-s25s35_0015877',
 'iDR4_3_SPLUS-s25s13_0012636',
 'iDR4_3_MC0125_0008151',
 'iDR4_3_MC0097_0015643',
 'iDR4_3_SPLUS-s46s24_0023799',
 'iDR4_3_SPLUS-s46s24_0023407',
 'iDR4_3_SPLUS-s46s25_0034017',
 'iDR4_3_SPLUS-s42s30_0018253',
 'iDR4_3_SPLUS-s42s31_0030929',
 'iDR4_3_SPLUS-s42s32_0018085',
 'iDR4_3_SPLUS-s38s32_0022611',
 'iDR4_3_SPLUS-s38s33_0011808',
 'iDR4_3_SPLUS-s25s13_0017486',
 'iDR4_3_SPLUS-s29s42_0028164',
 'iDR4_3_SPLUS-s26s45_0031867',
 'iDR4_3_HYDRA-0107_0006292',
 'iDR4_3_SPLUS-s24s09_0028206',
 'iDR4_3_SPLUS-s25s10_0026796',
 'iDR4_3_SPLUS-s21s22_0024741',
 'iDR4_3_SPLUS-n15s05_0037835',
 'iDR4_3_SPLUS-n15s19_0049168',
 'iDR4_3_SPLUS-n15s20_0043589',
 'iDR4_3_SPLUS-n17s20_0011455',
 'iDR4_3_SPLUS-n17s21_0060665',
 'iDR4_3_SPLUS-n17s21_0053822',
 'iDR4_3_SPLUS-n16s21_0006990',
 'iDR4_3_SPLUS-n09s38_0057827',
 'iDR4_3_SPLUS-s20s23_0025387',
 'iDR4_3_SPLUS-s45s41_0027231',
 'iDR4_3_SPLUS-s24s09_0027219',
 'iDR4_3_SPLUS-s25s10_0025796',
 'iDR4_3_SPLUS-s20s23_0041616',
 'iDR4_3_SPLUS-s20s23_0029315',
 'iDR4_3_STRIPE82-0005_0021998',
 'iDR4_3_SPLUS-n02s38_0008843',
 'iDR4_3_SPLUS-n01s23_0026692',
 'iDR4_3_SPLUS-n01s37_0009556',
 'iDR4_3_STRIPE82-0105_0023811',
 'iDR4_3_STRIPE82-0059_0016073',
 'iDR4_3_STRIPE82-0104_0005639',
 'iDR4_3_STRIPE82-0014_0039180']

x_values = ["ISO","AUTO","PETRO"]
columns_filter = ['Object', 'J0515_iso', 'J0515_auto', 'J0515_petro', 'e_J0515_iso', 'e_J0515_auto', 'e_J0515_petro']
df_J0515 = pd.DataFrame(columns = columns_filter)
table_to_skip = [1, 4, 5, 7, 8, 10, 14, 15, 21, 22, 23, 24, 26, 29, 31, 34, 38, 41, 42, 43, 45, 48, 49, 55, 57, 59, 61, 62, 63, 65, 68, 71, 73, 74, 80, 81, 83, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 96, 97, 99, 100, 107, 111, 112, 115]

iso_mags=[]
iso_errors=[]
auto_mags=[]
auto_errors=[]
petro_mags=[]
petro_errors=[]

plt.figure(figsize=(10,6))
for table, obj in enumerate(objs_DT_ID):
    if table in table_to_skip:
        continue
    iso_row = iso_list[table][iso_list[table]['ID']==obj][0]
    auto_row = auto_list[table][auto_list[table]['ID']==obj][0]
    petro_row = petro_list[table][petro_list[table]['ID']==obj][0]
    
    mag_values=[]
    error_values=[]
    mag_values.append(iso_row[f'{filter_select}_iso'])
    error_values.append(iso_row[f'e_{filter_select}_iso'])
    mag_values.append(auto_row[f'{filter_select}_auto'])
    error_values.append(auto_row[f'e_{filter_select}_auto'])
    mag_values.append(petro_row[f'{filter_select}_petro'])
    error_values.append(petro_row[f'e_{filter_select}_petro'])
    
    iso_mags.append(iso_row[f'{filter_select}_iso'])
    auto_mags.append(auto_row[f'{filter_select}_auto'])
    petro_mags.append(petro_row[f'{filter_select}_petro'])
    iso_errors.append(iso_row[f'e_{filter_select}_iso'])
    auto_errors.append(auto_row[f'e_{filter_select}_auto'])
    petro_errors.append(petro_row[f'e_{filter_select}_petro'])

    df2 = pd.DataFrame([[loaded_gal_sample["Object"].iloc[table], mag_values[0], mag_values[1], mag_values[2], error_values[0], error_values[1], error_values[2]]],columns=columns_filter)
    df_J0515 = pd.concat([df_J0515, df2],ignore_index=True)

iso_mean = sum(iso_mags) / len(iso_mags)
auto_mean = sum(auto_mags) / len(auto_mags)
petro_mean = sum(petro_mags) / len(petro_mags)

iso_error_mean = sum(iso_errors) / len(iso_errors)
auto_error_mean = sum(auto_errors) / len(auto_errors)
petro_error_mean = sum(petro_errors) / len(petro_errors)

plt.errorbar(['ISO'], [iso_mean], yerr=[iso_error_mean], fmt='o', capsize=5, color='green', label='ISO')
plt.errorbar(['AUTO'], [auto_mean], yerr=[auto_error_mean], fmt='o', capsize=5, color='red', label='AUTO')
plt.errorbar(['PETRO'], [petro_mean], yerr=[petro_error_mean], fmt='o', capsize=5, color='blue', label='PETRO')

plt.grid(linestyle='--', axis='y')
plt.ylabel("Mag (AB)")
plt.title("Filtro J0515 - Mgb Triplet")
plt.legend()
plt.show()

# %%

columns = ['J0515_iso', 'J0515_auto', 'J0515_petro', 'e_J0515_iso', 'e_J0515_auto', 'e_J0515_petro']
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df_J0515[columns].quantile(0.25)
Q3 = df_J0515[columns].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1
outliers = ((df_J0515[columns] < (Q1 - 1.5 * IQR)) | (df_J0515[columns] > (Q3 + 1.5 * IQR))).any(axis=1)
outlier_df_J0515 = df_J0515[outliers]
df_J0515_without_outliers = df_J0515[~outliers]

# %%

outlier_df_J0515.boxplot()

# %%

df_J0515_without_outliers.boxplot()

# %%

df_J0515_without_outliers[['e_J0515_iso','e_J0515_auto','e_J0515_petro']].boxplot()

# %%

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_J0515[columns])
df_J0515['pca1'] = pca_result[:,0]
df_J0515['pca2'] = pca_result[:,1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca1', y='pca2', data=df_J0515, color='purple')
plt.title('Análise de componentes principais (PCA)')
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_J0515['cluster'] = kmeans.fit_predict(df_J0515[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0515_iso', y='e_J0515_iso', color='green', data=df_J0515)
plt.show()

#%%

kmeans = KMeans(n_clusters=3)
df_J0515['cluster'] = kmeans.fit_predict(df_J0515[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0515_auto', y='e_J0515_auto', color='red', data=df_J0515)
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_J0515['cluster'] = kmeans.fit_predict(df_J0515[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0515_petro', y='e_J0515_petro', color='blue', data=df_J0515)
plt.show()

#%%

g = sns.JointGrid(x="J0515_iso", y="e_J0515_iso", data=df_J0515) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="J0515_auto", y="e_J0515_auto", data=df_J0515) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="J0515_petro", y="e_J0515_petro", data=df_J0515) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

x_label = range(len(df_J0515))
fig, ax = plt.subplots(figsize=(24, 6))
trans1 = Affine2D().translate(-0.25, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.25, 0.0) + ax.transData

plt.errorbar(df_J0515['Object'], df_J0515['J0515_iso'], yerr=df_J0515['e_J0515_iso'], fmt='o', capsize=5, color='green', transform=trans1)
plt.errorbar(df_J0515['Object'], df_J0515['J0515_auto'], yerr=df_J0515['e_J0515_auto'], fmt='o', capsize=5, color='red')
plt.errorbar(df_J0515['Object'], df_J0515['J0515_petro'], yerr=df_J0515['e_J0515_petro'], fmt='o', capsize=5, color='mediumblue', transform=trans2)
plt.title("Filtro J0515 - Mgb Triplet")
plt.legend(labels=['ISO', 'AUTO', 'PETRO'])
plt.xticks(rotation=90, ha='right')
plt.ylabel('Mag (AB)')
plt.show

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0515)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0515.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0515_auto'], row['J0515_iso'], yerr=row['e_J0515_iso'], xerr=row['e_J0515_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0515 - Mgb Triplet", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(12, 18.25, 0.25))
plt.yticks(np.arange(12, 18.25, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %% 

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0515)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0515.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0515_auto'], row['J0515_petro'], yerr=row['e_J0515_petro'], xerr=row['e_J0515_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0515 - Mgb Triplet", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('PETRO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(12, 18.25, 0.25))
plt.yticks(np.arange(12, 18.25, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0515)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0515.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0515_petro'], row['J0515_iso'], yerr=row['e_J0515_iso'], xerr=row['e_J0515_petro'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0515 - Mgb Triplet", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('PETRO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(11.5, 18.5, 0.25))
plt.yticks(np.arange(11.5, 18.5, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

filter_select='J0660'
objs_DT_ID=['iDR4_3_SPLUS-s38s28_0037355',
 'iDR4_3_SPLUS-s42s26_0009603',
 'iDR4_3_SPLUS-s37s33_0029178',
 'iDR4_3_SPLUS-n14s32_0029533',
 'iDR4_3_SPLUS-s19s23_0054919',
 'iDR4_3_SPLUS-s20s23_0027579',
 'iDR4_3_SPLUS-s27s07_0017147',
 'iDR4_3_SPLUS-s26s07_0019433',
 'iDR4_3_SPLUS-s21s10_0013331',
 'iDR4_3_SPLUS-s27s10_0013261',
 'iDR4_3_SPLUS-s24s11_0019810',
 'iDR4_3_SPLUS-s26s14_0016119',
 'iDR4_3_SPLUS-s25s18_0032854',
 'iDR4_3_SPLUS-s27s18_0020395',
 'iDR4_3_SPLUS-s27s19_0008651',
 'iDR4_3_SPLUS-s24s20_0018168',
 'iDR4_3_SPLUS-s25s21_0007448',
 'iDR4_3_SPLUS-s25s24_0039312',
 'iDR4_3_SPLUS-s37s20_0007226',
 'iDR4_3_SPLUS-s34s21_0008123',
 'iDR4_3_SPLUS-s24s26_0024907',
 'iDR4_3_SPLUS-s29s31_0019272',
 'iDR4_3_MC0103_0014596',
 'iDR4_3_SPLUS-s25s34_0007124',
 'iDR4_3_SPLUS-s24s35_0029130',
 'iDR4_3_SPLUS-s25s35_0013478',
 'iDR4_3_SPLUS-s30s34_0033827',
 'iDR4_3_SPLUS-s44s24_0025720',
 'iDR4_3_SPLUS-s28s36_0033655',
 'iDR4_3_SPLUS-s43s27_0018255',
 'iDR4_3_SPLUS-s41s30_0038650',
 'iDR4_3_SPLUS-s45s26_0000595',
 'iDR4_3_SPLUS-s38s32_0024249',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s30s39_0009952',
 'iDR4_3_SPLUS-s34s36_0027270',
 'iDR4_3_SPLUS-s37s34_0035918',
 'iDR4_3_SPLUS-s38s34_0035796',
 'iDR4_3_MC0001_0037018',
 'iDR4_3_SPLUS-s30s41_0001982',
 'iDR4_3_SPLUS-s25s45_0034698',
 'iDR4_3_MC0012_0039189',
 'iDR4_3_SPLUS-s39s34_0021319',
 'iDR4_3_SPLUS-s41s32_0034511',
 'iDR4_3_MC0021_0034624',
 'iDR4_3_SPLUS-s29s42_0019922',
 'iDR4_3_SPLUS-s41s33_0032981',
 'iDR4_3_MC0002_0006559',
 'iDR4_3_SPLUS-s36s39_0020464',
 'iDR4_3_SPLUS-n17s01_0006918',
 'iDR4_3_HYDRA-0104_0011186',
 'iDR4_3_HYDRA-0051_0052680',
 'iDR4_3_HYDRA-0015_0018824',
 'iDR4_3_HYDRA-0106_0049457',
 'iDR4_3_HYDRA-0028_0034942',
 'iDR4_3_HYDRA-0085_0056446',
 'iDR4_3_SPLUS-n20s03_0026145',
 'iDR4_3_SPLUS-n19s22_0026754',
 'iDR4_3_SPLUS-n20s23_0017293',
 'iDR4_3_SPLUS-s30s45_0010034',
 'iDR4_3_SPLUS-s20s23_0040961',
 'iDR4_3_SPLUS-s19s23_0047619',
 'iDR4_3_SPLUS-s23s26_0001674',
 'iDR4_3_SPLUS-s23s26_0041931',
 'iDR4_3_SPLUS-s24s65_0036645',
 'iDR4_3_SPLUS-s45s41_0034849',
 'iDR4_3_SPLUS-s23s46_0035512',
 'iDR4_3_SPLUS-s24s74_0016239',
 'iDR4_3_STRIPE82-0063_0034198',
 'iDR4_3_SPLUS-n13s31_0042282',
 'iDR4_3_SPLUS-n14s32_0041298',
 'iDR4_3_SPLUS-s39s31_0013592',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s21s23_0012649',
 'iDR4_3_HYDRA-0045_0074200',
 'iDR4_3_SPLUS-s34s39_0021318',
 'iDR4_3_SPLUS-s25s35_0015877',
 'iDR4_3_SPLUS-s25s13_0012636',
 'iDR4_3_MC0125_0008151',
 'iDR4_3_MC0097_0015643',
 'iDR4_3_SPLUS-s46s24_0023799',
 'iDR4_3_SPLUS-s46s24_0023407',
 'iDR4_3_SPLUS-s46s25_0034017',
 'iDR4_3_SPLUS-s42s30_0018253',
 'iDR4_3_SPLUS-s42s31_0030929',
 'iDR4_3_SPLUS-s42s32_0018085',
 'iDR4_3_SPLUS-s38s32_0022611',
 'iDR4_3_SPLUS-s38s33_0011808',
 'iDR4_3_SPLUS-s25s13_0017486',
 'iDR4_3_SPLUS-s29s42_0028164',
 'iDR4_3_SPLUS-s26s45_0031867',
 'iDR4_3_HYDRA-0107_0006292',
 'iDR4_3_SPLUS-s24s09_0028206',
 'iDR4_3_SPLUS-s25s10_0026796',
 'iDR4_3_SPLUS-s21s22_0024741',
 'iDR4_3_SPLUS-n15s05_0037835',
 'iDR4_3_SPLUS-n15s19_0049168',
 'iDR4_3_SPLUS-n15s20_0043589',
 'iDR4_3_SPLUS-n17s20_0011455',
 'iDR4_3_SPLUS-n17s21_0060665',
 'iDR4_3_SPLUS-n17s21_0053822',
 'iDR4_3_SPLUS-n16s21_0006990',
 'iDR4_3_SPLUS-n09s38_0057827',
 'iDR4_3_SPLUS-s20s23_0025387',
 'iDR4_3_SPLUS-s45s41_0027231',
 'iDR4_3_SPLUS-s24s09_0027219',
 'iDR4_3_SPLUS-s25s10_0025796',
 'iDR4_3_SPLUS-s20s23_0041616',
 'iDR4_3_SPLUS-s20s23_0029315',
 'iDR4_3_STRIPE82-0005_0021998',
 'iDR4_3_SPLUS-n02s38_0008843',
 'iDR4_3_SPLUS-n01s23_0026692',
 'iDR4_3_SPLUS-n01s37_0009556',
 'iDR4_3_STRIPE82-0105_0023811',
 'iDR4_3_STRIPE82-0059_0016073',
 'iDR4_3_STRIPE82-0104_0005639',
 'iDR4_3_STRIPE82-0014_0039180']

x_values = ["ISO","AUTO","PETRO"]
columns_filter = ['Object', 'J0660_iso', 'J0660_auto', 'J0660_petro', 'e_J0660_iso', 'e_J0660_auto', 'e_J0660_petro']
df_J0660 = pd.DataFrame(columns = columns_filter)
table_to_skip = [1, 4, 5, 7, 8, 10, 14, 15, 21, 22, 23, 24, 26, 29, 31, 34, 38, 41, 42, 43, 45, 48, 49, 55, 57, 59, 61, 62, 63, 65, 68, 71, 73, 74, 75, 80, 81, 83, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 96, 97, 99, 100, 107, 111, 112, 115]

iso_mags=[]
iso_errors=[]
auto_mags=[]
auto_errors=[]
petro_mags=[]
petro_errors=[]

plt.figure(figsize=(10,6))
for table, obj in enumerate(objs_DT_ID):
    if table in table_to_skip:
        continue
    iso_row = iso_list[table][iso_list[table]['ID']==obj][0]
    auto_row = auto_list[table][auto_list[table]['ID']==obj][0]
    petro_row = petro_list[table][petro_list[table]['ID']==obj][0]
    
    mag_values=[]
    error_values=[]
    mag_values.append(iso_row[f'{filter_select}_iso'])
    error_values.append(iso_row[f'e_{filter_select}_iso'])
    mag_values.append(auto_row[f'{filter_select}_auto'])
    error_values.append(auto_row[f'e_{filter_select}_auto'])
    mag_values.append(petro_row[f'{filter_select}_petro'])
    error_values.append(petro_row[f'e_{filter_select}_petro'])
    
    iso_mags.append(iso_row[f'{filter_select}_iso'])
    auto_mags.append(auto_row[f'{filter_select}_auto'])
    petro_mags.append(petro_row[f'{filter_select}_petro'])
    iso_errors.append(iso_row[f'e_{filter_select}_iso'])
    auto_errors.append(auto_row[f'e_{filter_select}_auto'])
    petro_errors.append(petro_row[f'e_{filter_select}_petro'])

    df2 = pd.DataFrame([[loaded_gal_sample["Object"].iloc[table], mag_values[0], mag_values[1], mag_values[2], error_values[0], error_values[1], error_values[2]]],columns=columns_filter)
    df_J0660 = pd.concat([df_J0660, df2],ignore_index=True)

iso_mean = sum(iso_mags) / len(iso_mags)
auto_mean = sum(auto_mags) / len(auto_mags)
petro_mean = sum(petro_mags) / len(petro_mags)

iso_error_mean = sum(iso_errors) / len(iso_errors)
auto_error_mean = sum(auto_errors) / len(auto_errors)
petro_error_mean = sum(petro_errors) / len(petro_errors)

plt.errorbar(['ISO'], [iso_mean], yerr=[iso_error_mean], fmt='o', capsize=5, color='green', label='ISO')
plt.errorbar(['AUTO'], [auto_mean], yerr=[auto_error_mean], fmt='o', capsize=5, color='red', label='AUTO')
plt.errorbar(['PETRO'], [petro_mean], yerr=[petro_error_mean], fmt='o', capsize=5, color='blue', label='PETRO')

plt.grid(linestyle='--', axis='y')
plt.ylabel("Mag (AB)")
plt.title("Filtro J0660 - Hα")
plt.legend()
plt.show()

# %%

columns = ['J0660_iso', 'J0660_auto', 'J0660_petro', 'e_J0660_iso', 'e_J0660_auto', 'e_J0660_petro']
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df_J0660[columns].quantile(0.25)
Q3 = df_J0660[columns].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1
outliers = ((df_J0660[columns] < (Q1 - 1.5 * IQR)) | (df_J0660[columns] > (Q3 + 1.5 * IQR))).any(axis=1)
outlier_df_J0660 = df_J0660[outliers]
df_J0660_without_outliers = df_J0660[~outliers]

# %%

outlier_df_J0660.boxplot()

# %%

df_J0660_without_outliers.boxplot()

# %%

df_J0660_without_outliers[['e_J0660_iso','e_J0660_auto','e_J0660_petro']].boxplot()

# %%

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_J0660[columns])
df_J0660['pca1'] = pca_result[:,0]
df_J0660['pca2'] = pca_result[:,1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca1', y='pca2', data=df_J0660, color='purple')
plt.title('Análise de componentes principais (PCA)')
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_J0660['cluster'] = kmeans.fit_predict(df_J0660[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0660_iso', y='e_J0660_iso', color='green', data=df_J0660)
plt.show()

#%%

kmeans = KMeans(n_clusters=3)
df_J0660['cluster'] = kmeans.fit_predict(df_J0660[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0660_auto', y='e_J0660_auto', color='red', data=df_J0660)
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_J0660['cluster'] = kmeans.fit_predict(df_J0660[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0660_petro', y='e_J0660_petro', color='blue', data=df_J0660)
plt.show()

#%%

g = sns.JointGrid(x="J0660_iso", y="e_J0660_iso", data=df_J0660) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="J0660_auto", y="e_J0660_auto", data=df_J0660) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="J0660_petro", y="e_J0660_petro", data=df_J0660) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

x_label = range(len(df_J0660))
fig, ax = plt.subplots(figsize=(24, 6))
trans1 = Affine2D().translate(-0.25, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.25, 0.0) + ax.transData

plt.errorbar(df_J0660['Object'], df_J0660['J0660_iso'], yerr=df_J0660['e_J0660_iso'], fmt='o', capsize=5, color='green', transform=trans1)
plt.errorbar(df_J0660['Object'], df_J0660['J0660_auto'], yerr=df_J0660['e_J0660_auto'], fmt='o', capsize=5, color='red')
plt.errorbar(df_J0660['Object'], df_J0660['J0660_petro'], yerr=df_J0660['e_J0660_petro'], fmt='o', capsize=5, color='mediumblue', transform=trans2)
plt.title("Filtro J0660 - Hα")
plt.legend(labels=['ISO', 'AUTO', 'PETRO'])
plt.xticks(rotation=90, ha='right')
plt.ylabel('Mag (AB)')
plt.show

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0660)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0660.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0660_auto'], row['J0660_iso'], yerr=row['e_J0660_iso'], xerr=row['e_J0660_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0660 - Hα", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(11, 18, 0.25))
plt.yticks(np.arange(11, 18, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %% 

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0660)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0660.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0660_auto'], row['J0660_petro'], yerr=row['e_J0660_petro'], xerr=row['e_J0660_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0660 - Hα", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('PETRO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(11, 17.5, 0.25))
plt.yticks(np.arange(11, 17.5, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0660)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0660.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0660_petro'], row['J0660_iso'], yerr=row['e_J0660_iso'], xerr=row['e_J0660_petro'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0660 - Hα", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('PETRO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(11, 17.5, 0.25))
plt.yticks(np.arange(11, 17.5, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

filter_select='J0861'
objs_DT_ID=['iDR4_3_SPLUS-s38s28_0037355',
 'iDR4_3_SPLUS-s42s26_0009603',
 'iDR4_3_SPLUS-s37s33_0029178',
 'iDR4_3_SPLUS-n14s32_0029533',
 'iDR4_3_SPLUS-s19s23_0054919',
 'iDR4_3_SPLUS-s20s23_0027579',
 'iDR4_3_SPLUS-s27s07_0017147',
 'iDR4_3_SPLUS-s26s07_0019433',
 'iDR4_3_SPLUS-s21s10_0013331',
 'iDR4_3_SPLUS-s27s10_0013261',
 'iDR4_3_SPLUS-s24s11_0019810',
 'iDR4_3_SPLUS-s26s14_0016119',
 'iDR4_3_SPLUS-s25s18_0032854',
 'iDR4_3_SPLUS-s27s18_0020395',
 'iDR4_3_SPLUS-s27s19_0008651',
 'iDR4_3_SPLUS-s24s20_0018168',
 'iDR4_3_SPLUS-s25s21_0007448',
 'iDR4_3_SPLUS-s25s24_0039312',
 'iDR4_3_SPLUS-s37s20_0007226',
 'iDR4_3_SPLUS-s34s21_0008123',
 'iDR4_3_SPLUS-s24s26_0024907',
 'iDR4_3_SPLUS-s29s31_0019272',
 'iDR4_3_MC0103_0014596',
 'iDR4_3_SPLUS-s25s34_0007124',
 'iDR4_3_SPLUS-s24s35_0029130',
 'iDR4_3_SPLUS-s25s35_0013478',
 'iDR4_3_SPLUS-s30s34_0033827',
 'iDR4_3_SPLUS-s44s24_0025720',
 'iDR4_3_SPLUS-s28s36_0033655',
 'iDR4_3_SPLUS-s43s27_0018255',
 'iDR4_3_SPLUS-s41s30_0038650',
 'iDR4_3_SPLUS-s45s26_0000595',
 'iDR4_3_SPLUS-s38s32_0024249',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s30s39_0009952',
 'iDR4_3_SPLUS-s34s36_0027270',
 'iDR4_3_SPLUS-s37s34_0035918',
 'iDR4_3_SPLUS-s38s34_0035796',
 'iDR4_3_MC0001_0037018',
 'iDR4_3_SPLUS-s30s41_0001982',
 'iDR4_3_SPLUS-s25s45_0034698',
 'iDR4_3_MC0012_0039189',
 'iDR4_3_SPLUS-s39s34_0021319',
 'iDR4_3_SPLUS-s41s32_0034511',
 'iDR4_3_MC0021_0034624',
 'iDR4_3_SPLUS-s29s42_0019922',
 'iDR4_3_SPLUS-s41s33_0032981',
 'iDR4_3_MC0002_0006559',
 'iDR4_3_SPLUS-s36s39_0020464',
 'iDR4_3_SPLUS-n17s01_0006918',
 'iDR4_3_HYDRA-0104_0011186',
 'iDR4_3_HYDRA-0051_0052680',
 'iDR4_3_HYDRA-0015_0018824',
 'iDR4_3_HYDRA-0106_0049457',
 'iDR4_3_HYDRA-0028_0034942',
 'iDR4_3_HYDRA-0085_0056446',
 'iDR4_3_SPLUS-n20s03_0026145',
 'iDR4_3_SPLUS-n19s22_0026754',
 'iDR4_3_SPLUS-n20s23_0017293',
 'iDR4_3_SPLUS-s30s45_0010034',
 'iDR4_3_SPLUS-s20s23_0040961',
 'iDR4_3_SPLUS-s19s23_0047619',
 'iDR4_3_SPLUS-s23s26_0001674',
 'iDR4_3_SPLUS-s23s26_0041931',
 'iDR4_3_SPLUS-s24s65_0036645',
 'iDR4_3_SPLUS-s45s41_0034849',
 'iDR4_3_SPLUS-s23s46_0035512',
 'iDR4_3_SPLUS-s24s74_0016239',
 'iDR4_3_STRIPE82-0063_0034198',
 'iDR4_3_SPLUS-n13s31_0042282',
 'iDR4_3_SPLUS-n14s32_0041298',
 'iDR4_3_SPLUS-s39s31_0013592',
 'iDR4_3_SPLUS-s37s33_0031325',
 'iDR4_3_SPLUS-s21s23_0012649',
 'iDR4_3_HYDRA-0045_0074200',
 'iDR4_3_SPLUS-s34s39_0021318',
 'iDR4_3_SPLUS-s25s35_0015877',
 'iDR4_3_SPLUS-s25s13_0012636',
 'iDR4_3_MC0125_0008151',
 'iDR4_3_MC0097_0015643',
 'iDR4_3_SPLUS-s46s24_0023799',
 'iDR4_3_SPLUS-s46s24_0023407',
 'iDR4_3_SPLUS-s46s25_0034017',
 'iDR4_3_SPLUS-s42s30_0018253',
 'iDR4_3_SPLUS-s42s31_0030929',
 'iDR4_3_SPLUS-s42s32_0018085',
 'iDR4_3_SPLUS-s38s32_0022611',
 'iDR4_3_SPLUS-s38s33_0011808',
 'iDR4_3_SPLUS-s25s13_0017486',
 'iDR4_3_SPLUS-s29s42_0028164',
 'iDR4_3_SPLUS-s26s45_0031867',
 'iDR4_3_HYDRA-0107_0006292',
 'iDR4_3_SPLUS-s24s09_0028206',
 'iDR4_3_SPLUS-s25s10_0026796',
 'iDR4_3_SPLUS-s21s22_0024741',
 'iDR4_3_SPLUS-n15s05_0037835',
 'iDR4_3_SPLUS-n15s19_0049168',
 'iDR4_3_SPLUS-n15s20_0043589',
 'iDR4_3_SPLUS-n17s20_0011455',
 'iDR4_3_SPLUS-n17s21_0060665',
 'iDR4_3_SPLUS-n17s21_0053822',
 'iDR4_3_SPLUS-n16s21_0006990',
 'iDR4_3_SPLUS-n09s38_0057827',
 'iDR4_3_SPLUS-s20s23_0025387',
 'iDR4_3_SPLUS-s45s41_0027231',
 'iDR4_3_SPLUS-s24s09_0027219',
 'iDR4_3_SPLUS-s25s10_0025796',
 'iDR4_3_SPLUS-s20s23_0041616',
 'iDR4_3_SPLUS-s20s23_0029315',
 'iDR4_3_STRIPE82-0005_0021998',
 'iDR4_3_SPLUS-n02s38_0008843',
 'iDR4_3_SPLUS-n01s23_0026692',
 'iDR4_3_SPLUS-n01s37_0009556',
 'iDR4_3_STRIPE82-0105_0023811',
 'iDR4_3_STRIPE82-0059_0016073',
 'iDR4_3_STRIPE82-0104_0005639',
 'iDR4_3_STRIPE82-0014_0039180']

x_values = ["ISO","AUTO","PETRO"]
columns_filter = ['Object', 'J0861_iso', 'J0861_auto', 'J0861_petro', 'e_J0861_iso', 'e_J0861_auto', 'e_J0861_petro']
df_J0861 = pd.DataFrame(columns = columns_filter)
table_to_skip = [1, 4, 5, 7, 8, 10, 14, 15, 21, 22, 23, 24, 26, 29, 31, 34, 38, 41, 42, 43, 45, 48, 49, 55, 57, 59, 61, 62, 63, 65, 68, 71, 73, 74, 75, 80, 81, 83, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 96, 97, 99, 100, 107, 111, 112, 115]

iso_mags=[]
iso_errors=[]
auto_mags=[]
auto_errors=[]
petro_mags=[]
petro_errors=[]

plt.figure(figsize=(10,6))
for table, obj in enumerate(objs_DT_ID):
    if table in table_to_skip:
        continue
    iso_row = iso_list[table][iso_list[table]['ID']==obj][0]
    auto_row = auto_list[table][auto_list[table]['ID']==obj][0]
    petro_row = petro_list[table][petro_list[table]['ID']==obj][0]
    
    mag_values=[]
    error_values=[]
    mag_values.append(iso_row[f'{filter_select}_iso'])
    error_values.append(iso_row[f'e_{filter_select}_iso'])
    mag_values.append(auto_row[f'{filter_select}_auto'])
    error_values.append(auto_row[f'e_{filter_select}_auto'])
    mag_values.append(petro_row[f'{filter_select}_petro'])
    error_values.append(petro_row[f'e_{filter_select}_petro'])
    
    iso_mags.append(iso_row[f'{filter_select}_iso'])
    auto_mags.append(auto_row[f'{filter_select}_auto'])
    petro_mags.append(petro_row[f'{filter_select}_petro'])
    iso_errors.append(iso_row[f'e_{filter_select}_iso'])
    auto_errors.append(auto_row[f'e_{filter_select}_auto'])
    petro_errors.append(petro_row[f'e_{filter_select}_petro'])

    df2 = pd.DataFrame([[loaded_gal_sample["Object"].iloc[table], mag_values[0], mag_values[1], mag_values[2], error_values[0], error_values[1], error_values[2]]],columns=columns_filter)
    df_J0861 = pd.concat([df_J0861, df2],ignore_index=True)

iso_mean = sum(iso_mags) / len(iso_mags)
auto_mean = sum(auto_mags) / len(auto_mags)
petro_mean = sum(petro_mags) / len(petro_mags)

iso_error_mean = sum(iso_errors) / len(iso_errors)
auto_error_mean = sum(auto_errors) / len(auto_errors)
petro_error_mean = sum(petro_errors) / len(petro_errors)

plt.errorbar(['ISO'], [iso_mean], yerr=[iso_error_mean], fmt='o', capsize=5, color='green', label='ISO')
plt.errorbar(['AUTO'], [auto_mean], yerr=[auto_error_mean], fmt='o', capsize=5, color='red', label='AUTO')
plt.errorbar(['PETRO'], [petro_mean], yerr=[petro_error_mean], fmt='o', capsize=5, color='blue', label='PETRO')

plt.grid(linestyle='--', axis='y')
plt.ylabel("Mag (AB)")
plt.title("Filtro J0861 - Ca Triplet")
plt.legend()
plt.show()

# %%

columns = ['J0861_iso', 'J0861_auto', 'J0861_petro', 'e_J0861_iso', 'e_J0861_auto', 'e_J0861_petro']
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df_J0861[columns].quantile(0.25)
Q3 = df_J0861[columns].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1
outliers = ((df_J0861[columns] < (Q1 - 1.5 * IQR)) | (df_J0861[columns] > (Q3 + 1.5 * IQR))).any(axis=1)
outlier_df_J0861 = df_J0861[outliers]
df_J0861_without_outliers = df_J0861[~outliers]

# %%

outlier_df_J0861.boxplot()

# %%

df_J0861_without_outliers.boxplot()

# %%

df_J0861_without_outliers[['e_J0861_iso','e_J0861_auto','e_J0861_petro']].boxplot()

# %%

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_J0861[columns])
df_J0861['pca1'] = pca_result[:,0]
df_J0861['pca2'] = pca_result[:,1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca1', y='pca2', data=df_J0861, color='purple')
plt.title('Análise de componentes principais (PCA)')
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_J0861['cluster'] = kmeans.fit_predict(df_J0861[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0861_iso', y='e_J0861_iso', color='green', data=df_J0861)
plt.show()

#%%

kmeans = KMeans(n_clusters=3)
df_J0861['cluster'] = kmeans.fit_predict(df_J0861[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0861_auto', y='e_J0861_auto', color='red', data=df_J0861)
plt.show()

# %%

kmeans = KMeans(n_clusters=3)
df_J0861['cluster'] = kmeans.fit_predict(df_J0861[columns])

plt.figure(figsize=(10, 6))
sns.scatterplot(x='J0861_petro', y='e_J0861_petro', color='blue', data=df_J0861)
plt.show()

#%%

g = sns.JointGrid(x="J0861_iso", y="e_J0861_iso", data=df_J0861) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="J0861_auto", y="e_J0861_auto", data=df_J0861) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

g = sns.JointGrid(x="J0861_petro", y="e_J0861_petro", data=df_J0861) 
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

# %%

x_label = range(len(df_J0861))
fig, ax = plt.subplots(figsize=(24, 6))
trans1 = Affine2D().translate(-0.25, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.25, 0.0) + ax.transData

plt.errorbar(df_J0861['Object'], df_J0861['J0861_iso'], yerr=df_J0861['e_J0861_iso'], fmt='o', capsize=5, color='green', transform=trans1)
plt.errorbar(df_J0861['Object'], df_J0861['J0861_auto'], yerr=df_J0861['e_J0861_auto'], fmt='o', capsize=5, color='red')
plt.errorbar(df_J0861['Object'], df_J0861['J0861_petro'], yerr=df_J0861['e_J0861_petro'], fmt='o', capsize=5, color='mediumblue', transform=trans2)
plt.title("Filtro J0861 - Ca Triplet")
plt.legend(labels=['ISO', 'AUTO', 'PETRO'])
plt.xticks(rotation=90, ha='right')
plt.ylabel('Mag (AB)')
plt.show

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0861)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0861.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0861_auto'], row['J0861_iso'], yerr=row['e_J0861_iso'], xerr=row['e_J0861_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0861 - Ca Triplet", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(10.5, 17.25, 0.25))
plt.yticks(np.arange(10.5, 17.25, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %% 

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0861)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0861.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0861_auto'], row['J0861_petro'], yerr=row['e_J0861_petro'], xerr=row['e_J0861_auto'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0861 - Ca Triplet", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('AUTO (Mag AB)')
plt.ylabel('PETRO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(10.5, 17.5, 0.25))
plt.yticks(np.arange(10.5, 17.5, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%

color_obj = ['darkturquoise', 'grey', 'rebeccapurple', 'navajowhite', 'darkcyan', 'violet', 'tan', 'tomato', 'blue', 'deeppink', 'blueviolet', 'brown', 'orchid', 'dodgerblue', 'goldenrod', 'pink', 'red', 'gold', 'cadetblue', 'darkkhaki', 'olive', 'salmon', 'yellowgreen', 'orangered', 'darkolivegreen', 'slategray', 'chartreuse', 'magenta', 'darkseagreen', 'chocolate', 'palegreen', 'steelblue', 'sandybrown', 'peru', 'mediumturquoise', 'darkviolet']
x_label = list(range(len(df_J0861)))
fig, ax = plt.subplots(figsize=(10, 12))

for i, row in df_J0861.iterrows():
    color = color_obj[i % len(color_obj)]
    plt.errorbar(row['J0861_petro'], row['J0861_iso'], yerr=row['e_J0861_iso'], xerr=row['e_J0861_petro'], fmt='o', capsize=5, label=f"{row['Object']} - {i}", color=color)

plt.title("Filtro J0861 - Ca Triplet", fontsize=19)
plt.xticks(rotation=90, ha='right')
plt.xlabel('PETRO (Mag AB)')
plt.ylabel('ISO (Mag AB)')
plt.grid(linestyle='--')
plt.xticks(np.arange(10.5, 17.5, 0.25))
plt.yticks(np.arange(10.5, 17.5, 0.25))
#plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', title='Galaxy')
plt.tight_layout()
plt.show()

# %%