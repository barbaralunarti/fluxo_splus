# %%

%pip install pyerfa
%pip install git+https://github.com/astropy/pyregion.git
%pip install splusdata==4.0
%pip install aplpy==2.1.0
%pip install astropy==5.3.4
%pip install pandas
%pip install numpy==1.26.2
%pip install matplotlib
%pip install setuptools==69.0.3

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

# %%

conn = splusdata.connect('username','password')

# %%

loaded_gal_sample = pd.read_pickle('gal_sample.pkl')

# %%

# Query Splus using the sample data

querytable = []

for i in range(0, len(loaded_gal_sample)):
    inp_ra = loaded_gal_sample['RA_ICRS'].iloc[i]
    inp_dec = loaded_gal_sample['DEC_ICRS'].iloc[i]
    radius = (60.0 * u.arcsec).to(u.degree).value

    query = f"""
    SELECT
    det.ID, det.RA, det.DEC, u.u_petro, j0378.J0378_petro, j0395.J0395_petro,
    j0410.J0410_petro, j0430.J0430_petro, g.g_petro, j0515.J0515_petro,
    r.r_petro, j0660.J0660_petro, i.i_petro, j0861.J0861_petro, z.z_petro,
    u.e_u_petro, j0378.e_J0378_petro, j0395.e_J0395_petro, j0410.e_J0410_petro,
    j0430.e_J0430_petro, g.e_g_petro, j0515.e_J0515_petro, r.e_r_petro,
    j0660.e_J0660_petro, i.e_i_petro, j0861.e_J0861_petro, z.e_z_petro
    
    FROM
    idr4_dual.idr4_detection_image AS det
    JOIN idr4_dual.idr4_dual_u AS u ON (det.ID = u.ID)
    JOIN idr4_dual.idr4_dual_j0378 AS j0378 ON (det.ID = j0378.ID)
    JOIN idr4_dual.idr4_dual_j0395 AS j0395 ON (det.ID = j0395.ID)
    JOIN idr4_dual.idr4_dual_j0410 AS j0410 ON (det.ID = j0410.ID)
    JOIN idr4_dual.idr4_dual_j0430 AS j0430 ON (det.ID = j0430.ID)
    JOIN idr4_dual.idr4_dual_g AS g ON (det.ID = g.ID)
    JOIN idr4_dual.idr4_dual_j0515 AS j0515 ON (det.ID = j0515.ID)
    JOIN idr4_dual.idr4_dual_r AS r ON (det.ID = r.ID)
    JOIN idr4_dual.idr4_dual_j0660 AS j0660 ON (det.ID = j0660.ID)
    JOIN idr4_dual.idr4_dual_i AS i ON (det.ID = i.ID)
    JOIN idr4_dual.idr4_dual_j0861 AS j0861 ON (det.ID = j0861.ID)
    JOIN idr4_dual.idr4_dual_z AS z ON (det.ID = z.ID)

    WHERE 1=CONTAINS( POINT('ICRS',{inp_ra}, {inp_dec}), CIRCLE('ICRS', det.RA, det.DEC, {radius}) )
    """

    querytable.append(conn.query(query))

# %%

for i in range(0,len(querytable)-1):
    with open(f'{loaded_gal_sample["Object"].iloc[i]}.pkl', mode='wb+') as file:
        pickle.dump(querytable[i],file)

# %%

loaded_list=[]
for i in range(0,len(loaded_gal_sample)-1):
    with open(f'{loaded_gal_sample["Object"].iloc[i]}.pkl', mode='rb') as file:
        loaded_list.append(pickle.load(file))

# %%

obj_index = 0
select_table = loaded_list[obj_index] # select the table of the galaxy (object)
table_index = 10 # select the index of the table
select_table

# %%

# PETRO aperture
# Calculating flux intensity in each filter

point_colors = ['mediumblue', 'mediumorchid', 'blue', 'royalblue', 'deepskyblue', 'darkturquoise', 'springgreen', 'limegreen', 'gold', 'orange', 'orangered', 'red']

eixo_x = np.array([0.3563, 0.377, 0.394, 0.4094, 0.4292, 0.4751, 0.5133, 0.6258, 0.6614, 0.7690, 0.8611, 0.8831])
dados_index = select_table[table_index]

colunas_y = ['u_petro', 'J0378_petro', 'J0395_petro', 'J0410_petro', 'J0430_petro', 'g_petro', 'J0515_petro', 'r_petro', 'J0660_petro', 'i_petro', 'J0861_petro', 'z_petro']

valores_y = np.array([dados_index[col] for col in colunas_y])
flux_jansky = 10**(-0.4 * (valores_y - 8.9))
flux_lambda = (3 * (10**(-13))) * (flux_jansky / eixo_x)  # flux in erg/s/cm**2/A units

fig, ax = plt.subplots()
ax.grid(which='both', linestyle='--')
ax.set_xlim(0.3, 0.95)
xticks = np.arange(0.3, 0.91, 0.1)

for i, col in enumerate(colunas_y):
    ax.scatter(eixo_x[i], flux_jansky[i], label=col, color=point_colors[i])

legend_labels = ['u', 'J0378', 'J0395', 'J0410', 'J0430', 'g', 'J0515', 'r', 'J0660', 'i', 'J0861', 'z']
plt.xlabel(r'$\lambda_{\mathrm{eff}}$ (µm)')
plt.ylabel(r'$F_{\lambda}$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)')
legend = plt.legend(legend_labels, title='Filter', loc="upper left", bbox_to_anchor=(1, 1))

legend.get_title().set_fontsize('12')

plt.title('Spectral Energy Distribution (SED)')
plt.show()

# %%

# PETRO aperture
# Comparing the flux of the central region (nucleus) and the external region (ring) in each filter 

eixo_x = np.array([0.3563, 0.377, 0.394, 0.4094, 0.4292, 0.4751, 0.5133, 0.6258, 0.6614, 0.7690, 0.8611, 0.8831])
dados_index_6 = select_table[6]
dados_index_37 = select_table[37]

colunas_y = ['u_petro', 'J0378_petro', 'J0395_petro', 'J0410_petro', 'J0430_petro', 'g_petro', 'J0515_petro', 'r_petro', 'J0660_petro', 'i_petro', 'J0861_petro', 'z_petro']
valores_y_index_6 = np.array([dados_index_6[col] for col in colunas_y])
valores_y_index_37 = np.array([dados_index_37[col] for col in colunas_y])

flux_jansky_index_6 = 10**(-0.4 * (valores_y_index_6 - 8.9))
flux_jansky_index_37 = 10**(-0.4 * (valores_y_index_37 - 8.9))

flux_lambda_index_6 = (3 * (10**(-13))) * (flux_jansky_index_6 / eixo_x) # flux in erg/s/cm**2/A units
flux_lambda_index_37 = (3 * (10**(-13))) * (flux_jansky_index_37 / eixo_x) # flux in erg/s/cm**2/A units

fig, ax = plt.subplots()
ax.grid(which='both', linestyle='--')

plt.xlabel(r'$\lambda_{\mathrm{eff}}$ (µm)')
plt.ylabel(r'$F_{\lambda}$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)')
plt.title('Spectral Energy Distribution (SED)')

# Plot the data for index 6
ax.plot(eixo_x, flux_lambda_index_6, marker='o', linestyle='None', label='Ring', color='dodgerblue')

# Plot the data for index 37
ax.plot(eixo_x, flux_lambda_index_37, marker='o', linestyle='None', label='Nucleus', color='red')

plt.legend()
plt.show()

# %%


