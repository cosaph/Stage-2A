# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    indice_perf_all_playersM21.py                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ccottet <ccottet@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/05 11:01:41 by ccottet           #+#    #+#              #
#    Updated: 2024/06/05 14:57:45 by ccottet          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

## ------ Importing Libraries ------ ##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# chargez tous les csv 

match_played = pd.read_csv("/Users/cosaph/Desktop/fc servette/Coralie/Match_player/Export Servette Pro - all games.csv")
match_played_M21 = pd.read_csv("/Users/cosaph/Desktop/fc servette/Coralie/Match_player/Servette Academy M21 - Match 23_24.csv")

all_training = pd.read_csv("/Users/cosaph/Desktop/fc servette/Coralie/Export Servette Pro - all training.csv")
all_training_M21 = pd.read_csv("/Users/cosaph/Desktop/fc servette/Coralie/Servette Academy M21 - Training 23_24.csv")

# merging the csv to obtain the names 

correspondance_M21 = pd.read_csv("/Users/cosaph/Desktop/fc servette/Coralie/corespondance/Servette Academy M21 - Correspondance.csv")

# merge player id et name M21

match_played_M21 = pd.merge(match_played_M21, correspondance_M21, on='PLAYER_ID')
all_training_M21 = pd.merge(all_training_M21, correspondance_M21, on='PLAYER_ID')

# ******************************---M21---*********************************** #

df_21 = pd.DataFrame()

df_21['Max_speed'] = match_played_M21.groupby('PLAYER_NAME')['Vmax'].max()
df_21['Accélération max'] = match_played_M21.groupby('PLAYER_NAME')['Peak accel'].max()

matchs_21 = match_played_M21[match_played['DURATION'] > 60]

distance_mmin_21 = matchs_21.groupby('PLAYER_NAME')['Distance m/min'].max()

df_21['Distance_m/min'] = distance_mmin_21

players_21 = all_training_M21['PLAYER_NAME'].unique()
nombre_train21 = all_training_M21['DATE'].nunique()

for player in players_21:
    attendance_player = all_training_M21[all_training_M21['PLAYER_NAME'] == player]['ATTENDANCE'].value_counts()
    
    if isinstance(attendance_player, pd.Series):
        attendance_player = attendance_player.reindex(['PRESENT', 'ADAPTE', 'BLESSE', 'READAPTATION', 'REPOS', 'EN SOIN', 'MALADE', 'ABSENT'], fill_value=0)
    
    attendance_player_sum = attendance_player.sum()

    present_player = attendance_player['PRESENT']
    
    pourcentage_present = (present_player / attendance_player_sum) * 100 if attendance_player_sum > 0 else 0

    df_21.loc[player, 'Attendance'] = pourcentage_present

# maintenant on va calculer un z-score pour chacune des variables
    
df_21['PLAYER_NAME'] = df_21.index
df_21['Max_speed_z'] = (df_21['Max_speed'] - df_21['Max_speed'].mean()) / df_21['Max_speed'].std()
df_21['Accélération max_z'] = (df_21['Accélération max'] - df_21['Accélération max'].mean()) / df_21['Accélération max'].std()
df_21['Distance_m/min_z'] = (df_21['Distance_m/min'] - df_21['Distance_m/min'].mean()) / df_21['Distance_m/min'].std()
df_21['Attendance_z'] = (df_21['Attendance'] - df_21['Attendance'].mean()) / df_21['Attendance'].std()
    
acc_mean_21 = df_21['Accélération max_z'].mean(skipna=True)
speed_mean_21 = df_21['Max_speed_z'].mean(skipna=True)
distance_mean_21 = df_21['Distance_m/min_z'].mean(skipna=True)
attendance_mean_21 = df_21['Attendance_z'].mean(skipna=True)
    
#print(df)
    
#save the csv 

df_21.to_csv("data_M21.csv")
    

# Définition des couleurs
start_color = '#7a303f'
end_color = '#f4eef0'

# Création de la palette
n_colors = 256
colors = [plt.cm.viridis(i) for i in np.linspace(0, 1, n_colors)]

# Interpolation des couleurs
start_rgb = np.array([int(start_color[i:i+2], 16)/255 for i in (1, 3, 5)])
end_rgb = np.array([int(end_color[i:i+2], 16)/255 for i in (1, 3, 5)])
new_colors = np.array([mcolors.to_rgb(mcolors.rgb2hex(start_rgb + (end_rgb - start_rgb) * i / (n_colors - 1))) for i in range(n_colors)])

# Calcul du z-score total en ignorant les valeurs nulles
df_21['Total_z'] = df_21[['Max_speed_z', 'Accélération max_z', 'Distance_m/min_z', 'Attendance_z']].sum(axis=1, skipna=True)

# On trie les joueurs par leur z-score total en faisant un barre plot

df = df_21.sort_values('Total_z', ascending=False)

start_color = '#85142B'
end_color = '#FFFFF0'

start_rgb = np.array(mcolors.hex2color(start_color))
end_rgb = np.array(mcolors.hex2color(end_color))


n_colors = len(df)
colors = [mcolors.to_hex(start_rgb + (end_rgb - start_rgb) * i / (n_colors - 1)) for i in range(n_colors)]

plt.figure(figsize=(8, 8))
sns.barplot(x='Total_z', y=df.index, data=df, palette=colors[::-1])
plt.xlabel('Total z-score', color='white')
plt.ylabel('Players', color='white')
plt.title('Total z-score of Players', color='white')
plt.gca().set_facecolor('black')
plt.gcf().set_facecolor('black')
plt.xticks(color='white')
plt.yticks(color='white')
plt.show()
