# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    indice_perf_all_playerM19.py                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ccottet <ccottet@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/05 11:01:41 by ccottet           #+#    #+#              #
#    Updated: 2024/06/05 14:55:59 by ccottet          ###   ########.fr        #
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
match_played_M19 = pd.read_csv("/Users/cosaph/Desktop/fc servette/Coralie/Match_player/Servette Academy M19 - Match 23_24.csv")
match_played_M21 = pd.read_csv("/Users/cosaph/Desktop/fc servette/Coralie/Match_player/Servette Academy M21 - Match 23_24.csv")

all_training = pd.read_csv("/Users/cosaph/Desktop/fc servette/Coralie/Export Servette Pro - all training.csv")
all_training_M19 = pd.read_csv("/Users/cosaph/Desktop/fc servette/Coralie/Servette Academy M19 - Training 23_24.csv")
all_training_M21 = pd.read_csv("/Users/cosaph/Desktop/fc servette/Coralie/Servette Academy M21 - Training 23_24.csv")

# merging the csv to obtain the names 

correspondance_M19 = pd.read_csv("/Users/cosaph/Desktop/fc servette/Coralie/corespondance/Servette Academy M19 - Correspondance.csv")

# merge player id et name M19

match_played_M19 = pd.merge(match_played_M19, correspondance_M19, on='PLAYER_ID')
all_training_M19 = pd.merge(all_training_M19, correspondance_M19, on='PLAYER_ID')

# ******************************---M19---*********************************** #

df_19 = pd.DataFrame()

df_19['Max_speed'] = match_played_M19.groupby('PLAYER_NAME')['Vmax'].max()
df_19['Accélération max'] = match_played_M19.groupby('PLAYER_NAME')['Peak accel'].max()

matchs_19 = match_played_M19[match_played['DURATION'] > 60]

distance_mmin_19 = matchs_19.groupby('PLAYER_NAME')['Distance m/min'].max()

df_19['Distance_m/min'] = distance_mmin_19


players_19 = all_training_M19['PLAYER_NAME'].unique()
nombre_train19 = all_training_M19['DATE'].nunique()

for player in players_19:
    attendance_player = all_training_M19[all_training_M19['PLAYER_NAME'] == player]['ATTENDANCE'].value_counts()
    
    if isinstance(attendance_player, pd.Series):
        attendance_player = attendance_player.reindex(['PRESENT', 'ADAPTE', 'BLESSE', 'READAPTATION', 'REPOS', 'EN SOIN', 'MALADE', 'ABSENT'], fill_value=0)
    
    attendance_player_sum = attendance_player.sum()

    present_player = attendance_player['PRESENT']
    
    pourcentage_present = (present_player / attendance_player_sum) * 100 if attendance_player_sum > 0 else 0


    df_19.loc[player, 'Attendance'] = pourcentage_present

# maintenant on va calculer un z-score pour chacune des variables
    
df_19['PLAYER_NAME'] = df_19.index
df_19['Max_speed_z'] = (df_19['Max_speed'] - df_19['Max_speed'].mean()) / df_19['Max_speed'].std()
df_19['Accélération max_z'] = (df_19['Accélération max'] - df_19['Accélération max'].mean()) / df_19['Accélération max'].std()
df_19['Distance_m/min_z'] = (df_19['Distance_m/min'] - df_19['Distance_m/min'].mean()) / df_19['Distance_m/min'].std()
df_19['Attendance_z'] = (df_19['Attendance'] - df_19['Attendance'].mean()) / df_19['Attendance'].std()
    
acc_mean_19 = df_19['Accélération max_z'].mean(skipna=True)
speed_mean_19 = df_19['Max_speed_z'].mean(skipna=True)
distance_mean_19 = df_19['Distance_m/min_z'].mean(skipna=True)
attendance_mean_19 = df_19['Attendance_z'].mean(skipna=True)
    
#print(df)
    
#save the csv 

df_19.to_csv("data_M19.csv")
    


# Calcul du z-score total en ignorant les valeurs nulles
df_19['Total_z'] = df_19[['Max_speed_z', 'Accélération max_z', 'Distance_m/min_z', 'Attendance_z']].sum(axis=1, skipna=True)

# On trie les joueurs par leur z-score total en faisant un barre plot

df = df_19.sort_values('Total_z', ascending=False)

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




