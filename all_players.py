# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    all_players.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ccottet <ccottet@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/03 13:53:57 by ccottet           #+#    #+#              #
#    Updated: 2024/06/05 14:03:46 by ccottet          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# Refaire le graph en demandant via l'interface de commande quel joueur on veut voir

# Importation des modules

import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from PIL import Image
import seaborn as sns
from mplsoccer import PyPizza, FontManager
from matplotlib.patches import Arc

from sklearn.linear_model import LinearRegression

# Importation des données

all_training = pd.read_csv("/Users/cosaph/Desktop/fc servette/Coralie/Export Servette Pro - all training.csv")
match_played = pd.read_csv("/Users/cosaph/Desktop/fc servette/Coralie/Match_player/Export Servette Pro - all games.csv")

nb_joueur = all_training['PLAYER_NAME'].nunique()

# On demande à l'utilisateur quel joueur il veut voir

player = input("Which player do you want to see ? ")

#calculate the highest speed for each player

max_speed_player = match_played.groupby('PLAYER_NAME')['Vmax'].max().reset_index()
max_speed_player = max_speed_player.sort_values(by='Vmax', ascending=False).reset_index(drop=True)
max_speed_player['Rank'] = max_speed_player.index + 1

rank_player = max_speed_player[max_speed_player['PLAYER_NAME'] == player ].index[0] + 1



##################################################################################################
# ---------------------------------- PRÉSENCE À L'ENTRAINEMENT --------------------------------- #
# calcule de présence à l'entraînement sur tous les entrainements.

nombre_train = all_training['DATE'].nunique()

attendance = all_training['ATTENDANCE'].value_counts()


attendance_player = all_training[all_training['PLAYER_NAME'] == player ]['ATTENDANCE'].value_counts()

if isinstance(attendance_player, pd.Series):
    attendance_player = attendance_player.reindex(['PRESENT', 'ADAPTE', 'BLESSE', 'READAPTATION', 'REPOS', 'EN SOIN', 'MALADE', 'ABSENT'], fill_value=0)

# Calculer la somme des présences
attendance_player_sum = 0
for col in ['PRESENT', 'ADAPTE', 'BLESSE', 'READAPTATION', 'REPOS', 'EN SOIN', 'MALADE', 'ABSENT']:
    try:
        attendance_player_sum += attendance_player[col]
    except (KeyError, TypeError):
        print(f"La colonne '{col}' est absente dans les données d'attendance pour le joueur {player}.")

# Récupérer le nombre de présences
try:
    present_player = attendance_player['PRESENT']
except (KeyError, TypeError):
    print(f"La colonne 'PRESENT' est absente dans les données d'attendance pour le joueur {player}.")
    present_player = 'N/A'

pourcentage_present = (present_player / attendance_player_sum) * 100

###################################################################################################



#on calcule le pourcentage de présence d'antunes

# Nombre de matchs joués par le joueur

nombre_match = match_played['DATE'].nunique()

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Nombre de minutes à l'entrainement 

# TO DO ----------------------------#

# TRAINING

############################################################################################################

############################################################################################################

total_distance_train = all_training[all_training['PLAYER_NAME'] == player ]['Distance totale'].sum()

total_distance_train_moy = all_training[all_training['PLAYER_NAME'] == player]['Distance totale'].mean()

distance_per_min_train = all_training[all_training['PLAYER_NAME'] == player]['Distance m/min'].mean()
nb_sprint_train  = all_training[all_training['PLAYER_NAME'] == player]['Nombre de sprints'].sum()



vitesse_max_moyenne_train = all_training[all_training['PLAYER_NAME'] == player]['Vmax'].mean()
high_speed_distance_train = all_training[all_training['PLAYER_NAME'] == player]['Distance [25-40]km/h'].sum()

total_minute_train = total_distance_train_moy / distance_per_min_train


# calcul du nombre de fois à l'entraînement ou il a atteint 90% de sa vitesse_max_match

vitesse_max_match = match_played[match_played['PLAYER_NAME'] == player]['Vmax'].max()
#print(vitesse_max_match)

nombre_train_90_vmax = 0
for i in range(len(all_training)):
    if all_training['PLAYER_NAME'][i] == player:
        if all_training['Vmax'][i] >= 0.9 * vitesse_max_match:
            nombre_train_90_vmax += 1
nombre_train_90_vmax= round(nombre_train_90_vmax, 2)


nombre_train_85_vmax = 0
for i in range(len(all_training)):
    if all_training['PLAYER_NAME'][i] == player:
        if all_training['Vmax'][i] >= 0.85 * vitesse_max_match:
            nombre_train_85_vmax += 1
nombre_train_85_vmax= round(nombre_train_85_vmax, 2)


nombre_train_inferier_85_vmax = 0
for i in range(len(all_training)):
    if all_training['PLAYER_NAME'][i] == player:
        if all_training['Vmax'][i] < 0.85 * vitesse_max_match:
            nombre_train_inferier_85_vmax += 1
nombre_train_inferier_70_vmax= round(nombre_train_inferier_85_vmax, 2)

# MATCH

#'PLAYER_NAME' == player' attendance

#print(all_training['PLAYER_NAME'].unique())

# MATCH
starter_matches = match_played[(match_played['PLAYER_NAME'] == player) & (match_played['ATTENDANCE'] == 'TITULAIRE')]
attendance_titulaire = starter_matches.shape[0]
attendance_match_remplaçant = match_played[(match_played['PLAYER_NAME'] == player) & (match_played['ATTENDANCE'] == 'REMPLACANT')].shape[0]
nombre_match_absent = nombre_match - (attendance_titulaire + attendance_match_remplaçant)

#nombre_match = attendance_titulaire + attendance_match_remplaçant
total_distance_match_moy = match_played[match_played['PLAYER_NAME'] == player]['Distance totale'].mean()
distance_per_min_match = match_played[match_played['PLAYER_NAME'] == player]['Distance m/min'].mean()
nb_sprint_match  = match_played[match_played['PLAYER_NAME'] == player]['Nombre de sprints'].sum()
vitesse_max_moyenne_match = match_played[match_played['PLAYER_NAME'] == player]['Vmax'].mean()
high_speed_distance_match = match_played[match_played['PLAYER_NAME'] == player]['Distance [25-40]km/h'].sum()
distance_total_match = match_played[match_played['PLAYER_NAME'] == player]['Distance totale'].sum()
""" total_minutes_all_match_ = distance_total_match / distance_per_min_match """
vitesse_max_match = match_played[match_played['PLAYER_NAME'] == player]['Vmax'].max()
total_minute_match = match_played[match_played['PLAYER_NAME'] == player]['DURATION'].sum()

############################################################################################################

############# ------------------------- TOTAL MINUTES GAME RATIO  -------------------------#################

# pour une date on va prendre le max de minutes jouées par un joueur

max_minute = match_played.groupby('DATE')['DURATION'].max()
max_minutes = max_minute.sum()

total_minutes_all_match_player = match_played[match_played['PLAYER_NAME'] == player]['DURATION'].sum()

ratio_minutes = total_minutes_all_match_player/max_minutes * 100

############################################################################################################


# j'ai que les dates au format 2023-07-22 18:00:00 donc faire un arrangement par mois puis apres faire le calcul de distance totale en moyenne par moiss

all_training['DATE'] = pd.to_datetime(all_training['DATE'])

all_training['MONTH'] = all_training['DATE'].dt.month

distance_moy_month_train = all_training[all_training['PLAYER_NAME'] == player].groupby('MONTH')['Distance totale'].mean()

# Pour les matchs

match_played['DATE'] = pd.to_datetime(match_played['DATE'])
match_played['MONTH'] = match_played['DATE'].dt.month
distance_moy_month_match = match_played[match_played['PLAYER_NAME'] == player].groupby('MONTH')['Distance totale'].mean()

#print(f"\nDistance moyenne par mois en match pour A.Antunes: \n{distance_moy_month_match}")

# TOP SPEED PER MONTHS

top_speed_month_train = all_training[all_training['PLAYER_NAME'] == player].groupby('MONTH')['Vmax'].max()


top_speed_month_match = match_played[match_played['PLAYER_NAME'] == player].groupby('MONTH')['Vmax'].max()


#calcul de max acceleration max speed et max deceleration

# EN MATCH 
max_acc_match = match_played[match_played['PLAYER_NAME'] == player]['Accélération max'].max()
#print(max_acc_match)

max_decc_match = match_played[match_played['PLAYER_NAME'] == player]['Décélération max'].max()
#print(max_decc_match)

# EN TRAIN

max_acc_train = all_training[all_training['PLAYER_NAME'] == player]['Accélération max'].max()
##print(max_acc_train)
max_decc_train = all_training[all_training['PLAYER_NAME'] == player]['Décélération max'].max()
#print(max_decc_train)

# Nombre d'accélation/decceleration par match  et entraînement en moyenne par mois

nb_acc_train = all_training[all_training['PLAYER_NAME'] == player].groupby('MONTH')['Nb acc [4/5] ms²'].mean()
#print(nb_acc_train)

nb_decc_train = all_training[all_training['PLAYER_NAME'] == player].groupby('MONTH')['Nb dec [-5/-4] ms²'].mean()
#print(nb_decc_train)

nb_acc_match = match_played[match_played['PLAYER_NAME'] == player].groupby('MONTH')['Nb acc [4/5] ms²'].mean()
#print(nb_acc_match)

nb_decc_match = match_played[match_played['PLAYER_NAME'] == player].groupby('MONTH')['Nb dec [-5/-4] ms²'].mean()
#print(nb_decc_match)


# nombre de sprint par mois

nb_sprint_train_month = all_training[all_training['PLAYER_NAME'] == player].groupby('MONTH')['Nombre de sprints'].mean()
#print(nb_sprint_train_month)

nb_sprint_match_month = match_played[match_played['PLAYER_NAME'] == player].groupby('MONTH')['Nombre de sprints'].mean()
#print(nb_sprint_match_month)

train_cumulative = np.cumsum(nb_sprint_train_month)
match_cumulative = np.cumsum(nb_sprint_match_month)


# -------- TOTAL DISTANCE PAR MOIS ------------# 
match_played['DATE'] = pd.to_datetime(match_played['DATE'])
all_training['DATE'] = pd.to_datetime(all_training['DATE'])
match_played['MONTH'] = match_played['DATE'].dt.strftime('%m-%Y')
all_training['MONTH'] = all_training['DATE'].dt.strftime('%m-%Y')

distance_train_month = all_training[all_training['PLAYER_NAME'] == player].groupby('MONTH')['Distance totale'].sum()
distance_match_month = match_played[match_played['PLAYER_NAME'] == player].groupby('MONTH')['Distance totale'].sum()


distance_train_month = distance_train_month / 1000
distance_match_month = distance_match_month / 1000

distance_total_month = distance_train_month.add(distance_match_month, fill_value=0)


distance_total_month.index = pd.to_datetime(distance_total_month.index, format='%m-%Y')
distance_total_month = distance_total_month.sort_index()


distance_total_month.index = distance_total_month.index.strftime('%m-%Y')

############################################################################################################
#Z-SCORE et sprint


# Load the data from the uploaded CSV file
data = match_played

# Drop rows with missing values in the 'Nombre de sprints' column
filtered_data = data.dropna(subset=['Nombre de sprints'])

# Convert the DATE column to datetime for proper sorting
filtered_data['DATE'] = pd.to_datetime(filtered_data['DATE'])

# Calculate mean and standard deviation for 'Nombre de sprints' for each date across all players
mean_sprints_date_all = filtered_data.groupby('DATE')['Nombre de sprints'].mean()
std_sprints_date_all = filtered_data.groupby('DATE')['Nombre de sprints'].std()

# Calculate upper and lower bounds for the shadow (mean ± std deviation)
upper_bound_sprints_all = mean_sprints_date_all + std_sprints_date_all
lower_bound_sprints_all = mean_sprints_date_all - std_sprints_date_all

# Ensure mean_sprints_date_all, upper_bound_sprints_all, and lower_bound_sprints_all are lists for plotting
mean_sprints_date_all = mean_sprints_date_all.dropna()
std_sprints_date_all = std_sprints_date_all.dropna()

mean_sprints_date_all_index = mean_sprints_date_all.index.tolist()
upper_bound_sprints_all_index = upper_bound_sprints_all.index.tolist()
lower_bound_sprints_all_index = lower_bound_sprints_all.index.tolist()

mean_sprints_date_all_values = mean_sprints_date_all.values.tolist()
upper_bound_sprints_all_values = upper_bound_sprints_all.values.tolist()
lower_bound_sprints_all_values = lower_bound_sprints_all.values.tolist()

# Merge mean and std deviation back to filtered_data for z-score calculation
filtered_data = filtered_data.merge(mean_sprints_date_all, on='DATE', suffixes=('', '_mean'))
filtered_data = filtered_data.merge(std_sprints_date_all, on='DATE', suffixes=('', '_std'))

# Calculate z-score for each player
filtered_data['z_score'] = (filtered_data['Nombre de sprints'] - filtered_data['Nombre de sprints_mean']) / filtered_data['Nombre de sprints_std']

# Filter data for a specific player (D.Kutesa in this example)
player_data = filtered_data[filtered_data['PLAYER_NAME'] == player]

# Calculate the mean z-score for the specific player
mean_z_score_player = player_data['z_score'].mean()

############################################################################################################
# --------------------------------------- Z-SCORE ET DISTANCE -------------------------------------------- #

# Load the data from the uploaded CSV file
data = match_played
# Filter data to include only rows where both Distance totale and Distance m/min are not NaN
filtered_data = data.dropna(subset=['Distance totale', 'Distance m/min'])

# Convert the DATE column to datetime for proper sorting
filtered_data['DATE'] = pd.to_datetime(filtered_data['DATE'])

# Calculate mean and standard deviation for Distance m/min for each date across all players
mean_distance_date_all = filtered_data.groupby('DATE')['Distance m/min'].mean()
std_distance_date_all = filtered_data.groupby('DATE')['Distance m/min'].std()

# Calculate upper and lower bounds for the shadow (mean ± std deviation)
upper_bound_date_all = mean_distance_date_all + std_distance_date_all
lower_bound_date_all = mean_distance_date_all - std_distance_date_all

# Ensure mean_distance_date_all, upper_bound_date_all, and lower_bound_date_all are lists for plotting
mean_distance_date_all = mean_distance_date_all.dropna()
std_distance_date_all = std_distance_date_all.dropna()

upper_bound_date_all = mean_distance_date_all + std_distance_date_all
lower_bound_date_all = mean_distance_date_all - std_distance_date_all

mean_distance_date_all_index = mean_distance_date_all.index.tolist()
upper_bound_date_all_index = upper_bound_date_all.index.tolist()
lower_bound_date_all_index = lower_bound_date_all.index.tolist()

mean_distance_date_all_values = mean_distance_date_all.values.tolist()
upper_bound_date_all_values = upper_bound_date_all.values.tolist()
lower_bound_date_all_values = lower_bound_date_all.values.tolist()

# Merge mean and std deviation back to filtered_data for z-score calculation
filtered_data = filtered_data.merge(mean_distance_date_all, on='DATE', suffixes=('', '_mean'))
filtered_data = filtered_data.merge(std_distance_date_all, on='DATE', suffixes=('', '_std'))

# Calculate z-score for each player
filtered_data['z_score'] = (filtered_data['Distance m/min'] - filtered_data['Distance m/min_mean']) / filtered_data['Distance m/min_std']

# Filter data for a specific player (D.Kutesa in this example)
player_data = filtered_data[filtered_data['PLAYER_NAME'] == player]

# month


# Calculate the mean z-score for the specific player
mean_z_score_player = player_data['z_score'].mean()



############################################################################################################

# ----- PIZZA PLOT ----- #

# ---- MATCH ---- #

max_match = match_played.groupby('PLAYER_NAME').agg({
    'Accélération max': 'max',
    'Décélération max': 'max',
    'Vmax': 'max'
}).reset_index()

#print(max_match)

# ---- PLAYER  ---- #
max_player = match_played[match_played['PLAYER_NAME'] == player].groupby('PLAYER_NAME').agg({
    'Accélération max': 'max',
    'Décélération max': 'max',
    'Vmax': 'max'
}).reset_index()

# Moyennes pour les matchs et les entraînements 
mean_match = max_match[['Accélération max', 'Décélération max', 'Vmax']].mean()
mean_player = max_player[['Accélération max', 'Décélération max', 'Vmax']].values[0]
#print(mean_player)

if mean_match.mean() < mean_player.mean():
    order_match = 2
    order_player = 1
else:
    order_match = 1
    order_player = 2

params = ['Max acceleration', 'Max deceleration', 'Vmax']
values_match = [round(value, 2) for value in mean_match.tolist()]
values_player = [round(value, 2) for value in mean_player]

min_range = [5, 7, 30]
max_range = [12, 11, 37] 


# Déterminer les indices des valeurs manquantes
missing_indices = [i for i, value in enumerate(values_match) if np.isnan(value)]

# Mise à jour des couleurs des tranches pour les valeurs manquantes
slice_colors_match = "#1A78CF" * len(params)
slice_colors_antunes = ["#D70232"] * len(params)
text_colors = ["white"] * len(params)

# Créer le graphique pizza pour la moyenne des joueurs
baker = PyPizza(
    params=params,
    min_range=min_range,
    max_range=max_range,
    straight_line_color="white",
    last_circle_color="white",
    last_circle_lw=2.5,
    other_circle_lw=0,
    other_circle_color="white",
    straight_line_lw=1,
    background_color="#F2F2F2",
)

fig, ax = baker.make_pizza(
    values_match,
    figsize=(5, 5.5),
    blank_alpha=0.4,

    slice_colors=["#471625", "#471625", "#471625"],
    kwargs_slices=dict(
        edgecolor="white", zorder=order_match, linewidth=1
    ),
    kwargs_params=dict(
        color="white", fontsize=11, va="center"
    ),
    kwargs_values=dict(
        color="white", fontsize=11, zorder=3,
        bbox=dict(
            edgecolor="white", facecolor="#471625",
            boxstyle="round,pad=0.1", lw=1,
        )
    )
)

baker.make_pizza(
    values_player,
    ax=ax,
    blank_alpha=0.4,
    slice_colors=["#DAA520", "#DAA520", "#DAA520"],
    kwargs_slices=dict(
        edgecolor="white", zorder=order_player, linewidth=1
    ),
    kwargs_params=dict(
        color="white", fontsize=11, va="center"
    ),
    kwargs_values=dict(
        color="white", fontsize=11, zorder=5,
        bbox=dict(
            edgecolor="white", facecolor="#DAA520",
            boxstyle="round,pad=0.1", lw=1
        )
    ),
)


fig.text(
    0.99, 0.02, f"Team average in burgundy, {player} ocher color", size=9,
    ha="left", color="white"
)
# backgroung color en noir 

fig.patch.set_facecolor('black')


#sauvegarde en png de taille normale

fig.savefig("comparaison_performance.png", dpi=300, bbox_inches='tight', facecolor='black')


############################################################################################################
user_input = player

# Extraction du nom de famille
last_name = user_input.split('.')[-1]
old_path = '/Users/cosaph/Desktop/fc servette/Coralie/image/1_Antunes.png'  # Chemin vers l'image du joueur
old_text = "Antunes"
image_path = old_path.replace(old_text, last_name)

############################################################################################################

############################------------ PLAYER PLOT ---------------########################################

############################################################################################################

player_image = Image.open(image_path)
logo = Image.open('/Users/cosaph/Desktop/fc servette/pictures/logo.png')
comparaison_performance = Image.open('comparaison_performance.png')
# a la place de facecolor = black je veux l'image banniere.jpeg
fig = plt.figure(figsize=(18, 12), facecolor='black')
gs = GridSpec(4, 5, figure=fig, width_ratios=[5,5,5,5,5], height_ratios=[2, 2, 2, 2])

# ----- Configuration des axes ----- #

# Définir les sous-graphiques
ax_minutes_played = fig.add_subplot(gs[0, 0])

ax_image = fig.add_subplot(gs[2:, 0])  

ax_title = fig.add_subplot(gs[0, 0:5])
ax_train = fig.add_subplot(gs[0, 1])
ax_match = fig.add_subplot(gs[0, 2])


ax_distance = fig.add_subplot(gs[1:5, 4:])
ax_distance_train = fig.add_subplot(gs[2:5, 4:])

ax_vitesse_max  = fig.add_subplot(gs[1, 0])  # Vitesse max juste au-dessus de l'image
ax_distance_month = fig.add_subplot(gs[1, 1])
ax_nb_sprint = fig.add_subplot(gs[1, 2])
ax_zscore_sprint = fig.add_subplot(gs[1,3])
ax_minutes_train_played = fig.add_subplot(gs[1, 4])

ax_pizza = fig.add_subplot(gs[2:10, 1:4])
ax_zscore_distance = fig.add_subplot(gs[0, 3])

ax_logo = fig.add_subplot(gs[0, 4])



fig.subplots_adjust(hspace=1)

# ----- LOGO ------#   

ax_logo.imshow(logo)
ax_logo.axis('off')

# ----- IMAGE ------#

ax_image.imshow(player_image)
ax_image.axis('off')

# ----- Pizza Plot ----- #

'''
Je ne vais pas pouvoir intégrer mon pizza plot comme ça je vais le mettre en image. 
'''
# agrandir l'image de la pizza

ax_pizza.imshow(comparaison_performance)
ax_pizza.axis('off')


# ----- PRÉSENCE À L'ENTRAINEMENT ------ #

arc_full_train = Arc((0.5, 0.5), 0.6, 0.6, theta1=0, theta2=180, color='grey', linewidth=8)
arc_played_train = Arc((0.5, 0.5), 0.6, 0.6, theta1=0, theta2=180 * (present_player/ attendance_player_sum), color='#471625', linewidth=8)
ax_train.add_patch(arc_full_train)
ax_train.add_patch(arc_played_train)
ax_train.text(0.5, 0.6, f'Training Presence\n{round((pourcentage_present),2)} %', color='white', fontsize=10, ha='center', va='center', fontfamily='Arial', fontweight='bold')
ax_train.axis('equal')  # Assure que l'arc de cercle est dessiné correctement
ax_train.axis('off')

# ----- MATCH JOUES EN TANT QUE TITULAIRE ET REMPLACANT ----- #

arc_full_match = Arc((0.5, 0.5), 0.6, 0.6, theta1=0, theta2=180, color='grey', linewidth=9)
arc_titulaire_match = Arc((0.5, 0.5), 0.6, 0.6, theta1=0, theta2=180 * attendance_titulaire/nombre_match, color='#471625', linewidth=9)
arc_remplacant_match = Arc((0.5, 0.5), 0.6, 0.6, theta1=0, theta2=180 * (attendance_match_remplaçant/nombre_match), color='#7a303f', linewidth=9)
ax_match.add_patch(arc_full_match)
ax_match.add_patch(arc_titulaire_match)
ax_match.add_patch(arc_remplacant_match)
ax_match.legend([f'Absent ({nombre_match_absent} / {nombre_match} )', f'Titulaire ({attendance_titulaire} / {nombre_match})', f'Remplaçant ({attendance_match_remplaçant} / {nombre_match})'], 
                loc='best', fontsize=7, facecolor='white')
ax_match.text(0.5, 0.6, f'Match Presence', color='white', fontsize=10, ha='center', va='center', fontfamily='Arial', fontweight='bold')
ax_match.axis('equal')  # Assure que l'arc de cercle est dessiné correctement
ax_match.axis('off')


# ----- DISTANCE MOYENNE MENSUEL AUX ENTRAÎNEMENTS ---#


ax_distance_month.plot(distance_total_month.index, distance_total_month.values, marker='o', linestyle='-', color='#7a303f', linewidth=2)

# Ajouter les étiquettes
for i, distance in enumerate(distance_total_month.values):
    ax_distance_month.text(distance_total_month.index[i], distance, f"{distance:.2f}", color='white', ha='center', va='bottom', fontsize=8)

ax_distance_month.set_title('Total Distance per Month', fontsize=14, color='white', fontfamily='Arial', fontweight='bold')
ax_distance_month.set_xlabel('Month', fontsize=12, color='white', fontfamily='Arial')
ax_distance_month.set_ylabel('Total Distance (km)', fontsize=12, color='white', fontfamily='Arial')
ax_distance_month.tick_params(axis='x', colors='white')
ax_distance_month.tick_params(axis='y', colors='white')
ax_distance_month.spines['top'].set_color('white')
ax_distance_month.spines['right'].set_color('white')
ax_distance_month.spines['left'].set_color('white')
ax_distance_month.spines['bottom'].set_color('white')

ax_distance_month.grid(False)

fig.patch.set_facecolor('black')
ax_distance_month.set_facecolor('black')

months = distance_total_month.index.to_list()
ax_distance_month.set_xticks(months)
ax_distance_month.set_xticklabels(months, rotation=45, ha='right', fontsize=10, color='white')

# Ajouter la légende
ax_distance_month.legend(loc='upper left', fontsize=10, frameon=False, labelcolor='white')


# ----- TOP SPEED PER MONTHS  ----- #

# Calculate the cumulative values for the training data

ax_nb_sprint.plot(range(len(train_cumulative)), train_cumulative, color='white', label='Training')
ax_nb_sprint.plot(range(len(match_cumulative)), match_cumulative, color='#471625', label='Match')

ax_nb_sprint.set_title('Cumulative Sprint Count per Month', fontsize=14, color='white', fontfamily='Arial', fontweight='bold')
ax_nb_sprint.set_xticks(range(len(months)))
ax_nb_sprint.set_xticklabels(months, rotation=45, ha='right', fontsize=10, color='white')
ax_nb_sprint.set_yticklabels([f'{count:.0f}' for count in ax_nb_sprint.get_yticks()], fontsize=10, color='white')
ax_nb_sprint.set_facecolor('black')

ax_nb_sprint.legend(loc='upper left', frameon=False, fontsize=10, labelcolor='white')
ax_nb_sprint.grid(False)


############################################################################################################
# --------------- Sprint et Z-score -------------------- #


ax_zscore_sprint.scatter(player_data['DATE'], player_data['Nombre de sprints'], c=player_data['z_score'], cmap='coolwarm', edgecolor=None, label=f'Z-Score: {mean_z_score_player:.2f}')

# Plot the shadow area (mean ± std deviation)
ax_zscore_sprint.fill_between(mean_sprints_date_all_index, lower_bound_sprints_all_values, upper_bound_sprints_all_values, color='grey', alpha=0.3)
ax_zscore_sprint.set_title('Z-Score of Sprint Speed', fontsize=14, color='white', fontfamily='Arial', fontweight='bold')
#ax_zscore_sprint.set_xlabel('Date', fontsize=12, color='white', fontfamily='Arial')
ax_zscore_sprint.set_ylabel('Nombre de sprints', fontsize=12, color='white', fontfamily='Arial')
#ax_zscore_sprint.tick_params(axis='x', colors='white')
ax_zscore_sprint.tick_params(axis='y', colors='white')
ax_zscore_sprint.spines['top'].set_color('white')
ax_zscore_sprint.spines['right'].set_color('white')
ax_zscore_sprint.spines['left'].set_color('white')
ax_zscore_sprint.spines['bottom'].set_color('white')
ax_zscore_sprint.grid(False)
fig.patch.set_facecolor('black')
ax_zscore_sprint.set_facecolor('black')




# Define parameters for the filled circles and text
circle_params_train = {'color': 'white', 'fill': True, 'alpha': 0.7}
circle_params_match = {'color': '#7a303f', 'fill': True, 'alpha': 0.7}
text_params = {'color': 'white', 'fontsize': 11, 'fontfamily': 'Arial', 'fontweight': 'bold', 'ha': 'center', 'va': 'center'}

############################################################################################################
# ----------------- DISTANCE & z-SCORE ----------------- #

ax_zscore_distance.scatter(player_data['DATE'], player_data['Distance m/min'], c=player_data['z_score'], cmap='coolwarm', edgecolor=None, label=f'Z-Score: {mean_z_score_player:.2f}')

# Plot the shadow area (mean ± std deviation)
ax_zscore_distance.fill_between(mean_distance_date_all_index, lower_bound_date_all_values, upper_bound_date_all_values, color='white', alpha=0.5)
ax_zscore_distance.set_title('Z-Score of Distance m/min', fontsize=14, color='white', fontfamily='Arial', fontweight='bold')
#ax_zscore_sprint.set_xlabel('Date', fontsize=12, color='white', fontfamily='Arial')
ax_zscore_distance.set_ylabel('Distance m/min', fontsize=12, color='white', fontfamily='Arial')
#ax_zscore_sprint.tick_params(axis='x', colors='white')
ax_zscore_distance.tick_params(axis='y', colors='white')
ax_zscore_distance.spines['top'].set_color('white')
ax_zscore_distance.spines['right'].set_color('white')
ax_zscore_distance.spines['left'].set_color('white')
ax_zscore_distance.spines['bottom'].set_color('white')
ax_zscore_distance.grid(False)
fig.patch.set_facecolor('black')
ax_zscore_distance.set_facecolor('black')

############################################################################################################


# ------ Nombre de minutes jouées en tant que titulaire en match ----- #

circle_params_min = {'color': '#471625', 'fill': True, 'alpha': 0.7}

# Minutes played circle plot
ax_minutes_played.add_patch(plt.Circle((0.5, 0.5), 0.4, **circle_params_min))
ax_minutes_played.text(0.5, 0.5, f'{total_minutes_all_match_player} / {max_minutes} min', **text_params)
ax_minutes_played.text(0.5, 0.4, f'{ratio_minutes:.2f} %', **text_params)
ax_minutes_played.set_title('Total Minutes Played', fontsize=16, color='white', fontfamily='Arial', fontweight='bold')
ax_minutes_played.axis('off')
ax_minutes_played.set_aspect('equal')

# ------ 90%VMAX AT TRAINING----- #

labels = ['90% Vmax', '85% Vmax', ' ≤ 85% Vmax']
sizes = [nombre_train_90_vmax, nombre_train_85_vmax, nombre_train_inferier_85_vmax]
colors = ['#471625', '#7a303f', '#a23d5c', '#d15a7b', '#e8a4b8']

# Création du graphique à secteurs
ax_minutes_train_played.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                            shadow=True, startangle=140, textprops={'color': 'white'})

centre_circle = plt.Circle((0, 0), 0.70, color='white', fc='white', linewidth=0)
ax_minutes_train_played.set_title('Max Speed at training', fontsize=16, color='white', fontfamily='Arial', fontweight='bold')
ax_minutes_train_played.axis('equal')

# ----- DISTANCE TOTALE ----- #

ax_distance.text(0.5, 0.5, f'Total Distance in match \n{distance_total_match/1000:.2f} km', color='white', fontsize=16, ha='center', va='center', fontfamily='Arial', fontweight='bold')
ax_distance.axis('off')

ax_distance_train.text(0.5, 0.5, f'Total Distance in training \n{total_distance_train/1000:.2f} km', color='white', fontsize=16, ha='center', va='center', fontfamily='Arial', fontweight='bold')
ax_distance_train.axis('off')

# ----- VITESSE MAX ----- #

ax_vitesse_max.text(0.5, 0.5, f'Max Speed in match \n{vitesse_max_match:.2f} km/h\n Rank : {rank_player}th / {nb_joueur}', color='white', fontsize=16, ha='center', va='center', fontfamily='Arial', fontweight='bold')
ax_vitesse_max.axis('off')



ax_title.set_title(f'Performances of  {player}', fontsize=20, color='white', fontfamily='Arial', fontweight='bold', ha='right')
ax_title.axis('off')



# Define the parameters for the pizza plot

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

