# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    try_streamlit.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ccottet <ccottet@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/06 10:23:28 by ccottet           #+#    #+#              #
#    Updated: 2024/06/06 15:29:08 by ccottet          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import plotly.graph_objects as go
from mplsoccer import PyPizza, FontManager
import numpy as np

import matplotlib.colors as mcolors
import seaborn as sns

############################################################################################################
# ************ Load data ************ #

all_training = pd.read_csv("/Users/cosaph/Desktop/fc servette/Coralie/Export Servette Pro - all training.csv")
match_played = pd.read_csv("/Users/cosaph/Desktop/fc servette/Coralie/Match_player/Export Servette Pro - all games.csv")


############################################################################################################

st.title("Player Performance Dashboard")

# ************* Sidebar ******$****** #

# set a sidebar color 

st.markdown("""
<style>
/* Changer la couleur de fond de la sidebar */
[data-testid=stSidebar] {
    background-color: #FFFFF0;
}

/* Changer la couleur de fond des éléments de la selectbox */
div[data-baseweb="select"] > div {
    background-color: #FFFFF0; /* Couleur de fond de la selectbox */
    border: 2px solid #85142B; /* Couleur du contour de la selectbox */
}

/* Changer la couleur de fond et le texte des options de la selectbox */
div[data-baseweb="select"] span {
    color: #85142B; /* Couleur du texte des options */
}

/* Changer la couleur de fond des options survolées */
div[role="listbox"] > ul > li:hover {
    background-color: #ADD8E6; /* Couleur de fond au survol */
    color: #85142B; /* Couleur du texte au survol */
}

/* Changer la couleur du texte dans la sidebar */
[data-testid="stSidebar"] * {
    color: #85142B; /* Couleur du texte dans la sidebar */
}
</style>
""", unsafe_allow_html=True)

# Ajouter la selectbox dans la sidebar
with st.sidebar:
    player = st.selectbox('Select a player', all_training['PLAYER_NAME'])
    st.subheader(f"Performance Details for {player}")
    st.subheader("Select the type of performance")
    performance_type = st.radio("Performance type", ["Individual", "Collective"])




nb_joueur = all_training['PLAYER_NAME'].nunique()

############################################################################################################
######################################## INDIVIDUEL #########################################################
############################################################################################################

if performance_type == "Individual":
    ############################################################################################################
    # ************* Display The player image ************* #

    # Extraction du nom de famille
    last_name = player.split('.')[-1]
    old_path = '/Users/cosaph/Desktop/fc servette/Coralie/image/1_Antunes.png'  # Chemin vers l'image du joueur
    old_text = "Antunes"
    image_path = old_path.replace(old_text, last_name)

    ############################################################################################################
    # ************* Display Coulour ************* #

    logo_path = "/Users/cosaph/Desktop/fc servette/pictures/logo.png"  # Assurez-vous que ce chemin est correct
    logo = mpimg.imread(logo_path)

    # Afficher l'image du joueur
    image = mpimg.imread(image_path)
    col1, col2 = st.columns([1, 3])

    # Placer le logo et l'image du joueur côte à côte
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(logo, use_column_width=True)
    with col2:
        st.image(image, use_column_width=True, width=150)


    # je veux la couleur du club en background #85142B

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://www.colorhexa.com/85142b.png");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)



    ############################################################################################################
    # ************* Display max speed ************* #


    # compute max speed for each player
    max_speed_player = match_played.groupby('PLAYER_NAME')['Vmax'].max().reset_index()
    max_speed_player = max_speed_player.sort_values(by='Vmax', ascending=False).reset_index(drop=True)
    max_speed_player['Rank'] = max_speed_player.index + 1
    rank_player = max_speed_player[max_speed_player['PLAYER_NAME'] == player ].index[0] + 1

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Max Speed", value=f"{max_speed_player['Vmax'].values[0]} km/h")

    with col2:
        st.metric(label="Rank", value=f"{rank_player} / {nb_joueur}")


    ############################################################################################################
    # ************* Display Présence à l'entrainement ************* #

    nombre_train = all_training['DATE'].nunique()

    attendance = all_training['ATTENDANCE'].value_counts()


    attendance_player = all_training[all_training['PLAYER_NAME'] == player ]['ATTENDANCE'].value_counts()

    if isinstance(attendance_player, pd.Series):
        attendance_player = attendance_player.reindex(['PRESENT', 'ADAPTE', 'BLESSE', 'READAPTATION', 'REPOS', 'EN SOIN', 'MALADE', 'ABSENT'], fill_value=0)


    attendance_player_sum = 0
    for col in ['PRESENT', 'ADAPTE', 'BLESSE', 'READAPTATION', 'REPOS', 'EN SOIN', 'MALADE', 'ABSENT']:
        try:
            attendance_player_sum += attendance_player[col]
        except (KeyError, TypeError):
            print(f"La colonne '{col}' est absente dans les données d'attendance pour le joueur {player}.")

        try:
            present_player = attendance_player['PRESENT']
        except (KeyError, TypeError):
            print(f"La colonne 'PRESENT' est absente dans les données d'attendance pour le joueur {player}.")
            present_player = 'N/A'

    pourcentage_present = (present_player / attendance_player_sum) * 100

    # Display

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Presence in training", value=f"{attendance_player_sum}")
    with col2:
        st.metric(label="Presence Rate", value=f"{pourcentage_present:.2f}%")


    ############################################################################################################
    # ************* Display Distance totale par mois ************* #

    # Back
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


    # Définir les styles CSS pour changer la couleur de fond de la page
    st.markdown(
    """
    <style>
    .reportview-container {
    background-color: #FFFFF0;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.subheader("Total Distance covered per month")

    # Création du graphique
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=distance_total_month.index, y=distance_total_month.values, mode='lines+markers', line=dict(color='#85142B', width=2)))

    # Ajout des annotations avec arrondi au centième
    for i, (date, distance) in enumerate(zip(distance_total_month.index, distance_total_month.values)):
        fig.add_annotation(x=date, y=distance, text=f'{round(distance, 2)}', font=dict(size=9), showarrow=False, yshift=10, xanchor='right')

    # Personnalisation du graphique
    fig.update_layout(
    plot_bgcolor='#FFFFF0',
    paper_bgcolor='#85142B',
    font_color='#85142B',
    xaxis_title='Month',
    yaxis_title='Distance covered',
    title='Total Distance Covered per Month',
    title_font_color='#85142B',
    xaxis_tickangle=45
    )

    # Affichage du graphique dans Streamlit
    st.plotly_chart(fig)


    ############################################################################################################
    # --------------------------------------- Z-SCORE ET DISTANCE -------------------------------------------- #


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

    # Display 

    st.subheader(f"Z-Score for {player}")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Mean Z-Score", value=f"{mean_z_score_player:.2f}")

    with col2:
        st.metric(label="Distance covered in match ", value=f"{player_data['Distance totale'].sum() / 1000:.2f} km")

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=mean_distance_date_all_index, y=mean_distance_date_all_values, mode='lines', line=dict(color='#85142B', width=3), name='Mean Distance m/min'))
    fig.add_trace(go.Scatter(x=mean_distance_date_all_index, y=lower_bound_date_all_values, mode='lines', line=dict(color='brown', width=1, dash='dash'), fillcolor='#85142B', fill='none', name='- 1 std deviation'))
    fig.add_trace(go.Scatter(x=mean_distance_date_all_index, y=upper_bound_date_all_values, mode='lines', line=dict(color='brown', width=1, dash='dash'), fillcolor='#85142B', fill='none', name='+ 1 std deviation'))

    fig.add_trace(go.Scatter(x=player_data['DATE'], y=player_data['Distance m/min'], mode='lines', line=dict(color='black', width=2), name=f'{player} Distance m/min'))

    # Set the title and labels
    fig.update_layout(
    plot_bgcolor='#FFFFF0',
    paper_bgcolor='#85142B',
    font_color='#85142B',
    xaxis_title='Date',
    yaxis_title='Distance m/min',
    title=f'Distance m/min for {player} with Mean and ± 1 std deviation for all players',
    title_font_color='white'
    )

    # Display the plot
    st.plotly_chart(fig)


    ############################################################################################################    

    # back

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
    straight_line_color="#FFFFF0",
    last_circle_color="#FFFFF0",
    last_circle_lw=2.5,
    other_circle_lw=0,
    other_circle_color="#FFFFF0",
    straight_line_lw=1,
    background_color="#FFFFF0",
    )

    fig, ax = baker.make_pizza(
    values_match,
    figsize=(3, 3.5),
    blank_alpha=0.4,

    slice_colors=["#471625", "#471625", "#471625"],
    kwargs_slices=dict(
    edgecolor="#FFFFF0", zorder=order_match, linewidth=1
    ),
    kwargs_params=dict(
    color="#FFFFF0", fontsize=11, va="center"
    ),
    kwargs_values=dict(
    color="#FFFFF0", fontsize=11, zorder=3,
    bbox=dict(
    edgecolor="#FFFFF0", facecolor="#471625",
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
    edgecolor="#FFFFF0", zorder=order_player, linewidth=1
    ),
    kwargs_params=dict(
    color="#FFFFF0", fontsize=11, va="center"
    ),
    kwargs_values=dict(
    color="#FFFFF0", fontsize=11, zorder=5,
    bbox=dict(
    edgecolor="#FFFFF0", facecolor="#DAA520",
    boxstyle="round,pad=0.1", lw=1
    )
    ),
    )


    fig.text(
    0.02, 0.02, f"Team average in burgundy, {player} ocher color", size=9,
    ha="right", color="#FFFFF0"
    )
    # backgroung color en noir 

    fig.patch.set_facecolor('#85142B')

    # Afficher le graphique avec Streamlit
    st.pyplot(fig)


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

    # Display the mean z-score for the specific player

    st.subheader(f"Z-Score for {player} - Number of sprints")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Mean Z-Score", value=f"{mean_z_score_player:.2f}")

    with col2:
        st.metric(label="Number of sprints", value=f"{player_data['Nombre de sprints'].sum()}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=mean_sprints_date_all_index, y=mean_sprints_date_all_values, mode='lines', line=dict(color='#85142B', width=3), name='Mean Number of sprints'))
    fig.add_trace(go.Scatter(x=mean_sprints_date_all_index, y=lower_bound_sprints_all_values, mode='lines', line=dict(color='brown', width=1, dash='dash'), fillcolor='#85142B', fill='none', name='- 1 std deviation'))
    fig.add_trace(go.Scatter(x=mean_sprints_date_all_index, y=upper_bound_sprints_all_values, mode='lines', line=dict(color='brown', width=1, dash='dash'), fillcolor='#85142B', fill='none', name='+ 1 std deviation'))

    fig.add_trace(go.Scatter(x=player_data['DATE'], y=player_data['Nombre de sprints'], mode='lines', line=dict(color='black', width=2), name=f'{player} Number of sprints'))

    # Set the title and labels
    fig.update_layout(
    plot_bgcolor='#FFFFF0',
    paper_bgcolor='#85142B',
    font_color='#85142B',
    xaxis_title='Date',
    yaxis_title='Number of sprints',
    title=f'Number of sprints for {player} with Mean and ± 1 std deviation for all players',
    title_font_color='white'
    )


    # Display the plot
    st.plotly_chart(fig)


    ############################################################################################################
    # ************* Display Distance 90-85-Inf 85 Vmax ************* #

    # Back
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

    st.subheader(f"Vmax for {player}")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Number of training with Vmax > 90% Vmax match", value=f"{nombre_train_90_vmax}")

    with col2:
        st.metric(label="Number of training with Vmax > 85% Vmax match", value=f"{nombre_train_85_vmax}")

    col3, col4 = st.columns(2)

    with col3:
        st.metric(label="Number of training with Vmax < 85% Vmax match", value=f"{nombre_train_inferier_85_vmax}")

    labels = ['> 90% Vmax', '> 85% Vmax', '< 85% Vmax']
    sizes = [nombre_train_90_vmax, nombre_train_85_vmax, nombre_train_inferier_85_vmax]
    colors = ['#85142B', '#DAA520', '#471625']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'color':"white"})

    ax.axis('equal')
    fig.patch.set_facecolor('#85142B')
    plt.setp(ax.texts, size=12, weight="bold")
    ax.set_facecolor('#85142B')


    #ax.legend(labels, loc="best")
    st.pyplot(fig)

############################################################################################################
######################################## COLLECTIF #########################################################
############################################################################################################

df = pd.DataFrame()

# Prendre le max et faire un groupby par joueur
df['Max_speed'] = match_played.groupby('PLAYER_NAME')['Vmax'].max()
df['Accélération max'] = match_played.groupby('PLAYER_NAME')['Accélération max'].max()

# Filtrer les matchs avec une durée > 60 minutes
matchs = match_played[match_played['DURATION'] > 60]

# Prendre le max de distance m/min pour chaque joueur
distance_mmin = matchs.groupby('PLAYER_NAME')['Distance m/min'].max()
df['Distance_m/min'] = distance_mmin

players = all_training['PLAYER_NAME'].unique()

# Boucle pour chaque joueur
for player in players:
    attendance_player = all_training[all_training['PLAYER_NAME'] == player]['ATTENDANCE'].value_counts()
    
    if isinstance(attendance_player, pd.Series):
        attendance_player = attendance_player.reindex(['PRESENT', 'ADAPTE', 'BLESSE', 'READAPTATION', 'REPOS', 'EN SOIN', 'MALADE', 'ABSENT'], fill_value=0)
    
    attendance_player_sum = attendance_player.sum()

    present_player = attendance_player['PRESENT']
    
    pourcentage_present = (present_player / attendance_player_sum) * 100 if attendance_player_sum > 0 else 0

    df.loc[player, 'Attendance'] = pourcentage_present

# Calcul des z-scores pour chacune des variables
df['PLAYER_NAME'] = df.index
df['Max_speed_z'] = (df['Max_speed'] - df['Max_speed'].mean()) / df['Max_speed'].std()
df['Accélération max_z'] = (df['Accélération max'] - df['Accélération max'].mean()) / df['Accélération max'].std()
df['Distance_m/min_z'] = (df['Distance_m/min'] - df['Distance_m/min'].mean()) / df['Distance_m/min'].std()
df['Attendance_z'] = (df['Attendance'] - df['Attendance'].mean()) / df['Attendance'].std()

# Supprimer certains joueurs spécifiques
df = df.loc[~df['PLAYER_NAME'].isin(['N.Vouilloz', 'P.Pflücke', 'J.Onguene', 'B.Fofana', 'R.Rodelin'])]


if performance_type == "Collective":
    st.header("Collective Performance Analysis")

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://www.colorhexa.com/85142b.png");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)

    df['Total_z'] = df[['Max_speed_z', 'Accélération max_z', 'Distance_m/min_z', 'Attendance_z']].sum(axis=1, skipna=True)

    df = df.sort_values('Total_z', ascending=False)

    plt.figure(figsize=(10, 10))
    sns.barplot(x='Total_z', y='PLAYER_NAME', data=df, palette = 'Greys')
    plt.xlabel('Total z-score', color='white')
    plt.ylabel('Players', color='white')
    plt.title('Total z-score of Players', color='white')
    plt.gca().set_facecolor('#85142B')
    plt.gcf().set_facecolor('#85142B')
    plt.xticks(color='white')
    plt.yticks(color='white')
    st.pyplot(plt)
