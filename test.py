import pandas as pd
import seaborn as sns
import matplotlib
#récupération du df nettoyé
data = pd.read_pickle("data_cleaned.pkl")
#Mise en contexte
#nombre employés
nombre_employés=len(data.loc[data['DateofTermination'].isna()])

#embauches départs sur 2018
date_référence_N=pd.to_datetime('01/01/2018')
nombre_embauches_N=len(data.loc[data['DateofHire']>date_référence_N])
nombre_departs_N=len(data.loc[data['DateofTermination']>date_référence_N])


#ancienneté moyenne
# Convertir les colonnes en datetime64 (Pandas standard)
data['DateofHire'] = pd.to_datetime(data['DateofHire'])
data['DateofTermination'] = pd.to_datetime(data['DateofTermination'])

# Calcul de la séniorité
data['Seniority'] = (data['DateofTermination'] - data['DateofHire']).where(
    data['DateofTermination'].notna(),
    pd.to_datetime('today') - data['DateofHire']
).dt.days  # Convertit en nombre de jours

ancienete_moyenne_annees=round((data['Seniority']).mean()/365.25,2)

#Performance
perf=data.iloc[:,2:3].describe().round(2)
perf=perf.reset_index()
perf=perf.rename(columns={'index':'Statistique', 'PerfScoreID': 'Scores de performance'})
perf.iloc[3,0]='Min'
perf.iloc[4,0]='Q25%'
perf.iloc[5,0]='Médiane'
perf.iloc[6,0]='Q75%'
perf.iloc[7,0]='Max'
perf=perf.drop(index=[0,1,2])

# Calcul des effectifs
effectifs = data["PerfScoreID"].value_counts()  
effectifs=effectifs.sort_index(ascending=False)
# Extraction des modalités
modalites = effectifs.index

# Création de la table de fréquence
tab_perf = pd.DataFrame(modalites, columns=["PerfScoreID"])


# Ajout de la colonne des effectifs
tab_perf["n"] = effectifs.values  

# Ajout de la colonne des fréquences (pourcentages)
tab_perf["%"] = ((tab_perf["n"] / len(data)) * 100)
tab_perf["%"] = tab_perf["%"].map(lambda x: f"{x:.1f}")

# Renommer les colonnes
tab_perf = tab_perf.rename(columns={'PerfScoreID': 'Score de performance', 'n': 'Effectifs'})


tab_perf["%"]=tab_perf["%"].astype(float)

#calcul des indicateurs de satisfaction et engagement
#colonnes : EngagementSurvey et EmpSatisfaction

data['EmpSatisfaction'].describe() #==>boite à moustache
data['EngagementSurvey'].describe() #==>boite à moustache

#distribution scores satisfaction pour les employés encore en poste:
employes_actifs=data.loc[data['DateofTermination'].isna()]
tableau_satisfaction=employes_actifs.groupby('EmpSatisfaction')['EmpSatisfaction'].count()
tableau_satisfaction=tableau_satisfaction.to_frame()
tableau_satisfaction = tableau_satisfaction.rename(columns={'EmpSatisfaction': 'effectifs'})
tableau_satisfaction.reset_index(inplace=True)
tableau_satisfaction = tableau_satisfaction.rename(columns={'EmpSatisfaction': 'Score de satisfaction'})
tableau_satisfaction['frequence']=tableau_satisfaction['effectifs']/len(employes_actifs)

tableau_engagement=employes_actifs.copy()
tableau_engagement['Score engagement']=pd.cut(tableau_engagement['EngagementSurvey'], bins=5,labels=[1,2,3,4,5])
tableau_engagement=tableau_engagement.groupby('Score engagement')['Score engagement'].count()
tableau_engagement=tableau_engagement.to_frame()
tableau_engagement=tableau_engagement.rename(columns={'Score engagement':'effectifs'})
tableau_engagement['frequence']=tableau_engagement['effectifs']/len(employes_actifs)
tableau_engagement.reset_index(inplace=True)

import scipy.stats as st
import numpy as np
pearson=st.pearsonr(employes_actifs["EmpSatisfaction"],employes_actifs["EngagementSurvey"])[0]
cov=np.cov(employes_actifs["EmpSatisfaction"],employes_actifs["EngagementSurvey"],ddof=0)[1,0]
employes_actifs['engagement_class']=pd.cut(employes_actifs['EngagementSurvey'], bins=[1,2,3,4,5])
employes_actifs['engagement_class'].unique()

#recherche des facteurs améliorant l'engagement (projets, manageur, génération (DOB),departement et salaire)
annalyse_engagement = employes_actifs.loc[:, ['PerfScoreID','EngagementSurvey','Salary', 'Position', 'DOB', 'Department', 'ManagerID', 'SpecialProjectsCount']]
annee_naissance = pd.DatetimeIndex(annalyse_engagement['DOB']).year
annalyse_engagement['Generation'] = pd.cut(annee_naissance, bins=[1946,1964, 1980, 1996, 2012, 2020], labels=['Baby boomers', 'X generation', 'Millennials', 'Z generation', 'Alpha generation'])

# Création des bins
bins_engagement = [1, 2, 3, 4, 5]
annalyse_engagement['engagement_class'] = pd.cut(annalyse_engagement['EngagementSurvey'], bins=bins_engagement)

# Création des milieux d'intervalles
intervals_engagement = pd.IntervalIndex.from_breaks(bins_engagement)
bin_engagement_mids = intervals_engagement.mid

# Création d’un mapping entre l’intervalle et son milieu
interval_to_mid = dict(zip(intervals_engagement, bin_engagement_mids))

# Application du mapping via .map()
annalyse_engagement['engagement_class'] = annalyse_engagement['engagement_class'].map(lambda x: interval_to_mid.get(x, np.nan))

annalyse_engagement['engagement_class']=annalyse_engagement['engagement_class'].astype(int)
annalyse_engagement['ManagerID']=annalyse_engagement['ManagerID'].astype(int)


mots_cles_cadre = [
    'manager', 'engineer', 'director', 'responsable', 'lead',
    'architect', 'chief', 'ceo', 'cio', 'head','senior', 'sr.', 'vp'
]

annalyse_engagement['type_poste'] = annalyse_engagement['Position'].str.lower().apply(
    lambda x: 'cadre' if any(keyword in x for keyword in mots_cles_cadre) else 'non cadre'
)

annalyse_engagement['type_poste'] = annalyse_engagement['type_poste'].astype('string')



annalyse_engagement['type_poste'].unique()


#etude de l'engagement et performances medians en fonctions des salaires, nbr projets, position, departement, manageur, generation

# 1/ Salaire
#mettre le salaire en classes avec 1 valeur centrale pour chaque classe

# Étape 1 : définir les bins
bins_salaire = [45000, 65500, 86000, 106500, 127000, 147500, 168000, 188500, 209000, 229500, 250000]

# Étape 2 : couper en classes (on garde les labels=False pour obtenir les indices)
annalyse_engagement['salary_class'] = pd.cut(
    annalyse_engagement['Salary'], 
    bins=bins_salaire, 
    labels=False,
    include_lowest=True,
    duplicates='drop'
)

# Étape 3 : calculer les milieux des classes
bin_salaire_mids = pd.IntervalIndex.from_breaks(bins_salaire).mid

# Étape 4 : associer chaque indice au milieu correspondant
annalyse_engagement['salary_class'] = annalyse_engagement['salary_class'].map(lambda x: bin_salaire_mids[x] if pd.notna(x) else np.nan)

#calculer perf et engagement median en fonction du salaire
salaire_engage=annalyse_engagement.groupby('salary_class')[['engagement_class',
                                                               'PerfScoreID']].median().reset_index()

# 2/ Projets
projets_engage=annalyse_engagement.groupby('SpecialProjectsCount')[['engagement_class',
                                                               'PerfScoreID']].median().reset_index()

# 3/ type de poste
cadres_engage=annalyse_engagement.groupby('type_poste')[['engagement_class',
                                                               'PerfScoreID']].median().reset_index()

# 3/ departement
department_engage=annalyse_engagement.groupby('Department')[['engagement_class',
                                                               'PerfScoreID']].median().reset_index()
# 3/ Manageur
manager_engage=annalyse_engagement.groupby('ManagerID')[['engagement_class',
                                                               'PerfScoreID']].median().reset_index()

# 3/ Generation
generation_engage=annalyse_engagement.groupby('Generation')[['engagement_class',
                                                               'PerfScoreID']].median().reset_index()
#-----------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec 
import seaborn as sns
import textwrap

fig4 = plt.figure(figsize=(12, 30)) 

gs4 = gridspec.GridSpec(nrows=10, ncols=6, height_ratios=[0.2, 1, 0.5,1, 1, 1,1, 0.7, 0.005,0.7]) #config ratios en colonnes et lignes 

# Créer les subplots 

ax16 = fig4.add_subplot(gs4[0, :])  #titre en haut ==> 1

ax17 = fig4.add_subplot(gs4[1, 0:3])  #matrice engagement satisfaction performance ==> 2

ax18 = fig4.add_subplot(gs4[1, 3:6])   #box plot engagement performance ==> 3

ax19 = fig4.add_subplot(gs4[2, 0:3])   #texte ==> 4

ax20 = fig4.add_subplot(gs4[2, 3:6])   #texte ==> 5

ax21 = fig4.add_subplot(gs4[3, 0:3])   #salaire ==> 6

ax22 = fig4.add_subplot(gs4[3, 3:6])   #nombre de projets ==> 7

ax23 = fig4.add_subplot(gs4[4, 0:3])   #Generation ==> 8

ax24 = fig4.add_subplot(gs4[4, 3:6])   #departement ==> 9

ax25 = fig4.add_subplot(gs4[5, 0:4])   #Manageur ===> 10

ax26 = fig4.add_subplot(gs4[5, 4:6])   #cadre/non cadre==>11

ax27 = fig4.add_subplot(gs4[6, 0:6])   #conclusion ==> 12

ax28 = fig4.add_subplot(gs4[7, 0:6])   #note ==> 13

ax29 = fig4.add_subplot(gs4[8, 0:6])   #resumé ==> 13



#----------------------------------------palette couleurs--------------------------------------
colors = {
    'magenta': (144/255, 39/255, 83/255),
    'gold': (196/255, 177/255, 8/255),
    'slate_gray': (108/255, 136/255, 137/255),
    'oxford_blue': (9/255, 7/255, 34/255),
    'almond': (235/255, 215/255, 193/255),
    'chinese_violet': (113/255, 99/255, 126/255),
    'brown': (39/255, 30/255, 22/255),
    'light_slate_gray': (233/255, 237/255, 237/255),
    'less_light_slate_gray': (210/255, 218/255, 218/255),
    'light_gold': (254/255, 252/255, 235/255),
    'less_light_gold': (245/255, 235/255, 224/255),
    'light_chinese_violet': (225/255, 221/255, 228/255),
    'less_light_chinese_violet': (214/255, 210/255, 218/255),
    'pink_lavender': (219/255, 178/255, 209/255),
    'lin' : (250/255, 245/255, 240/255),
}

#-----------------------------------------Axe 1 : Titre----------------------------------------- 
ax16.set_xticks([])
ax16.set_yticks([])  # On désactive l'axe pour qu'il n'y ait pas de graphique
for spine in ax16.spines.values():  # Supprime les bordures des subplots
    spine.set_color('white')
ax16.set_facecolor(colors['slate_gray']) 
ax16.text(0.5, 0.5, #positions x, y 

        'Recherche des facteurs influant sur la performance',  

        fontsize=12, #taille 

        color='white', #couleur 

        ha='center', #alignement horizontal 

        va='center', #alignement vertical 
        
        fontweight='bold'
        )

#------------------------- Axe 2 : #matrice engagement satisfaction performance----------------------

ax17.set_title('Matrice de corrélation',color=colors['brown'],weight='bold')
perf_sat_eng = employes_actifs.loc[:,['EngagementSurvey', 'EmpSatisfaction','PerfScoreID']] 
perf_sat_eng=perf_sat_eng.rename(columns={'EngagementSurvey':'Engagement',
                                          'EmpSatisfaction':'Satisfaction', 
                                          'PerfScoreID':'Performance' })
perf_sat_eng
corr_matrix = perf_sat_eng.corr()
# Sélectionner les corrélations
corr_series = corr_matrix.loc[['Performance', 'Engagement'], ['Satisfaction', 'Engagement']]
# Créer la heatmap
sns.heatmap(corr_series, annot=True, cmap='vlag', fmt='.1f', vmin=0, vmax=1, ax=ax17)



#---------------------------- Axe 3 : box plot engagement performance---------------------------
# Création du relplot
sns.boxplot(
    x='engagement_class', 
    y='PerfScoreID', 
    width=0.7, 
    notch=True,
    boxprops={"facecolor": colors['gold']},
    medianprops={"color": colors['magenta'], "linewidth": 2},  
    data=employes_actifs,
    ax=ax18
)

ax18.set_title('Corrélation performance et engagement',color=colors['brown'],weight='bold')
ax18.set_xlabel('Score engagement', color=colors['brown'], weight='bold')
ax18.set_ylabel('Score performance', color=colors['brown'])   
ax18.set_facecolor(colors['light_slate_gray'])  
for spine in ax18.spines.values(): 
    spine.set_color(colors['magenta'])#couleur bordures
    spine.set_linewidth(0.3)
ax18.grid(True, axis='x', which='major', linewidth=0.2, color=(144/255,39/255,83/255))
ax18.tick_params(axis='both', 
                 labelcolor=colors['brown'], 
                 color=colors['magenta'])#couleur des axes


#------------------------------------ Ax 4/5 : texte--------------------------------------------------
wrapped9=textwrap.fill(f" Il n'y a pas de corrélation significative entre la satisfaction et la performance, ni entre la satisfaction et l'engagement.",
                       width=45)
wrapped10=textwrap.fill(f"Cependant, cette matrice met en évidence une corrélation positive modérée entre l'engagement et la performance (0.6). Ceci suggère que l'engagement est un facteur plus important pour la performance que la satisfaction.",
                       width=45)
wrapped_combo2=wrapped9+'\n\n'+wrapped10
ax19.text(-0.025, 0.5,
          wrapped_combo2,
          ha='left',
          va='center',
          fontsize=11,
          color=colors['brown'])
ax19.set_xticks([])
ax19.set_yticks([])  # On désactive l'axe pour qu'il n'y ait pas de graphique
for spine in ax19.spines.values():  # Supprime les bordures des subplots
    spine.set_color(colors['lin'])
ax19.set_facecolor('white')

wrapped11=textwrap.fill(f"Ce boxplot visualise la relation entre les niveaux d'engagement et la performance. On observe une tendance à la hausse de la médiane des scores de performance avec l'augmentation du niveau d'engagement.",
                       width=45)
wrapped12=textwrap.fill(f"Un niveau d'engagement de 3 semble être un objectif à atteindre pour optimiser la performance.",
                       width=45)
wrapped_combo3=wrapped11+'\n\n'+wrapped12
ax20.text(-0.025,0.5,
          wrapped_combo3,
          ha='left',
          va='center',
          fontsize=11,
          color=colors['brown'])
ax20.set_xticks([])
ax20.set_yticks([])  # On désactive l'axe pour qu'il n'y ait pas de graphique
for spine in ax20.spines.values():  # Supprime les bordures des subplots
    spine.set_color('white')
ax20.set_facecolor('white')




#-------------------------------Axe 6 : salaire median engagement performance--------------------------
# Tracer les deux nuages de points
sns.lineplot(data=salaire_engage, 
                x='salary_class', 
                y='PerfScoreID',
                ax=ax21,
                color=colors['oxford_blue'],
                label='performance',
                legend=False,
                )
ax21b = ax21.twinx()
sns.lineplot(data=salaire_engage, 
                x='salary_class', 
                y='engagement_class',  
                ax=ax21b,
                color=colors['magenta'],
                legend=False,
                label='engagement')

#titre
wrapped13=textwrap.fill(f"Engagement et performance selon le salaire",
                       width=35)

ax21.set_title(wrapped13,
               weight='bold', color=colors['brown']) 

ax21.set_facecolor(colors['lin'])

# Axe Y principal
ax21.set_ylabel('Performance', color=colors['oxford_blue'])  
ax21.tick_params(axis='both', labelcolor=colors['oxford_blue'], color=colors['oxford_blue'])
ax21.set_ylim(0, 4.5)

    
# Axe Y secondaire

ax21b.set_ylabel('Engagement', color=colors['magenta'])  
ax21b.tick_params(axis='y', labelsize=10, colors=colors['magenta'], labelcolor=colors['magenta'])
ax21b.set_ylim(0, 5.5)
    
#Forcer les 10 valeurs de l’axe X
ax21.set_xlabel('Salaire', color=colors['brown'], weight='bold') 
salary_ticks = [55250, 75750, 96250, 116750, 137250, 157750, 178250, 198250, 219250, 239750]
ax21.set_xticks(salary_ticks)
font_tick = {'family': 'Arial', 'size': 9, 'color': colors['brown']}

# Axe X: formatage
ax21.set_xticklabels([f"{int(x/1000)}K" for x in salary_ticks], fontdict=font_tick)

#bordures
for spine in ax21.spines.values():
    spine.set_color(colors['light_slate_gray'])
    spine.set_linewidth(0.5)

for spine in ax21b.spines.values():
    spine.set_color(colors['light_slate_gray'])
    spine.set_linewidth(0.5)

    
#   affichage grille
ax21.grid(True, axis='both', which='both', linewidth=0.3, color=colors['less_light_slate_gray'])
    
     
#-----------------------------------Axe 7 : nombre de projets----------------------------------------- 
# Tracer les deux nuages de points
sns.lineplot(data=projets_engage, 
                x='SpecialProjectsCount', 
                y='PerfScoreID',
                ax=ax22,
                color=(9/255,7/255,34/255),
                legend=False,
                label='performance')
ax22b = ax22.twinx()
sns.lineplot(data=projets_engage, 
                x='SpecialProjectsCount', 
                y='engagement_class',  
                ax=ax22b,
                color=(144/255,39/255,83/255),
                legend=False,
                label='engagement')

#titre
wrapped14=textwrap.fill(f"Engagement et performance selon le nombre de projets",
                       width=35)

ax22.set_title(wrapped14,
               weight='bold', color=colors['brown']) 

ax22.set_facecolor(colors['lin'])

# Axe Y principal
ax22.set_ylabel('Performance', color=colors['oxford_blue'])  
ax22.tick_params(axis='both', labelcolor=colors['oxford_blue'], color=colors['oxford_blue'])
ax22.set_ylim(0, 4.5)
    
# Axe Y secondaire

ax22b.set_ylabel('Engagement', color=colors['magenta'])  
ax22b.tick_params(axis='y', labelsize=10, colors=colors['magenta'], labelcolor=colors['magenta'])
ax22b.set_ylim(0, 5.5)
    
#Axe X
ax22.set_xlabel('Nombre de projets', color=colors['brown'],weight='bold') 


#bordures
for spine in ax22.spines.values():
    spine.set_color(colors['light_slate_gray'])
    spine.set_linewidth(0.5)

for spine in ax22b.spines.values():
    spine.set_color(colors['light_slate_gray'])
    spine.set_linewidth(0.5)

    
#   affichage grille
ax22.grid(True, axis='both', which='both', linewidth=0.3, color=colors['less_light_slate_gray'])

#-------------------------------------------- Axe 8 : generation-----------------------------------

#tracé
sns.stripplot(x=annalyse_engagement['Generation'],
           y=annalyse_engagement['PerfScoreID'],
           color=colors['oxford_blue'], 
           dodge=True, 
           alpha=.2, 
           legend=False,
           label='performance',
           ax=ax23
)

sns.pointplot(data=generation_engage, 
                x='Generation', 
                y='PerfScoreID',
                linestyle="none", 
                errorbar=None,
                marker="_", 
                markersize=10, 
                markeredgewidth=3,
                color=colors['oxford_blue'],
                legend=False,
                ax=ax23
                )

ax23b = ax23.twinx()

sns.stripplot(x=annalyse_engagement['Generation'],
           y=annalyse_engagement['EngagementSurvey'], 
           color=colors['magenta'], 
           dodge=True, 
           alpha=.25, 
           legend=False,
           label='engagement',
           ax=ax23b         
)

sns.pointplot(data=generation_engage, 
                x='Generation', 
                y='engagement_class',
                linestyle="none", 
                errorbar=None,
                marker="_", 
                markersize=10, 
                markeredgewidth=3,
                legend=False,
                ax=ax23b,
                color=colors['magenta']
                )

wrapped17=textwrap.fill(f"Engagement et performance selon la génération",
                       width=35)

ax23.set_title(wrapped17,
               weight='bold', color=colors['brown']) 

ax23.set_facecolor(colors['lin'])

# Axe Y principal
ax23.set_ylabel('Performance', color=colors['oxford_blue'])  
ax23.tick_params(axis='both', labelcolor=colors['oxford_blue'], color=colors['oxford_blue'])
ax23.set_ylim(0, 4.5)
    
# Axe Y secondaire
ax23b.set_ylabel('Engagement', color=colors['magenta'])  
ax23b.tick_params(axis='y', labelsize=10, colors=colors['magenta'], labelcolor=colors['magenta'])
ax23b.set_ylim(0, 5.5)
    
#Axe X
ax23.set_xlabel('Générations', color=colors['brown'], weight='bold')
generations=['Baby boomers', 'X generation', 'Millennials', 'Z generation', 'Alpha generation'] 
ax23.set_xticklabels(generations,fontdict=font_tick, rotation=45)

#bordures
for spine in ax23.spines.values():
    spine.set_color(colors['light_slate_gray'])
    spine.set_linewidth(0.5)

for spine in ax23b.spines.values():
    spine.set_color(colors['light_slate_gray'])
    spine.set_linewidth(0.5)

    
#   affichage grille
ax23.grid(True, axis='both', which='both', linewidth=0.3, color=colors['less_light_slate_gray'])
    


#---------------------------------------------- Axe 9 : département-------------------------------------
#tracé
sns.stripplot(x=annalyse_engagement['Department'],
           y=annalyse_engagement['PerfScoreID'],
           color=colors['oxford_blue'], 
           dodge=True, 
           alpha=.2, 
           legend=False,
           label='performance',
           ax=ax24
)

sns.pointplot(data=department_engage, 
                x='Department', 
                y='PerfScoreID',
                linestyle="none", 
                errorbar=None,
                marker="_", 
                markersize=30, 
                markeredgewidth=3,
                color=colors['oxford_blue'],
                legend=False,
                ax=ax24
                )

ax24b = ax24.twinx()

sns.stripplot(x=annalyse_engagement['Department'],
           y=annalyse_engagement['EngagementSurvey'], 
           color=colors['magenta'], 
           dodge=True, 
           alpha=.25, 
           legend=False,
           label='engagement',
           ax=ax24b         
)

sns.pointplot(data=department_engage, 
                x='Department', 
                y='engagement_class',
                linestyle="none", 
                errorbar=None,
                marker="_", 
                markersize=30, 
                markeredgewidth=3,
                legend=False,
                ax=ax24b,
                color=colors['magenta'],
                )

#titre
wrapped16=textwrap.fill(f"Engagement et performance selon le département",
                       width=35)

ax24.set_title(wrapped16,
               weight='bold', color=colors['brown']) 

ax24.set_facecolor(colors['lin'])

# Axe Y principal
ax24.set_ylabel('Performance', color=colors['oxford_blue'])  
ax24.tick_params(axis='both', labelcolor=colors['oxford_blue'], color=colors['oxford_blue'])
ax24.set_ylim(0, 4.5)
    
# Axe Y secondaire
ax24b.set_ylabel('Engagement', color=colors['magenta'])  
ax24b.tick_params(axis='y', labelsize=10, colors=colors['magenta'], labelcolor=colors['magenta'])
ax24b.set_ylim(0, 5.5)
    
#Axe X
ax24.set_xlabel('Département', color=colors['brown'], weight='bold')
departement_ticks = [0, 1, 2, 3, 4, 5]
departement_labels = ['Prod',  'Software eng.','IT/IS', 'Admin of.', 'Sales', 'Executive of.']
ax24.set_xticks(departement_ticks)
ax24.set_xticklabels(departement_labels, fontdict=font_tick, rotation=45)
ax24.tick_params(axis='x', labelrotation=45)  # Pivote les labels à 45 degrés



#bordures
for spine in ax24.spines.values():
    spine.set_color(colors['light_slate_gray'])
    spine.set_linewidth(0.5)

for spine in ax24b.spines.values():
    spine.set_color(colors['light_slate_gray'])
    spine.set_linewidth(0.5)

    
#   affichage grille
ax24.grid(True, axis='both', which='both', linewidth=0.3, color=colors['less_light_slate_gray'])
    



#------------------------------------------Axe 10 : manageur-----------------------------------
wrapped17=textwrap.fill(f"Engagement et performance selon le manageur",
                       width=35)

#tracé
sns.stripplot(x=annalyse_engagement['ManagerID'],
           y=annalyse_engagement['PerfScoreID'],
           color=colors['oxford_blue'], 
           dodge=True, 
           alpha=.2, 
           legend=False,
           label='performance',
           ax=ax25
)

sns.pointplot(data=manager_engage, 
                x='ManagerID', 
                y='PerfScoreID',
                linestyle="none", 
                errorbar=None,
                marker="_", 
                markersize=10, 
                markeredgewidth=3,
                color=colors['oxford_blue'],
                legend=False,
                ax=ax25
                )

ax25b = ax25.twinx()

sns.stripplot(x=annalyse_engagement['ManagerID'],
           y=annalyse_engagement['EngagementSurvey'], 
           color=colors['magenta'], 
           dodge=True, 
           alpha=.25, 
           legend=False,
           label='engagement',
           ax=ax25b         
)

sns.pointplot(data=manager_engage, 
                x='ManagerID', 
                y='engagement_class',
                linestyle="none", 
                errorbar=None,
                marker="_", 
                markersize=10, 
                markeredgewidth=3,
                ax=ax25b,
                legend=False,
                color=colors['magenta']
                )

#titre
ax25.set_title(wrapped16,
               weight='bold', color=colors['brown']) 

ax25.set_facecolor(colors['lin'])

# Axe Y principal
ax25.set_ylabel('Performance', color=colors['oxford_blue'])  
ax25.tick_params(axis='both', labelcolor=colors['oxford_blue'], color=colors['oxford_blue'])
ax25.set_ylim(0, 4.5)
    
# Axe Y secondaire
ax25b.set_ylabel('Engagement', color=colors['magenta'])  
ax25b.tick_params(axis='y', labelsize=10, colors=colors['magenta'], labelcolor=colors['magenta'])
ax25b.set_ylim(0, 5.5)
    
#Axe X
ax25.set_xlabel('Manager ID', color=colors['brown'], weight='bold') 


#bordures
for spine in ax25.spines.values():
    spine.set_color(colors['light_slate_gray'])
    spine.set_linewidth(0.5)

for spine in ax25b.spines.values():
    spine.set_color(colors['light_slate_gray'])
    spine.set_linewidth(0.5)

    
#   affichage grille
ax25.grid(True, axis='both', which='both', linewidth=0.3, color=colors['less_light_slate_gray'])
    

#-------------------------------Axe 11 : type de position (cadre/non cadre)----------------------------
#tracé des deux axes
sns.stripplot(x=annalyse_engagement['type_poste'],
           y=annalyse_engagement['PerfScoreID'],
           color=colors['oxford_blue'], 
           dodge=True, 
           alpha=.2, 
           legend=False,
           label='performance',
           ax=ax26
)

sns.pointplot(data=cadres_engage, 
                x='type_poste', 
                y='PerfScoreID',
                linestyle="none", 
                errorbar=None,
                marker="_", 
                markersize=30, 
                markeredgewidth=3,
                color=colors['oxford_blue'],
                legend=False,
                ax=ax26
                
                )

ax26b = ax26.twinx()

sns.stripplot(x=annalyse_engagement['type_poste'],
           y=annalyse_engagement['EngagementSurvey'], 
           color=colors['magenta'], 
           dodge=True, 
           alpha=.25, 
           legend=False,
           label='engagement',
           ax=ax26b         
)

sns.pointplot(data=cadres_engage, 
                x='type_poste', 
                y='engagement_class',
                linestyle="none", 
                errorbar=None,
                marker="_", 
                markersize=30, 
                markeredgewidth=3,
                ax=ax26b,
                color=colors['magenta'],
                legend=False
                )

#titre
wrapped15=textwrap.fill(f"Engagement et performance selon le type de poste",
                       width=35)

ax26.set_title(wrapped15,
               weight='bold', color=colors['brown']) 

ax26.set_facecolor(colors['lin'])

# Axe Y principal
ax26.set_ylabel('Performance', color=colors['oxford_blue'])  
ax26.tick_params(axis='both', labelcolor=colors['oxford_blue'], color=colors['oxford_blue'])
ax26.set_ylim(0, 4.5)
    
# Axe Y secondaire
ax26b.set_ylabel('Engagement', color=colors['magenta'])  
ax26b.tick_params(axis='y', labelsize=10, colors=colors['magenta'], labelcolor=colors['magenta'])
ax26b.set_ylim(0, 5.5)
    
#Axe X
ax26.set_xlabel('Type de poste', color=colors['brown'], weight='bold') 
ax26.set_xticks([0.15, 0.85])

#bordures
for spine in ax26.spines.values():
    spine.set_color(colors['light_slate_gray'])
    spine.set_linewidth(0.5)

for spine in ax26b.spines.values():
    spine.set_color(colors['light_slate_gray'])
    spine.set_linewidth(0.5)

    
#   affichage grille
ax26.grid(True, axis='both', which='both', linewidth=0.3, color=colors['less_light_slate_gray'])

#----------------------------------Axe 12 texte---------------------------------------------------------------

wrapped18=textwrap.fill(f"Cette analyse montre une corrélation modérée (r = 0.6) entre engagement et performance, suggérant une influence positive de l'engagement.",
                       width=130)
wrapped19=textwrap.fill(f"Cependant, cette relation reste partielle et ne permet pas, à elle seule, d’expliquer les variations observées.",
                       width=130)

wrapped20=textwrap.fill(f"Aucun facteur analysé (salaire, projets, génération, etc.) n'explique les motivateurs d'engagement ou de performance pour les employés.",
                       width=130)
wrapped21=textwrap.fill(f"Une enquête qualitative est recommandée, afin d'identifier si d'autres variables non mesurées (conditions de travail, reconnaissance, etc.) peuvent influer l'engagement des employés et donc leur niveau de performance.",
                       width=130)
wrapped22=textwrap.fill(f"En résumé, si cette étude pose des premières bases sur le lien entre engagement et performance, elle souligne surtout les limites des indicateurs actuellement disponibles pour expliquer ce lien. Ceci ouvre la voie à de futures investigations plus ciblées.",
                       width=130)
wrapped_combo4=wrapped18+'\n\n'+wrapped19+'\n\n'+wrapped20+'\n\n'+wrapped21+'\n\n'+wrapped22
ax27.text(0.02, 0.5,
          wrapped_combo4,
          ha='left',
          va='center',
          fontsize=11,
          color=colors['brown'])
ax27.set_xticks([])
ax27.set_yticks([])  # On désactive l'axe pour qu'il n'y ait pas de graphique
for spine in ax27.spines.values():  # Supprime les bordures des subplots
    spine.set_color('white')
ax27.set_facecolor(colors['lin'])

#----------------------------------Axe 13 note---------------------------------------------------------------
wrapped23=textwrap.fill(f"Deux managers présentent des niveaux d'engagement et de performance inférieurs à la médiane, nécessitant une attention particulière pour identifier et corriger d'éventuels dysfonctionnements ou besoins spécifiques.",
                       width=130)
ax28.text(0.05, 0.5,
          wrapped23,
          ha='left',
          va='center',
          fontsize=8,
          color=colors['brown'])
ax28.set_xticks([])
ax28.set_yticks([])  # On désactive l'axe pour qu'il n'y ait pas de graphique
for spine in ax28.spines.values():  # Supprime les bordures des subplots
    spine.set_color('white')
ax28.set_facecolor(colors['light_slate_gray'])

#----------------------------------Axe 12 Recap---------------------------------------------------------------

wrapped24=textwrap.fill(f"Jusqu'à une certaine mesure, le niveau d'engagement a un effet positif sur la performance.",
                       width=130)
wrapped25=textwrap.fill(f"Aucun des indicateurs analysés (salaire, projets, génération, département, manager...) n’explique clairement ce qui favorise l’engagement.",
                       width=130)

wrapped26=textwrap.fill(f"Deux managers présentent des scores d’engagement et de performance en dessous de la médiane : à investiguer.",
                       width=130)
wrapped27=textwrap.fill(f"Des facteurs non mesurés semblent déterminants.",
                       width=130)
wrapped28=textwrap.fill(f"Une enquête complémentaire est recommandée pour identifier les leviers réels de l’engagement.",
                       width=130)
wrapped_combo5=wrapped24+'\n\n'+wrapped25+'\n\n'+wrapped26+'\n\n'+wrapped27+'\n\n'+wrapped28
ax29.text(0.02, 0.5,
          wrapped_combo5,
          ha='left',
          va='center',
          fontsize=11,
          color=colors['brown'])
ax29.set_xticks([])
ax29.set_yticks([])  # On désactive l'axe pour qu'il n'y ait pas de graphique
for spine in ax29.spines.values():  # Supprime les bordures des subplots
    spine.set_color(colors['lin'])
ax29.set_facecolor(colors['lin'])





# Ajuster les espacements
plt.tight_layout(pad=2)
# Afficher la figure
plt.show()

