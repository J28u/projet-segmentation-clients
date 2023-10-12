import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer, ColumnTransformer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import make_column_transformer
import ssl
from skimage import io
from math import ceil, pi


def display_image(url: str, title: str, fig_size: tuple):
    """
    Affiche une image à partir de son url

    Positional arguments : 
    -------------------------------------
    url : str : jurl de l'image à afficher 
    title : str : titre à afficher au dessus de l'image
    figsize : tuple : taille de la zone d'affichage de l'image (largeur, hauteur)
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    img = io.imread(url)
    plt.figure(figsize=fig_size)
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontsize=20, fontname='Corbel', pad=20)
    plt.imshow(img)

    plt.show()


def plot_donut(dataset: pd.DataFrame, categ_var: str, title: str, figsize: tuple, text_color='#595959',
               colors={'outside': sns.color_palette('Set2')}, nested=False, sub_categ_var=None, labeldistance=1.1,
               textprops={'fontsize': 20, 'color': '#595959', 'fontname': 'Open Sans'}):
    """
    Affiche un donut de la répartition d'une variable qualitative

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant les valeurs à afficher
    categ_var : str : nom de la colonne contenant les valeurs de la variable qualitative
    
    palette : strings : nom de la palette seaborn à utiliser
    title : str : titres du graphique
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    
    Optionnal arguments : 
    -------------------------------------
    text_color : str : couleur du texte
    colors : dict : couleurs du donut extérieur et couleurs du donut intérieur
    nested : bool : créer un double donut ou non
    sub_categ_var : str : nom de la colonne contenant les catégories à afficher dans le donut intérieur
    labeldistance : float : distance à laquelle placer les labels du donut extérieur
    textprops : dict : personnaliser les labels du donut extérieur (position, couleur ...)
    """
    with plt.style.context('seaborn-white'):
        sns.set_theme(style='whitegrid')
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title, fontname='Corbel', fontsize=30)
        plt.rcParams.update(
            {'axes.labelcolor': text_color, 'axes.titlecolor': text_color, 'legend.labelcolor': text_color,
             'axes.titlesize': 16, 'axes.labelpad': 10})

    pie_series = dataset[categ_var].value_counts(sort=False, normalize=True)
    patches, texts, autotexts = ax.pie(pie_series, labels=pie_series.index, autopct='%.0f%%', pctdistance=0.85,
                                       colors=colors['outside'], labeldistance=labeldistance,
                                       textprops=textprops,
                                       wedgeprops={'edgecolor': 'white', 'linewidth': 2})

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(16)

    centre_circle = plt.Circle((0, 0), 0.7, fc='white')

    if nested:
        inside_pie_series = dataset[sub_categ_var].value_counts(sort=False, normalize=True)
        patches_sub, texts_sub, autotexts_sub = ax.pie(inside_pie_series, autopct='%.0f%%', pctdistance=0.75,
                                                       colors=colors['inside'], radius=0.7,
                                                       wedgeprops={'edgecolor': 'white', 'linewidth': 2})

        for autotext in autotexts_sub:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(14)

        plt.legend(patches_sub, inside_pie_series.index, title=sub_categ_var, fontsize=14, title_fontsize=16, loc=0)
        centre_circle = plt.Circle((0, 0), 0.4, fc='white')

    ax.axis('equal')
    ax.add_artist(centre_circle)

    plt.tight_layout()
    plt.show()


def plot_empirical_distribution(column_to_plot: pd.Series, color: tuple, titles: dict, figsize: tuple, vertical=True):
    """
    Affiche un histogramme de la distribution empirique de la variable choisie

    Positional arguments : 
    -------------------------------------
    column_to_plot : np.array : valeurs observées
    color : tuple : couleur des barres de l'histogramme
    titles : dict : titres du graphique et des axes - ex: {'chart_title': 'c', 'y_title': 'b', 'x_title': 'a'}
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    
    Optional arguments : 
    -------------------------------------
    vertical : bool : True pour afficher l'histogramme à la verticale, False à l'horizontale
    """

    plt.figure(figsize=figsize)
    rgb_text = sns.color_palette('Greys', 15)[12]

    with sns.axes_style('white'):
        if vertical:
            ax = sns.histplot(column_to_plot, stat="percent", discrete=True, shrink=.9, edgecolor=color, linewidth=3,
                              alpha=0.4, color=color)
            ax.set(yticklabels=[])
            sns.despine(left=True)
        else:
            ax = sns.histplot(y=column_to_plot, stat="percent", discrete=True, shrink=.6, edgecolor=color, linewidth=3,
                              alpha=0.4, color=color)
            ax.set(xticklabels=[])
            sns.despine(bottom=True)

    for container in ax.containers:
        ax.bar_label(container, size=18, fmt='%.1f%%', fontname='Open Sans', padding=5)

    plt.title(titles['chart_title'], size=24, fontname='Corbel', pad=40, color=rgb_text)
    plt.ylabel(titles['y_title'], fontsize=20, fontname='Corbel', color=rgb_text)
    ax.set_xlabel(titles['x_title'], rotation=0, labelpad=20, fontsize=20, fontname='Corbel', color=rgb_text)
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()
    plt.show()


def build_frequency_df_with_thresh(dataset: pd.DataFrame, column_to_count: str, thresh: float, other_label: str):
    """
    Retourne un dataframe avec la fréquence empirique de chaque modalité et regroupe les modalités peu représentées
    (i.e. fréquence < limite choisie)

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant la variable à étudier
    column_to_count : str : nom de la colonne contenant la variable à étudier
    thresh : float : fréquence limite en dessous de laquelle les modalités peu représentées sont regroupées
    other_label : str : nom de la nouvelle modalité (qui regroupe les modalités peu représentées)
    """
    frequency_df = dataset[[column_to_count]].copy()
    effectifs = dataset[column_to_count].value_counts(normalize=True).to_dict()
    frequency_df['frequency'] = frequency_df.apply(lambda row: effectifs[row[column_to_count]], axis=1)

    other = other_label.format(str(thresh * 100))

    frequency_df[column_to_count] = frequency_df.apply(
        lambda row: other if (row['frequency'] < thresh) else row[column_to_count], axis=1)
    frequency_df = frequency_df.sort_values('frequency', ascending=False)

    return frequency_df


def clean_duplicates(dataset: pd.DataFrame, key_column: str, date_column: str, verbose=True):
    """
    Retourne un dataframe sans doublons (garde l'individu le plus récent parmis les doublons)

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : dataframe dont on souhaite retirer les doublons
    key_column : str : nom de la colonne contenant la clé d'unicité (doublons = individus avec la même valeur
    dans cette colonne)
    date_column : str : nom de la colonne contenant les dates utilisées pour garder l'individu le plus récent parmis
    les doublons
    """

    duplicates_all = dataset.loc[dataset.duplicated(subset=key_column, keep=False)].copy()
    if verbose:
        print('Il y a', duplicates_all.shape[0] - len(duplicates_all[key_column].unique()), 'doublon(s)')

    if duplicates_all.empty:
        return dataset

    subset = dataset.copy()
    duplicates_all = duplicates_all.sort_values(date_column, ascending=False)

    duplicates_to_drop = duplicates_all.loc[duplicates_all.duplicated(subset=key_column)]
    subset.drop(index=duplicates_to_drop.index.values, inplace=True)

    if verbose:
        print(dataset.shape[0] - subset.shape[0], 'ligne(s) supprimée(s)')
        print('Il reste', subset.loc[subset.duplicated(subset=key_column)].shape[0], 'doublon(s)')

    return subset


def plot_screeplot(pca: sklearn.decomposition.PCA, n_components: int, figsize: tuple, titles: dict, color_bar: str,
                   legend_x: float, legend_y: float):
    """
    Affiche l'éboulis des valeurs propres avec la courbe de la somme cumulée des inertie

    Positional arguments : 
    -------------------------------------
    pca : sklearn.decomposition.PCA : modèle d'analyse en composantes principales (déjà entrainé)
    n_components : int : nombre d'axes d'inertie
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    titles : dict : titres du graphique et des axes - ex: {'chart_title': 'blabla', 'y_title': 'blabla', 'x_title': 'a'}
    color_bar : str : couleur utilisée pour le diagramme à bar
    """
    scree = (pca.explained_variance_ratio_ * 100).round(2)
    scree_cum = scree.cumsum().round(2)
    x_list = range(1, n_components + 1)

    rgb_text = sns.color_palette('Greys', 15)[12]

    with plt.style.context('seaborn-white'):
        sns.set_theme(style='whitegrid')
        plt.rcParams.update({'xtick.labelsize': 16, 'ytick.labelsize': 16, 'axes.titlesize': 18})
        fig, ax = plt.subplots(figsize=figsize)

        ax.bar(x_list, scree, color=color_bar)
        ax.set_xticks(x_list)
        ax.plot(x_list, scree_cum, color='coral', marker='o', markerfacecolor='white', markeredgecolor='coral',
                markersize=18, markeredgewidth=2)
        ax.text(legend_x, legend_y, "variance cumulée", fontsize=20, color='coral', fontname='Corbel')

    plt.title(titles['chart_title'], fontname='Corbel', fontsize=23, pad=20, color=rgb_text)
    plt.ylabel(titles['y_label'], color=rgb_text, fontsize=18)
    plt.xlabel(titles['x_label'], color=rgb_text, fontsize=18)
    plt.grid(False, axis='x')

    plt.show()


def plot_heatmap(data: pd.DataFrame, vmax: float, titles: dict, figsize: tuple, fmt: str, annotation=True, vmin=0.0,
                 palette="rocket_r", square=False):
    """
    Affiche une heatmap 

    Positional arguments : 
    -------------------------------------
    data : pd.DataFrame : jeu de données contenant les valeurs pour colorer la heatmap
    vmax : float : valeur maximale de l'échelle des couleurs
    titles : dict : titres du graphique et des axes - ex: {'chart_title': 'blabla', 'y_title': 'blabla', 'x_title': 'a'}
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    fmt : str : format annotations 
    
    Optional arguments : 
    -------------------------------------
    annotation : bool or pd.DataFrame : valeurs à afficher dans les cases de la heatmap - True : utilise data 
    vmin : float : valeur minimale de l'échelle des couleurs
    palette : str : couleurs de la heatmap
    square : bool : affiche les cases de la heatmap en carré
    """

    plt.figure(figsize=figsize)

    with sns.axes_style('white'):
        ax = sns.heatmap(data, annot=annotation, vmin=vmin, vmax=vmax, cmap=sns.color_palette(palette, as_cmap=True),
                         annot_kws={"fontsize": 16, 'fontname': 'Open Sans'}, linewidth=1, linecolor='w', fmt=fmt,
                         square=square)

    if fmt == 'd':
        for t in ax.texts:
            t.set_text("{:_}".format(int(t.get_text())))
    plt.title(titles['chart_title'], size=28, fontname='Corbel', pad=40)
    plt.xlabel(titles['x_title'], fontname='Corbel', fontsize=24, labelpad=20)
    ax.xaxis.set_label_position('top')
    plt.ylabel(titles['y_title'], fontname='Corbel', fontsize=24, labelpad=20)
    plt.tick_params(axis='both', which='major', labelsize=14, labeltop=True, labelbottom=False)

    plt.show()


def adjust_text(texts: list):
    """
    Retourne la liste des annotations d'un graphique avec des coordonnées ajustées pour éviter que les annotations
    ne se superposent

    Positional arguments : 
    -------------------------------------
    texts : list of matplotlib Text objects : liste des annotations à ajuster
    """
    for index, text in enumerate(texts):
        for text_next in texts[index + 1:]:
            x_text = text.get_position()[0]
            y_text = text.get_position()[1]

            x_text_next = text_next.get_position()[0]
            y_text_next = text_next.get_position()[1]

            if abs(x_text - x_text_next) < 0.12 and abs(y_text - y_text_next) < 0.12:
                text.set_position((x_text - 0.20, y_text - 0.20))
    return texts


def plot_correlation_circle(pca: sklearn.decomposition.PCA, x: int, y: int, figsize: tuple, features: [str],
                            arrow_color: str):
    """
    Affiche le cercle des corrélations

    Positional arguments : 
    -------------------------------------
    pca : sklearn.decomposition.PCA : modèle d'analyse en composantes principales (déjà entrainé)
    x : int : index de l'axe d'inertie affiché en abscisse 
    y : int : index de l'axe d'inertie affiché en ordonnée
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    features : list of strings : liste des variables initiales à projeter sur le cercle des corrélations
    arrow_color : str : couleur utilisée pour les flèches
    """
    with plt.style.context('seaborn-white'):
        plt.rcParams['axes.labelpad'] = '20'
        sns.set_theme(style='ticks', palette='colorblind')
        fig, ax = plt.subplots(figsize=figsize)
        texts = []

        for i in range(0, pca.components_.shape[1]):  # pour chaque variable on trace une flèche
            ax.arrow(0, 0,  # abscisse, ordonnée origine de la flèche
                     pca.components_[x, i], pca.components_[y, i],
                     # abscisse, ordonnée du bout de la flèche (coef correlation avec Fx, Fy)
                     head_width=0.07,
                     head_length=0.07,
                     width=0.01,
                     linewidth=.5,
                     color=arrow_color)

            texts.append(plt.text(pca.components_[x, i] + 0.02,
                                  pca.components_[y, i] + 0.05,
                                  features[i], fontname='Corbel', fontsize=14,
                                  color='black'))  # ajoute une étiquette avec le nom de la variable

        texts = adjust_text(texts)

        # affichage des axes du graph
        plt.plot([-1, 1], [0, 0], color='#D3D3D3', ls='--')
        plt.plot([0, 0], [-1, 1], color='#D3D3D3', ls='--')

        # nom des axes, avec le pourcentage d'inertie expliqué par l'axe
        plt.xlabel('F{} ({}%)'.format(x + 1, round(100 * pca.explained_variance_ratio_[x], 1)), fontsize=16,
                   fontname='Corbel')
        plt.ylabel('F{} ({}%)'.format(y + 1, round(100 * pca.explained_variance_ratio_[y], 1)), fontsize=16,
                   fontname='Corbel')

        plt.title("Cercle des corrélations (F{} et F{})".format(x + 1, y + 1), fontsize=20, fontname='Corbel')

        # Trace le cercle
        an = np.linspace(0, 2 * np.pi, 100)  # renvoie une liste de 100 angles entre 0° et 360° régulièrement espacés
        plt.plot(np.cos(an), np.sin(an))  # relie les points du cercle (abscisse = cos(angle), ordonnée=sin(angle))
        plt.axis('equal')
        sns.despine()

        plt.show()


def plot_heatmap_correlation_matrix(correlation_matrix: pd.DataFrame, title: str, figsize: tuple, palette: str):
    """
    Affiche la matrice de corrélation sous forme de heatmap

    Positional arguments : 
    -------------------------------------
    correlation_matrix : pd.DataFrame : matrice de corrélation
    titles : str : titres du graphique
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    palette : str : palette seaborn utilisée
    """

    sns.set_theme(style='white')
    plt.figure(figsize=figsize)

    mask_upper_triangle = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    ax = sns.heatmap(correlation_matrix, annot=True, mask=mask_upper_triangle, vmin=-1, vmax=1, center=0,
                     cmap=sns.color_palette(palette, as_cmap=True),
                     annot_kws={"fontsize": 16, 'fontname': 'Open Sans'},
                     cbar_kws={"shrink": .5},
                     linewidth=1.5, linecolor='w', fmt='.2f', square=True)

    plt.title(title, size=20, fontname='Corbel', pad=20)

    plt.show()


def fit_transform_pca(dataset: pd.DataFrame, n_components: int, scale=True, transform=False):
    """
    Retourne un modèle d'analyse en composantes principales entrainé, et optionellement les données après PCA

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données d'entrainement
    n_components : int : nombre de composantes principales
    
    Optionnal arguments : 
    -------------------------------------
    scale : bool : True = centrer/réduire les données avant PCA
    transfom : bool : False = ne renvoie pas les données après PCA
    """
    x = dataset.copy()
    if scale:
        scaler = StandardScaler()
        x = scaler.fit_transform(x.values)

    pca = PCA(n_components=n_components)
    pca.fit(x)
    explained_variance_cum = pca.explained_variance_ratio_.cumsum()

    print('Pourcentage de variance expliquée par la première composante : {:.2f}%'.format(
        explained_variance_cum[0] * 100))
    for i in range(1, n_components):
        print('Pourcentage de variance expliquée par les {} premières composantes ensembles : {:.2f}%'.format(
            i + 1, explained_variance_cum[i] * 100))

    if transform:
        x_proj = pca.transform(x)
        return pca, x_proj

    return pca


def display_dendrogram(x: pd.DataFrame, level_n: int, fig_size=(12, 8), palette='Set2'):
    """
    Affiche un dendrogramme 

    Positional arguments : 
    -------------------------------------
    X : pd.DataFrame : jeu de données dans lequel on souhaite trouver des clusters
    level_n : int : profondeur max du dendrogramme affiché
    
    Optionnal arguments : 
    -------------------------------------
    fig_size : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    palette : str : palette de couleur seaborn à utiliser
    """
    z = linkage(x, method="ward")  # Calcul distance entre points (matrice de distance)

    sns.set_theme(style='white', palette='Set2')
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    _ = dendrogram(z, p=level_n, truncate_mode="lastp", ax=ax)

    plt.title("Clustering Hiérarchique (dendrogramme)")
    plt.xlabel("Nombre de points dans les noeuds")
    plt.ylabel("Distance")
    plt.show()


def display_pca_scatterplot(pca: sklearn.decomposition.PCA, x: int, y: int, x_proj: np.array, nb_components: int,
                            figsize: tuple,
                            with_hue=False, hue_data=None, hue_palette=None):
    """
    Affiche le nuage des individus projeté dans un plan factoriel et retourne le dataframe associé

    Positional arguments : 
    -------------------------------------
    pca : sklearn.decomposition.PCA : modèle d'analyse en composantes principales (déjà entrainé)
    x : int : index de l'axe d'inertie affiché en abscisse 
    y : int : index de l'axe d'inertie affiché en ordonnée
    X_proj : np.array : tableau des individus projetés sur les axes d'inertie 
    nb_components : int : nombres de composantes de l'ACP
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    
    Optional arguments : 
    -------------------------------------
    with_hue : bool : True pour colorer le nuage des individus selon les modalités prises par une variable qualitative
    hue_data : np.array : valeurs prises par la variable qualitative
    hue_palette : list of strings or string : palette seaborn utilisée pour colorer données ou liste de couleurs
    """

    plt.rcParams['axes.labelpad'] = '20'
    sns.set_theme(style='whitegrid', palette='bright')
    plt.figure(figsize=figsize)

    x_proj_df = pd.DataFrame(data=x_proj, columns=['F' + str(s) for s in range(1, nb_components + 1)])

    if with_hue:
        x_proj_df['cluster'] = hue_data
        x_proj_df = x_proj_df.sort_values('cluster')
        sns.scatterplot(data=x_proj_df, x=x_proj_df.columns[x], y=x_proj_df.columns[y], s=50, hue='cluster',
                        palette=hue_palette, alpha=0.6)
    else:
        sns.scatterplot(data=x_proj_df, x=x_proj_df.columns[x], y=x_proj_df.columns[y], s=50)

    x_lim = ceil(x_proj_df['F' + str(x + 1)].abs().max())
    y_lim = ceil(x_proj_df['F' + str(y + 1)].abs().max())
    plt.plot([-x_lim, x_lim], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-y_lim, y_lim], color='grey', ls='--')
    plt.xlim(-x_lim, x_lim)
    plt.ylim(-y_lim, y_lim)

    plt.title('Projection du nuage des individus (F{} et F{})'.format(x + 1, y + 1), fontsize=20, fontname='Corbel',
              pad=10)
    plt.xlabel('F{} ({}%)'.format(x + 1, round(100 * pca.explained_variance_ratio_[x], 1)), fontsize=16,
               fontname='Corbel')
    plt.ylabel('F{} ({}%)'.format(y + 1, round(100 * pca.explained_variance_ratio_[y], 1)), fontsize=16,
               fontname='Corbel')
    sns.despine(left=True, right=True)

    plt.show()

    return x_proj_df


def pairplot_clusters(x: pd.DataFrame, clusters: np.array, palette='Set2'):
    """
    Affiche un nuage de points colorés par cluster pour tous les couples de features possibles

    Positional arguments : 
    -------------------------------------
    x : pd.DataFrame : jeu de données
    clusters : np.array : clusters
    """
    x_copy = x.copy()
    x_copy['cluster'] = clusters

    sns.set_theme(style='whitegrid')
    sns.pairplot(x_copy, hue="cluster", palette=palette, corner=True)
    plt.show()


def plot_boxplot_by_dimension(dataset: pd.DataFrame, dimensions: [str], column_nb: int, title: str, figsize: tuple,
                              y_column=None, sharey=False,
                              top=0.85, wspace=0.2, hspace=1.8):
    """
    Affiche des boxplots (un graphique différent par variable étudiée) 

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant les variables à afficher
    dimensions : list of strings : listes des variables à afficher 
    column_nb : int : nombre de graphique par ligne
    title : str : titre principal (suptitle)
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    
    Optional arguments : 
    -------------------------------------
    y_column : str : nom de la colonne contenant les catégories (pour agréger les données par catégorie et afficher
    plusieurs boxplots par graphique)
    sharey : bool : True pour que les graphiques partagent tous le même axe des ordonnées
    top : float : position de départ des graphiques dans la figure
    wspace : float : largeur de l'espace entre les graphiques
    hspace : float : hauteur de l'espace entre les graphiques
    """

    color_list_text = sns.color_palette('Greys', 15)
    rgb_text = color_list_text[12]

    with plt.style.context('seaborn-white'):
        plt.rcParams['axes.labelpad'] = '40'
        sns.set_theme(style='whitegrid', palette='deep')

        fig, axes = plt.subplots(ceil(len(dimensions) / column_nb), column_nb, figsize=figsize, sharey=sharey)
        fig.tight_layout()
        suptitle_text = 'Boxplot ' + title
        fig.suptitle(suptitle_text, fontname='Corbel', fontsize=70, color=rgb_text)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=top, wspace=wspace, hspace=hspace)

        (l, c) = (0, 0)

        for dimension in dimensions:
            sns.boxplot(data=dataset, x=dimension, y=y_column, ax=axes[l, c],
                        orient='h',
                        showfliers=False,
                        medianprops={"color": "coral", 'linewidth': 4.0},
                        showmeans=True,
                        meanprops={'marker': 'o', 'markeredgecolor': 'black',
                                   'markerfacecolor': 'coral', 'markersize': 20},
                        boxprops={'edgecolor': 'black', 'linewidth': 4.0},
                        capprops={'color': 'black', 'linewidth': 4.0},
                        whiskerprops={'color': 'black', 'linewidth': 4.0})

            axes[l, c].set_title(dimension, fontname='Corbel', color=rgb_text, fontsize=45, pad=50)
            axes[l, c].set_xlabel(None, fontsize=40, fontname='Corbel', color=rgb_text)
            axes[l, c].set_ylabel(y_column, fontsize=40, fontname='Corbel', color=rgb_text)

            axes[l, c].tick_params(axis='both', which='major', labelsize=40, labelcolor=rgb_text)
            axes[l, c].xaxis.offsetText.set_fontsize(40)

            (c, l) = (0, l + 1) if c == column_nb - 1 else (c + 1, l)

    plt.show()


def reshape_data_for_boxplot(dataset: pd.DataFrame, category_name: str, categories: np.array):
    """
    Retourne un dataframe avec une colonne contenant les effectifs de la catégorie selon laquelle agréger les données
    dans les boxplot

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données 
    category_name : str : nom de la catégorie 
    categories : np.array : modalité de la catégorie pour chaque observation
    """
    dataset_boxplot = dataset.copy()
    dataset_boxplot[category_name] = categories

    effectif = dataset_boxplot.groupby([category_name]).count().iloc[:, 0].to_dict()
    dataset_boxplot[category_name + '_n'] = dataset_boxplot.apply(
        lambda row: str(row[category_name]) + '\n(n=' + str(effectif[row[category_name]]) + ')', axis=1)

    return dataset_boxplot.sort_values(category_name)


def reshape_for_nested_donut(dataset: pd.DataFrame, order: dict, big_customers: [str]):
    """
        Renvoie un dataframe prêt à être utilisé pour afficher deux donuts imbriqués

        Positional arguments :
        -------------------------------------
        dataset : pd.DataFrame : jeu de données
        order : dict : ordre dans lequel afficher les catégories de clients dans le donut intérieur
        big_customers : list of strings : liste des catégories de "bons" clients
    """
    new_dataset = dataset.copy()
    small = 'Petits Clients :\n- commandent peu\net\n- dépensent peu'
    big = 'Gros Clients :\n- commandent bcp\net/ou\n- dépensent bcp'

    new_dataset['customer'] = new_dataset.apply(lambda r: big if r['cluster'] in big_customers else small, axis=1)
    new_dataset['cluster_order'] = new_dataset.apply(lambda r: order[r['cluster']], axis=1)

    new_dataset = new_dataset.sort_values(['customer', 'cluster_order'], ascending=[False, True])

    return new_dataset


def display_spider_chart(dataset: pd.DataFrame, categories: str, column_nb: int, figsize: tuple, title: str,
                         palette='Set2',
                         top=0.85, wspace=0.2, hspace=1.8):
    """
        Affiche des radarplots (un graphique par catégorie)

        Positional arguments :
        -------------------------------------
        dataset : pd.DataFrame : jeu de données
        categories : string : nom de la colonne contenant les modalités pour lesquelles créer un radarplot
        column_nb : integer : nombre de graphique à afficher par ligne
        figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
        title : string : titre de la zone de graphique
        
        Optional arguments : 
        -------------------------------------
        palette : string : nom de la palette seaborn à utiliser
        top : float : position de départ des graphiques dans la figure
        wspace : float : largeur de l'espace entre les graphiques
        hspace : float : hauteur de l'espace entre les graphiques
    """
    dataset_mean = dataset.groupby(categories, sort=False).mean().reset_index()
    features = dataset.drop(columns=categories).columns.values
    features_n = len(features)
    categories_n = dataset_mean.shape[0]

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(features_n) * 2 * pi for n in range(features_n)]
    angles += angles[:1]

    rgb_text = sns.color_palette('Greys', 15)[12]
    color = sns.color_palette(palette, categories_n)

    plt.figure(figsize=figsize, dpi=100)
    plt.tight_layout()

    suptitle_text = 'RadarPlot ' + title
    plt.suptitle(suptitle_text, fontname='Corbel', fontsize=25, color=rgb_text)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=top, wspace=wspace, hspace=hspace)

    for categorie in range(0, categories_n):
        ax = plt.subplot(ceil(categories_n / column_nb), column_nb, categorie + 1, polar=True)

        # First axis on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        # Draw one axe per variable + add labels
        plt.xticks(angles[:-1], features, color='grey', size=12)

        # Draw yticks
        # ax.set_rlabel_position(0)
        ax.yaxis.set_ticklabels([])

        values = dataset_mean.loc[categorie].drop(categories).values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color[categorie], linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=color[categorie], alpha=0.4)

        # Add a title
        ax.set_title(dataset_mean[categories][categorie], color=color[categorie], y=1.1, fontsize=22, pad=30,
                     fontname='Corbel')

    plt.show()


def reshape_for_radar(dataset_scaled: np.array, columns: np.array, category_name: str, groups: np.array,
                      groups_rename: dict):
    """
        Renvoie un dataframe prêt à être utilisé pour afficher des radarplots

        Positional arguments :
        -------------------------------------
        dataset_scaled : pd.DataFrame : jeu de données standardisées
        columns : np.array : nom des colonnes du jeu de données 
        category_name : string : nom de la catégorie (on va créer un radarplot pour chacune de ses modalité)
        groups : np.array : modalités prises par chaque individu du jeu de données
        groups_rename : dict : ditionnaire contenant les nouveaux noms des modalités
    """   
    reshaped_dataset = pd.DataFrame(dataset_scaled, columns=columns)
    reshaped_dataset[category_name] = groups
    reshaped_dataset = reshaped_dataset.sort_values(category_name)
    reshaped_dataset[category_name] = reshaped_dataset[category_name].apply(lambda row: groups_rename[row])

    return reshaped_dataset


def display_neighbors_distance_elbow(dataset: pd.DataFrame, n_neighbors: int, figsize: tuple, set_lim=False,
                                     x_lim=None, y_lim=None, palette='Set2'):
    """
        Affiche un graphique représentant la distance moyenne de chaque individu à ses n voisins 

        Positional arguments :
        -------------------------------------
        dataset : pd.DataFrame : jeu de données
        n_neighbors : int : nombre de voisins
        figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
        
        Optional arguments : 
        -------------------------------------
        set_lim : bool : fixer ou non les graduations des axes de coordonnées
        x_lim : float : graduation de l'axe des abscisses
        y_lim : float : graduation de l'axe des ordonnées
        palette : string : nom de la palette seaborn à utiliser
    """
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors_fit = neighbors.fit(dataset)
    distances, indices = neighbors_fit.kneighbors(dataset)

    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    sns.set_theme(style='whitegrid', palette=palette)
    plt.figure(figsize=figsize)
    plt.title('Distance moyenne de chaque observation à ses {} voisins'.format(str(n_neighbors)),
              fontsize=20, fontname='Corbel')
    if set_lim:
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.title('Distance moyenne de chaque observation à ses {} voisins \n -zoom-'.format(str(n_neighbors)),
                  fontsize=20, fontname='Corbel')

    plt.xlabel('Observations', fontsize=16, fontname='Corbel')
    plt.ylabel('Distance moyenne aux {} voisins'.format(str(n_neighbors)), fontsize=16, fontname='Corbel')

    plt.plot(distances)


def display_distribution(dataset: pd.DataFrame, numeric_features: [str], column_n: int, figsize: tuple, top=0.85,
                         wspace=0.2, hspace=1.8):
    """
    Affiche la distribution de chaque variable de la liste.

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données
    numeric_features : list of strings : liste des variables numériques dont on souhaite afficher la distribution
    column_n : int : nombre de graphique à afficher par ligne
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    
    Optional arguments : 
    -------------------------------------
    top : float : position de départ des graphiques dans la figure
    wspace : float : largeur de l'espace entre les graphiques
    hspace : float : hauteur de l'espace entre les graphiques
    """
    rgb_text = sns.color_palette('Greys', 15)[12]
    sns.set_theme(style='whitegrid', palette='Set2')

    fig = plt.figure(figsize=figsize)
    fig.tight_layout()

    fig.suptitle('Distribution variables numériques', fontname='Corbel', fontsize=20, color=rgb_text)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=top, wspace=wspace, hspace=hspace)

    for i, feature in enumerate(numeric_features):
        sub = fig.add_subplot(ceil(len(numeric_features) / column_n), column_n, i + 1)
        sub.set_xlabel(feature, fontsize=14, fontname='Corbel', color=rgb_text)
        sub.set_title(feature, fontsize=16, fontname='Corbel', color=rgb_text)

        sns.histplot(dataset, x=feature)
        sub.grid(False, axis='x')
        sub.tick_params(axis='both', which='major', labelsize=14, labelcolor=rgb_text)

    plt.show()


def plot_dist_comparison(data_before: pd.DataFrame, data_after: pd.DataFrame, dimensions: [str], title: str,
                         figsize: tuple, without_outliers=False,
                         top=0.93, wspace=0.2, hspace=0.5):
    """
    Affiche la distribution de chaque variable choisie avant et après modification

    Positional arguments : 
    -------------------------------------
    data_before : pd.DataFrame : dataframe contenant les données avant modification
    data_after : pd.DataFrame : dataframe contenant les données après modification
    dimensions : list of strings : liste des variables dont on souhaite afficher la distribution
    title : str : titre principal (suptitle)
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    
    Optional arguments : 
    -------------------------------------
    without_outliers : bool : True affiche la distribution sans outliers mais indique le nombre d'outliers
    False affiche la distribution avec les outliers
    """

    color_list_text = sns.color_palette('Greys', 15)
    rgb_text = color_list_text[12]

    data = {0: {'dataset': data_before, 'title': 'avant'}, 1: {'dataset': data_after, 'title': 'après'}}

    with plt.style.context('seaborn-white'):
        plt.rcParams['axes.labelpad'] = '40'
        sns.set_theme(style='whitegrid', palette='Set2')

        fig, axes = plt.subplots(len(dimensions), 2, figsize=figsize, sharey=False)
        fig.tight_layout()
        suptitle_text = 'Distribution ' + title
        fig.suptitle(suptitle_text, fontname='Corbel', fontsize=60, color=rgb_text)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=top, wspace=wspace, hspace=hspace)

        (l, c) = (0, 0)

    for dimension in dimensions:
        for c in range(0, 2):
            dataset = data[c]['dataset']
            if without_outliers:
                subset_without_outliers, outliers = filter_outlier(dataset.loc[~dataset[dimension].isnull()], dimension)
                sns.histplot(data=subset_without_outliers, x=dimension, ax=axes[l, c])
                axes[l, c].text(1, 1, '\n outliers: {:_} \n'.format(len(outliers)), transform=axes[l, c].transAxes,
                                fontsize=50,
                                verticalalignment='top', horizontalalignment='right',
                                bbox={'facecolor': sns.color_palette("husl", 8)[0], 'alpha': 0.3, 'pad': 0,
                                      'boxstyle': 'round'},
                                style='italic', fontname='Open Sans')
            else:
                sns.histplot(data=dataset, x=dimension, ax=axes[l, c])

            axes[l, c].set_title("Distribution \"{}\" {}".format(dimension, data[c]['title']), fontname='Corbel',
                                 color=rgb_text, fontsize=45, pad=50)
            axes[l, c].set_xlabel(dimension, fontsize=40, fontname='Corbel', color=rgb_text)
            # axes[l,c].set_ylabel('Nombre de produits', fontsize=40, fontname='Corbel', color=rgb_text)

            axes[l, c].tick_params(axis='both', which='major', labelsize=40, labelcolor=rgb_text)
            axes[l, c].xaxis.offsetText.set_fontsize(40)

        (c, l) = (0, l + 1)

    plt.show()


def test_transformer(dataset: pd.DataFrame, vars_to_transform: [str], transformers: list, transformer_name: str,
                     figsize: tuple, top=0.9, wspace=0.1, hspace=0.7):
    """
    Affiche la distribution des variables choisies avant et après transformation

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données
    vars_to_transform : list of str : liste des variables dont on souhaite transformer la distribution
    transformers : list : liste des objects de transformation (ex: StandardScaler)
    transformer_name : str : nom de la transformation appliquée à afficher sur le graphique
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    
    Optional arguments : 
    -------------------------------------
    top : float : position de départ des graphiques dans la figure
    wspace : float : largeur de l'espace entre les graphiques
    hspace : float : hauteur de l'espace entre les graphiques
    """
    df_transform = dataset.copy()
    for transformer in transformers:
        df_transform[vars_to_transform] = transformer.fit_transform(df_transform[vars_to_transform].values)

        if transformer_name == 'Yeo-Johnson':
            lbd = transformer.lambdas_

    rgb_text = sns.color_palette('Greys', 15)[12]
    sns.set_theme(style='whitegrid', palette='Set2')

    fig, axes = plt.subplots(len(vars_to_transform), 2, figsize=figsize)
    fig.tight_layout()

    fig.suptitle('Distribution avant et après transformation ({})'.format(transformer_name), fontname='Corbel',
                 fontsize=30, color=rgb_text)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=top, wspace=wspace, hspace=hspace)

    if len(vars_to_transform) > 1:
        for i, feature in enumerate(vars_to_transform):
            sns.histplot(dataset, x=feature, ax=axes[i, 0])
            sns.histplot(df_transform, x=feature, ax=axes[i, 1])

            axes[i, 0].set_title(feature + ' avant transformation (skewness : {:.4f})'.format(dataset[feature].skew()),
                                 fontname='Corbel', fontsize=20, color=rgb_text)

            if transformer_name == 'Yeo-Johnson':
                text_after = feature + ' après transformation (skewness : {:.4f} , lambda : {:.4f})'.format(
                    df_transform[feature].skew(), lbd[i])
            else:
                text_after = feature + ' après transformation (skewness : {:.4f})'.format(df_transform[feature].skew())

            axes[i, 1].set_title(text_after, fontname='Corbel', fontsize=20, color=rgb_text)

            for j in range(2):
                axes[i, j].tick_params(axis='both', which='major', labelsize=16, labelcolor=rgb_text)
                axes[i, j].grid(False, axis='x')

    elif len(vars_to_transform) == 1:

        sns.histplot(dataset, x=vars_to_transform[0], ax=axes[0])
        sns.histplot(df_transform, x=vars_to_transform[0], ax=axes[1])

        axes[0].set_title(vars_to_transform[0] + ' avant transformation (skewness : {:.4f})'.format(
            dataset[vars_to_transform[0]].skew()),
                          fontname='Corbel', fontsize=20, color=rgb_text)
        axes[1].set_title(vars_to_transform[0] + ' après transformation (skewness : {:.4f})'.format(
            df_transform[vars_to_transform[0]].skew()),
                          fontname='Corbel', fontsize=20, color=rgb_text)

        for j in range(2):
            axes[j].tick_params(axis='both', which='major', labelsize=16, labelcolor=rgb_text)
            axes[j].grid(False, axis='x')

    plt.show()


def missing_values_by_column(dataset: pd.DataFrame):
    """
    Retourne un dataframe avec le nombre et le pourcentage de valeurs manquantes par colonnes

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : dataframe contenant les colonnes dont on veut connaitre le pourcentage de vide
    """

    missing_values_series = dataset.isnull().sum()
    missing_values_df = missing_values_series.to_frame(name='Number of Missing Values')
    missing_values_df = missing_values_df.reset_index().rename(columns={'index': 'FEATURES'})

    missing_values_df['Missing Values (%)'] = round(
        missing_values_df['Number of Missing Values'] / (dataset.shape[0]) * 100, 2)

    missing_values_df = missing_values_df.sort_values('Number of Missing Values')

    return missing_values_df


def fill_with_sum(dataset: pd.DataFrame, missing_value_col: str, sum_col1: str, sum_col2: str, verbose=False):
    """
        Remplace les valeurs manquantes par la somme de deux variables

        Positional arguments :
        -------------------------------------
        dataset : pd.DataFrame : jeu de données
        missing_value_col : str : nom de la colonne contenant des valeurs manquantes
        sum_col1 : str : nom de la première colonne à sommer
        sum_col2 : str : nom de la deuxième colonne à commer

        Optional arguments :
        -------------------------------------
        verbose : bool : affiche ou non le nombre de valeurs remplacées
    """
    subset = dataset.copy()
    mask_missing = subset[missing_value_col].isnull()

    missing_value_n = subset.loc[mask_missing].shape[0]

    if missing_value_n > 0:
        subset.loc[mask_missing, missing_value_col] = subset.loc[mask_missing].apply(
            lambda row: row[sum_col1] + row[sum_col2], axis=1)

    if verbose:
        print('{:_} valeur(s) remplacée(s)'.format(
            missing_value_n - subset.loc[subset[missing_value_col].isnull()].shape[0]))

    return subset


def fill_with_average_cost(dataset: pd.DataFrame, missing_value_col: str, cost_col: str, verbose=False):
    """
        Remplace les valeurs manquantes en utilisant le coût moyen

        Positional arguments :
        -------------------------------------
        dataset : pd.DataFrame : jeu de données
        missing_value_col : str : nom de la colonne contenant des valeurs manquantes
        cost_col : str : nom de la variable contenant le coût total

        Optional arguments :
        -------------------------------------
        verbose : bool : affiche ou non le nombre de valeurs remplacées
    """
    new_dataset = dataset.copy()
    mask_missing = (new_dataset[missing_value_col].isnull())
    missing_value_n = new_dataset.loc[mask_missing].shape[0]

    if missing_value_n > 0:
        average_cost = new_dataset.loc[~mask_missing, cost_col].sum() / new_dataset.loc[
            ~mask_missing, missing_value_col].sum()
        new_dataset.loc[mask_missing, missing_value_col] = new_dataset.loc[mask_missing].apply(
            lambda row: row[cost_col] / average_cost, axis=1)

    if verbose:
        print('{:_} valeur(s) remplacée(s)'.format(
            missing_value_n - new_dataset.loc[new_dataset[missing_value_col].isnull()].shape[0]))

    return new_dataset


def fill_with_mean(dataset: pd.DataFrame, missing_value_col: str, verbose=False):
    """
        Remplace les valeurs manquantes par la moyenne

        Positional arguments :
        -------------------------------------
        dataset : pd.DataFrame : jeu de données
        missing_value_col : str : nom de la colonne contenant des valeurs manquantes

        Optional arguments :
        -------------------------------------
        verbose : bool : affiche ou non le nombre de valeurs remplacées
    """
    subset = dataset.copy()
    mean = subset[missing_value_col].mean()
    mask_missing = subset[missing_value_col].isnull()

    missing_value_n = subset.loc[mask_missing].shape[0]

    if missing_value_n > 0:
        subset.loc[mask_missing, missing_value_col] = round(mean,
                                                            0)

    if verbose:
        print('{:_} valeur(s) remplacée(s)'.format(
            missing_value_n - subset.loc[subset[missing_value_col].isnull()].shape[0]))

    return subset


def get_mode(row: pd.Series):
    """
        Renvoie le mode de la colonne

        Positional arguments :
        -------------------------------------
        row : pd.Series : colonne dont on souhaite connaitre le mode
    """
    mode = pd.Series.mode(row)
    if len(mode) == 0:
        mode = np.nan
    elif mode.dtype == np.ndarray:
        mode = mode[0]
    return mode


def drop_date(dataset: pd.DataFrame, date_col: str, before_date: str, after_date: str):
    """
        Filtre un jeu de données en fonction d'une colonne date

        Positional arguments :
        -------------------------------------
        dataset : pd.DataFrame : jeu de données
        date_col : str : nom de la colonne contenant les dates sur lesquelles filtrer
        before_date : str : date de début (sera incluse dans le jeu de données filtré)
        after_date : str : date de din (sera incluse dans le jeu de données filtré)
    """
    mask_before = dataset[date_col] >= pd.to_datetime(before_date + ' 00:00:00', format='%Y-%m-%d %H:%M:%S')
    mask_after = dataset[date_col] <= pd.to_datetime(after_date + ' 00:00:00', format='%Y-%m-%d %H:%M:%S')

    subset = dataset.loc[mask_before & mask_after]

    return subset


def columns_to_datetime(dataset: pd.DataFrame, columns: [str], format_str='%Y-%m-%d %H:%M:%S', errors='coerce'):
    """
        Convertit une ou plusieurs colonnes d'un dataframe en type datetime

        Positional arguments :
        -------------------------------------
        dataset : pd.DataFrame : jeu de données
        columns : list of str : liste des noms des colonnes à convertir

        Optional arguments :
        -------------------------------------
        format_str : str : format du datetime
        errors : str : gestion des erreurs
    """
    new_dataset = dataset.copy()
    for feature in columns:
        new_dataset[feature] = pd.to_datetime(new_dataset[feature], format=format_str, errors=errors)

    return new_dataset


def read_csv_files(files_path: dict, to_datetime=False, col_to_datime=None):
    """
        Renvoie un dictionnaire contenant des dataframes construits à partir de fichiers csv.

        Positional arguments :
        -------------------------------------
        files_path : dict : dictionnaire contenant les emplacements des fichiers csv

        Optional arguments :
        -------------------------------------
        to_datetime : bool : convertir ou non certaines colonnes des dataframes en datetime
        col_to_datetime : list of strings : liste des colonnes à convertir en datetime
    """
    datasets = dict()
    for key, value in files_path.items():
        dataset = pd.read_csv(value)
        datasets[key] = dataset
        if to_datetime and key in col_to_datime.keys():
            datasets[key] = columns_to_datetime(dataset, col_to_datime[key])
    return datasets


def build_master_dataset(datasets: dict, verbose=False):
    """
        Renvoie un jeu de données avec une ligne par commande obtenu par agrégation de plusieurs jeux de données

        Positional arguments :
        -------------------------------------
        dataset : dict : dictionnaire contenant les jeux de données à agréger

        Optional arguments :
        -------------------------------------
        verbose : bool : afficher ou non le résultat du traitement des valeurs manquantes
    """
    master_dataset = pd.merge(datasets['order_data'], datasets['customer_data'], on='customer_id', how='left')

    payments_data_order = datasets['payments_data'].groupby('order_id')['payment_value'].sum()
    master_dataset = master_dataset.merge(payments_data_order.reset_index(), on='order_id', how='left')

    items_data_order = datasets['items_data'].groupby('order_id').agg({'order_item_id': 'count',
                                                                       'price': 'sum',
                                                                       'freight_value': 'sum'})
    master_dataset = master_dataset.merge(items_data_order.reset_index(), on='order_id', how='left')
    master_dataset.rename(columns={"order_item_id": "products_n"}, inplace=True)

    most_recent_review_df = clean_duplicates(datasets['reviews_data'], 'order_id', 'review_creation_date', False)
    master_dataset = master_dataset.merge(most_recent_review_df[['review_score', 'order_id']], on='order_id',
                                          how='left')

    # Gestion des valeurs manquantes
    master_dataset = fill_with_sum(master_dataset, 'payment_value', 'price', 'freight_value', verbose=verbose)
    master_dataset = fill_with_average_cost(master_dataset, 'products_n', 'payment_value', verbose=verbose)
    master_dataset = fill_with_mean(master_dataset, 'review_score', verbose=verbose)

    return master_dataset


def build_behavioral_dataset(master_dataset: pd.DataFrame):
    """
        Renvoie un jeu de données avec une ligne par client obtenu par agrégation et feature engineering

        Positional arguments :
        -------------------------------------
        master_dataset : pd.DataFrame : jeu de données avec une ligne par commande
    """
    behavioral_data = master_dataset.groupby('customer_unique_id').agg({'payment_value': 'sum',
                                                                        'order_id': 'count',
                                                                        'order_purchase_timestamp': 'max',
                                                                        'products_n': 'mean',
                                                                        'review_score': 'mean',
                                                                        }).reset_index()

    behavioral_data.rename(columns={'payment_value': 'payment_value_total',
                                    'order_id': 'order_n',
                                    'products_n': 'products_mean',
                                    'order_purchase_timestamp': 'last_order_timestamp',
                                    'review_score': 'review_score_mean'
                                    }, inplace=True)

    behavioral_data['days_since_last_purchase'] = (
            behavioral_data['last_order_timestamp'].max() - behavioral_data['last_order_timestamp']).dt.days
    behavioral_data.drop(columns=['last_order_timestamp', 'products_mean'], inplace=True)

    return behavioral_data


def filter_order_by(master_dataset: pd.DataFrame, filter_by_date=False, from_date=None, to_date=None):
    """
        Renvoie un jeu de données avec une ligne par client ayant commandé au court de la période de temps renseignée

        Positional arguments :
        -------------------------------------
        master_dataset : pd.DataFrame : jeu de données avec une ligne par commande

        Optional arguments :
        -------------------------------------
        filter_by_date : bool : filtrer ou non le jeu de données sur la date de commande
        from_date : str : date de début (sera incluse dans le jeu de données filtré)
        to_date : str : date de fin (sera incluse dans le jeu de données filtré)
    """
    if filter_by_date:
        master_dataset = drop_date(master_dataset,
                                   'order_purchase_timestamp',
                                   from_date,
                                   to_date
                                   )

    behavioral_data = build_behavioral_dataset(master_dataset)
    behavioral_data = behavioral_data.set_index('customer_unique_id')

    return behavioral_data


def fit_kmeans(x: pd.DataFrame):
    """
        Renvoie un pipeline entrainé contenant un StandardScaler et un modèle KMeans

        Positional arguments :
        -------------------------------------
        x: pd.DataFrame : jeu de données d'entrainement

    """
    pipeline = make_pipeline(StandardScaler(),
                             KMeans(n_clusters=5, n_init=10, random_state=8)
                             )
    pipeline.fit(x)

    return pipeline


def compute_ari_score(master_dataset: pd.DataFrame, start: str, end: str, first_period_end: str, freq='1MS'):
    """
        Renvoie une liste d'indices de rand ajustés obtenus après simulation pour établir la fréquence de mise à jour

        Positional arguments :
        -------------------------------------
        master_dataset : pd.DataFrame : jeu de données avec une ligne par commande
        start : str : date de début de la simulation
        end : str : date de fin de la simulation
        first_period_end : str : date de fin de la période d'initialisation

        Optional arguments :
        -------------------------------------
        freq : str : fréquence d'itération
    """
    ari_score = []
    date_range_end = pd.date_range(start=first_period_end,
                                   end=end,
                                   freq=freq,
                                   inclusive='right'
                                   ).strftime(date_format='%Y-%m-%d')

    customers_0 = filter_order_by(master_dataset,
                                  True,
                                  start,
                                  first_period_end
                                  )
    model_0 = fit_kmeans(customers_0)

    for i_period_end in date_range_end:
        customers_i = filter_order_by(master_dataset, True,
                                      start, i_period_end
                                      )
        model_i = fit_kmeans(customers_i)

        ari = adjusted_rand_score(model_0.predict(customers_i),
                                  model_i[1].labels_
                                  )
        ari_score.append(ari)

    return ari_score


def display_ari_by_frequency(ari_score: [float], thresh: float, frequency: str, fig_size=(15, 7)):
    """
        Affiche un graphique avec en abscisse l'itération en semaine et en ordonnée le ARI

        Positional arguments :
        -------------------------------------
        ari_score : list of floats : liste des indices de rand ajustés
        thresh : float : limite en dessous de laquelle réentrainer le modèle
        frequency : str : fréquence d'itération à afficher (ex: semaine)

        Optional arguments :
        -------------------------------------
        fig_size : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    """
    sns.set_theme(style='whitegrid', palette='deep')
    plt.figure(figsize=fig_size)
    plt.plot(np.arange(1, len(ari_score) + 1), ari_score)
    update_freq = next(freq
                       for freq, ari in enumerate(ari_score)
                       if ari < thresh) + 1

    plt.axvline(x=update_freq, linewidth=2, color='coral', alpha=0.7, ls='--')
    plt.axhline(y=ari_score[update_freq - 1], linewidth=2, color='coral',
                alpha=0.7, ls='--'
                )

    plt.title('Evolution de l\'indice de rand ajusté par ' +
              frequency, fontname='Corbel', fontsize=20, pad=10)
    plt.xlabel(frequency, fontname='Corbel', fontsize=16)
    plt.ylabel('Indice de rand ajusté', fontname='Corbel', fontsize=16)

    plt.text(len(ari_score), 0.95, '\n Mettre à jour au bout de {} {}(s) \n'.format(update_freq, frequency),
             fontsize=15, style='normal', fontname='Open Sans',
             verticalalignment='top', horizontalalignment='right', color='red',
             bbox={'pad': 0, 'boxstyle': 'round',
                   'facecolor': 'white', 'edgecolor': 'black'
                   }
             )

    plt.show()

    
def make_preprocessor(transformers: [dict]):
    """
    Retourne un objet preprocessor contenant les transformations à appliquer aux données avant l'entrainement du modèle

    Positional arguments : 
    -------------------------------------
    transformers : list of tuples : liste des tuples (modifications à appliquer 
    categorical_features : list of strings : liste des variables catégorielles à transformer (one hot encoder)
    """
    steps = []
    for transformer in transformers:
        pipeline = make_pipeline(*transformer['estimator'])
        steps.append((pipeline, transformer['feature']))

    preprocessor = make_column_transformer(*steps)

    return preprocessor
