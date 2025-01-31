"""
verification des resultats des k obtenues pour le modele global

"""

import casadi as cas
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
from ezc3d import c3d
from mpl_toolkits import mplot3d
import time
from scipy.interpolate import interp1d
import pickle


n = 15  # nombre de mailles sur le grand cote
m = 9  # nombre de mailles sur le petit cote

Nb_ressorts = 2 * n * m + n + m  # nombre de ressorts non obliques total dans le modele
Nb_ressorts_cadre = 2 * n + 2 * m  # nombre de ressorts entre le cadre et la toile
Nb_ressorts_croix = 2 * (m - 1) * (n - 1)  # nombre de ressorts obliques dans la toile
Nb_ressorts_horz = n * (m - 1)  # nombre de ressorts horizontaux dans la toile (pas dans le cadre)
Nb_ressorts_vert = m * (n - 1)  # nombre de ressorts verticaux dans la toile (pas dans le cadre)

masse_type='repartie' #'ponctuelle' #'repartie'

#FONCTIONS AVEC LES PARAMÈTRES FIXES :
def Param_fixe(ind_masse,Masse_centre):
    """

    :param ind_masse: int: indice du point où la masse est appliquée dans la collecte (point qui descend le plus bas)
    :param Masse_centre: float: masse appliquée au point d'indice ind_masse
    :return: dict_fixed_params: dictionnaire contenant tous les paramètres fixes (longueurs, l_repos, masses)
    """
#ESPACES ENTRE LES MARQUEURS :

    # de bas en haut :
    dL = np.array([510.71703748, 261.87522103, 293.42186099, 298.42486747, 300.67352585,
                   298.88879749, 299.6946861, 300.4158083, 304.52115312, 297.72780618,
                   300.53723415, 298.27144226, 298.42486747, 293.42186099, 261.87522103,
                   510.71703748]) * 0.001  # ecart entre les lignes, de bas en haut
    dL = np.array([np.mean([dL[i], dL[-(i + 1)]]) for i in range(16)])

    dLmilieu = np.array([151.21983556, 153.50844775]) * 0.001
    dLmilieu = np.array([np.mean([dLmilieu[i], dLmilieu[-(i + 1)]]) for i in range(2)])
    # de droite a gauche :
    dl = np.array(
        [494.38703513, 208.96708367, 265.1669672, 254.56358938, 267.84760997, 268.60351809, 253.26974254,
         267.02823864,
         208.07894712, 501.49013437]) * 0.001
    dl = np.array([np.mean([dl[i], dl[-(i + 1)]]) for i in range(10)])

    dlmilieu = np.array([126.53897435, 127.45173517]) * 0.001
    dlmilieu = np.array([np.mean([dlmilieu[i], dlmilieu[-(i + 1)]]) for i in range(2)])

    l_droite = np.sum(dl[:5])
    l_gauche = np.sum(dl[5:])

    L_haut = np.sum(dL[:8])
    L_bas = np.sum(dL[8:])

####################################################################################################################

    # LONGUEURS AU REPOS
    l_repos = np.zeros(
        Nb_ressorts_cadre)  # on fera des append plus tard, l_repos prend bien en compte tous les ressorts non obliques

    # entre la toile et le cadre :
    # ecart entre les marqueurs - taille du ressort en pretension + taille ressort hors trampo
    l_bord_horz = dl[0] - 0.388 + 0.264
    l_bord_vert = dL[0] - 0.388 + 0.264
    l_repos[0:n], l_repos[n + m:2 * n + m] = l_bord_horz, l_bord_horz
    l_repos[n:n + m], l_repos[2 * n + m:2 * n + 2 * m] = l_bord_vert, l_bord_vert

    l_bord_coin = np.mean([l_bord_vert, l_bord_horz])  # pas sure !!!
    l_repos[0], l_repos[n - 1], l_repos[n + m], l_repos[
        2 * n + m - 1] = l_bord_coin, l_bord_coin, l_bord_coin, l_bord_coin
    l_repos[n], l_repos[n + m - 1], l_repos[2 * n + m], l_repos[
        2 * (n + m) - 1] = l_bord_coin, l_bord_coin, l_bord_coin, l_bord_coin

    # dans la toile : on dit que les longueurs au repos sont les memes que en pretension
    # ressorts horizontaux internes a la toile :
    l_horz = np.array([dl[j] * np.ones(n) for j in range(1, m)])
    l_horz = np.reshape(l_horz, Nb_ressorts_horz)
    l_repos = np.append(l_repos, l_horz)

    # ressorts verticaux internes a la toile :
    l_vert = np.array([dL[j] * np.ones(m) for j in range(1, n)])
    l_vert = np.reshape(l_vert, Nb_ressorts_vert)
    l_repos = np.append(l_repos, l_vert)

    # # ressorts obliques internes a la toile :
    l_repos_croix = []
    for j in range(m - 1):  # on fait colonne par colonne
        l_repos_croixj = np.zeros(n - 1)
        l_repos_croixj[0:n - 1] = (l_vert[0:m * (n - 1):m] ** 2 + l_horz[j * n] ** 2) ** 0.5
        l_repos_croix = np.append(l_repos_croix, l_repos_croixj)
    # dans chaque maille il y a deux ressorts obliques :
    l_repos_croix_double = np.zeros((int(Nb_ressorts_croix / 2), 2))
    for i in range(int(Nb_ressorts_croix / 2)):
        l_repos_croix_double[i] = [l_repos_croix[i], l_repos_croix[i]]
    l_repos_croix = np.reshape(l_repos_croix_double, Nb_ressorts_croix)

    dict_fixed_params = {'dL': dL,
                         'dLmilieu': dLmilieu,
                         'dl': dl,
                         'dlmilieu': dlmilieu,
                         'l_droite': l_droite,
                         'l_gauche': l_gauche,
                         'L_haut': L_haut,
                         'L_bas': L_bas,
                         'l_repos': l_repos,
                         'l_repos_croix': l_repos_croix}

    return dict_fixed_params  # np

def Param_variable(k_type, ind_masse, Masse_repartie):
    """
    Répartir les raideurs des ressorts à partir des différents types de ressorts
    :param k_type: cas.MX(12): les 12 types de ressorts qu'on retrouve dans la toile (8 structurels, 4 cisaillement)
    :return: k: cas.MX(Nb_ressorts): ensemble des raideurs des ressorts non obliques, dont ressorts du cadre
    :return: k_oblique: cas.MX(Nb_ressorts_croix): ensemble des raideurs des ressorts obliques
    """
    # RAIDEURS A CHANGER
    k1 = k_type[0] #un type de coin (ressort horizontal)
    k2 = k_type[1] #ressorts horizontaux du bord (bord vertical)
    k3 = k_type[2] #un type de coin (ressort vertical)
    k4 = k_type[3] #ressorts verticaux du bord (bord horizontal)
    k5 = k_type[4] #ressorts horizontaux du bord horizontal de la toile
    k6 = k_type[5] #ressorts horizontaux
    k7 = k_type[6] #ressorts verticaux du bord vertical de la toile
    k8 = k_type[7] #ressorts verticaux
    k_oblique_1 = k_type[8] #4 ressorts des coins
    k_oblique_2 = k_type[9] #ressorts des bords verticaux
    k_oblique_3 = k_type[10] #ressorts des bords horizontaux
    k_oblique_4 = k_type[11] #ressorts quelconques

    # ressorts entre le cadre du trampoline et la toile : k1,k2,k3,k4
    k_bord = np.zeros(Nb_ressorts_cadre)
    # cotes verticaux de la toile :
    k_bord[0:n], k_bord[n + m:2 * n + m] = k2, k2
    # cotes horizontaux :
    k_bord[n:n + m], k_bord[2 * n + m:2 * n + 2 * m] = k4, k4
    # coins :
    k_bord[0], k_bord[n - 1], k_bord[n + m], k_bord[2 * n + m - 1] = k1, k1, k1, k1
    k_bord[n], k_bord[n + m - 1], k_bord[2 * n + m], k_bord[2 * (n + m) - 1] = k3, k3, k3, k3

    # ressorts horizontaux dans la toile
    k_horizontaux = k6 * np.ones(n * (m - 1))
    k_horizontaux[0:n*(m-1):n] = k5  # ressorts horizontaux du bord DE LA TOILE en bas
    k_horizontaux[n - 1:n * (m - 1):n] = k5  # ressorts horizontaux du bord DE LA TOILE en haut

    # ressorts verticaux dans la toile
    k_verticaux = k8 * np.ones(m * (n - 1))
    k_verticaux[0:m * (n - 1):m] = k7  # ressorts verticaux du bord DE LA TOILE a droite
    k_verticaux[m - 1:n * m - m:m] = k7  # ressorts verticaux du bord DE LA TOILE a gauche

    k = np.hstack((k_horizontaux, k_verticaux))
    k = np.hstack((k_bord, k))

######################################################################################################################

    # RESSORTS OBLIQUES
    #milieux :
    k_oblique = np.zeros(Nb_ressorts_croix)

    #coins :
    k_oblique[0], k_oblique[1] = k_oblique_1,k_oblique_1  # en bas a droite
    k_oblique[2*(n-1) - 1], k_oblique[2*(n-1) - 2] = k_oblique_1, k_oblique_1 #en haut a droite
    k_oblique[Nb_ressorts_croix - 1], k_oblique[Nb_ressorts_croix - 2] = k_oblique_1, k_oblique_1  # en haut a gauche
    k_oblique[2*(n-1)*(m-2)], k_oblique[2*(n-1)*(m-2) + 1] = k_oblique_1, k_oblique_1  # en bas a gauche

    #côtés verticaux :
    k_oblique[2 : 2*(n-1) - 2] = k_oblique_2 #côté droit
    k_oblique[2*(n-1)*(m-2) + 2 : Nb_ressorts_croix - 2] = k_oblique_2 #côté gauche

    # côtés horizontaux :
    k_oblique[28:169:28], k_oblique[29:170:28] = k_oblique_3, k_oblique_3 #en bas
    k_oblique[55:196:28], k_oblique[54:195:28] = k_oblique_3, k_oblique_3 #en haut

    #milieu :
    for j in range (1,7) :
        k_oblique[2 + 2*j*(n-1) : 26 + 2*j*(n-1)] = k_oblique_4

######################################################################################################################
    Mtrampo = 5.00
    mressort_bord = 0.322
    mressort_coin = 0.553
    mattache = 0.025  # attache metallique entre trampo et ressort

    mmilieu = Mtrampo / ((n - 1) * (m - 1))  # masse d'un point se trouvant au milieu de la toile
    mpetitbord = 0.5 * (Mtrampo / ((n - 1) * (m - 1))) + 2 * (
            (mressort_bord / 2) + mattache)  # masse d un point situé sur le petit bord
    mgrandbord = 0.5 * (Mtrampo / ((n - 1) * (m - 1))) + (33 / 13) * (
            (mressort_bord / 2) + mattache)  # masse d un point situé sur le grand bord
    mcoin = 0.25 * (Mtrampo / ((n - 1) * (m - 1))) + mressort_coin + 4 * (
            (mressort_bord / 2) + mattache)  # masse d un point situé dans un coin

    M = mmilieu * np.ones(n * m)  # on initialise toutes les masses a celle du centre
    M[0], M[n - 1], M[n * (m - 1)], M[n * m - 1] = mcoin, mcoin, mcoin, mcoin
    M[n:n * (m - 1):n] = mpetitbord  # masses du cote bas
    M[2 * n - 1:n * m - 1:n] = mpetitbord  # masses du cote haut
    M[1:n - 1] = mgrandbord  # masse du cote droit
    M[n * (m - 1) + 1:n * m - 1] = mgrandbord  # masse du cote gauche

    #masse du disque sur les points concernés M = [4.71, 11.09, 7.04, 10.11, 14.12]
    M[ind_masse] += Masse_repartie[0]
    M[ind_masse + 1] += Masse_repartie[1]
    M[ind_masse - 1] += Masse_repartie[2]
    M[ind_masse + 15] += Masse_repartie[3]
    M[ind_masse - 15] += Masse_repartie[4]


    return k,k_oblique, M  #  sx

def rotation_points (Pos_repos,Pt_ancrage) :
    """
    Appliquer la rotation pour avoir la même orientation que les points de la collecte
    :param Pos_repos: cas.DM(n*m,3): coordonnées (2D) des points de la toile
    :param Pt_ancrage: cas.DM(2*n+2*m,3): coordonnées des points du cadre
    :return: Pos_repos, Pt_ancrage
    """

    mat_base_collecte = np.array([[ 0.99964304, -0.02650231,  0.00338079],
               [ 0.02650787,  0.99964731, -0.00160831],
               [-0.00333697,  0.00169736,  0.99999299]])
    #calcul inverse
    # mat_base_inv_np = np.linalg.inv(mat_base_collecte)
    mat_base_inv_np = mat_base_collecte

    Pt_ancrage_new = np.zeros((Nb_ressorts_cadre,3))
    for index in range (Nb_ressorts_cadre) :
        Pt_ancrage_new[index,:] = np.matmul(Pt_ancrage[index,:], mat_base_inv_np) #multplication de matrices en casadi

    Pos_repos_new = np.zeros((n*m, 3))
    for index in range(n*m):
        Pos_repos_new[index, :] = np.matmul(Pos_repos[index, :], mat_base_inv_np)  # multplication de matrices en casadi

    return Pt_ancrage_new, Pos_repos_new

def Points_ancrage_repos(dict_fixed_params):
    """
    :param dict_fixed_params: dictionnaire contenant les paramètres fixés
    :return: Pos_repos: cas.DM(n*m,3): coordonnées (2D) des points de la toile
    :return: Pt_ancrage: cas.DM(2*n+2*m,3): coordonnées des points du cadre
    """
    dL = dict_fixed_params['dL']
    dl = dict_fixed_params['dl']
    l_droite = dict_fixed_params['l_droite']
    l_gauche = dict_fixed_params['l_gauche']
    L_haut = dict_fixed_params['L_haut']
    L_bas = dict_fixed_params['L_bas']

    # repos :
    Pos_repos = np.zeros((n * m, 3))

    # on dit que le point numero 0 est a l'origine
    for j in range(m):
        for i in range(n):
            # Pos_repos[i + j * n] = np.array([-np.sum(dl[:j + 1]), np.sum(dL[:i + 1]), 0])
            Pos_repos[i + j * n,:] = np.array([-np.sum(dl[:j + 1]), np.sum(dL[:i + 1]), 0])

    Pos_repos_new = np.zeros((n * m, 3))
    for j in range(m):
        for i in range(n):
            Pos_repos_new[i + j * n,:] = Pos_repos[i + j * n,:] - Pos_repos[67,:]
    # Pos_repos_new = np.copy(Pos_repos)

    # ancrage :
    Pt_ancrage = np.zeros((2 * (n + m), 3))
    # cote droit :
    for i in range(n):
        Pt_ancrage[i, 1:2] = Pos_repos_new[i, 1:2]
        Pt_ancrage[i, 0] = l_droite
    # cote haut : on fait un truc complique pour center autour de l'axe vertical
    Pt_ancrage[n + 4, :] = np.array([0,L_haut, 0])
    for j in range(n, n + 4):
        Pt_ancrage[j, :] = np.array([0,L_haut, 0]) + np.array([np.sum(dl[1 + j - n:5]), 0, 0])
    for j in range(n + 5, n + m):
        Pt_ancrage[j, :] = np.array([0,L_haut, 0]) - np.array([np.sum(dl[5: j - n + 1]), 0, 0])
    # cote gauche :
    for k in range(n + m, 2 * n + m):
        Pt_ancrage[k, 1:2] = - Pos_repos_new[k - n - m, 1:2]
        Pt_ancrage[k, 0] = -l_gauche
    # cote bas :
    Pt_ancrage[2 * n + m + 4, :] = np.array([0, -L_bas, 0])

    Pt_ancrage[2 * n + m, :] = np.array([0, -L_bas, 0]) - np.array([np.sum(dl[5:9]), 0, 0])
    Pt_ancrage[2 * n + m + 1, :] = np.array([0, -L_bas, 0]) - np.array([np.sum(dl[5:8]), 0, 0])
    Pt_ancrage[2 * n + m + 2, :] = np.array([0, -L_bas, 0]) - np.array([np.sum(dl[5:7]), 0, 0])
    Pt_ancrage[2 * n + m + 3, :] = np.array([0, -L_bas, 0]) - np.array([np.sum(dl[5:6]), 0, 0])

    Pt_ancrage[2 * n + m + 5, :] = np.array([0, -L_bas, 0]) + np.array([np.sum(dl[4:5]), 0, 0])
    Pt_ancrage[2 * n + m + 6, :] = np.array([0, -L_bas, 0]) + np.array([np.sum(dl[3:5]), 0, 0])
    Pt_ancrage[2 * n + m + 7, :] = np.array([0, -L_bas, 0]) + np.array([np.sum(dl[2:5]), 0, 0])
    Pt_ancrage[2 * n + m + 8, :] = np.array([0, -L_bas, 0]) + np.array([np.sum(dl[1:5]), 0, 0])

    Pt_ancrage,Pos_repos_new = rotation_points(Pos_repos_new,Pt_ancrage)

    return Pt_ancrage, Pos_repos #np

def Spring_bouts(Pt, Pt_ancrage): #sx
    """

    :param Pt: cas.MX(n*m,3): coordonnées des n*m points de la toile
    :param Pt_ancrage: cas.DM(2*n+2*m,3): coordonnées des points du cadre

    :return: Spring_bout_1: cas.MX((Nb_ressorts, 3)): bout 1 de chaque ressort non oblique dont ressorts du cadre
    :return: Spring_bout_2: cas.MX((Nb_ressorts, 3)): bout 2 de chaque ressort non oblique dont ressorts du cadre
    """

    # Definition des ressorts (position, taille)
    Spring_bout_1 = np.zeros((Nb_ressorts, 3))

    # RESSORTS ENTRE LE CADRE ET LA TOILE
    for i in range(0, Nb_ressorts_cadre):
        Spring_bout_1[i, :] = Pt_ancrage[i, :]

    # RESSORTS HORIZONTAUX : il y en a n*(m-1)
    for i in range(Nb_ressorts_horz):
        Spring_bout_1[Nb_ressorts_cadre + i, :] = Pt[i, :]

    # RESSORTS VERTICAUX : il y en a m*(n-1)
    k = 0
    for i in range(n - 1):
        for j in range(m):
            Spring_bout_1[Nb_ressorts_cadre + Nb_ressorts_horz + k, :] = Pt[i + n * j, :]
            k += 1
    ####################################################################################################################
    Spring_bout_2 = np.zeros((Nb_ressorts, 3))

    # RESSORTS ENTRE LE CADRE ET LA TOILE
    for i in range(0, n):  # points droite du bord de la toile
        Spring_bout_2[i, :] = Pt[i, :]

    k = 0
    for i in range(n - 1, m * n, n):  # points hauts du bord de la toile
        Spring_bout_2[n + k, :] = Pt[i, :]
        k += 1

    k = 0
    for i in range(m * n - 1, n * (m - 1) - 1, -1):  # points gauche du bord de la toile
        Spring_bout_2[n + m + k, :] = Pt[i, :]
        k += 1

    k = 0
    for i in range(n * (m - 1), -1, -n):  # points bas du bord de la toile
        Spring_bout_2[2 * n + m + k, :] = Pt[i, :]
        k += 1

    # RESSORTS HORIZONTAUX : il y en a n*(m-1)
    k = 0
    for i in range(n, n * m):
        Spring_bout_2[Nb_ressorts_cadre + k, :] = Pt[i, :]
        k += 1

    # RESSORTS VERTICAUX : il y en a m*(n-1)
    k = 0
    for i in range(1, n):
        for j in range(m):
            Spring_bout_2[Nb_ressorts_cadre + Nb_ressorts_horz + k, :] = Pt[i + n * j, :]
            k += 1

    return (Spring_bout_1, Spring_bout_2)

def Spring_bouts_croix(Pt): #sx
    """

    :param Pt: cas.MX(n*m,3): coordonnées des n*m points de la toile

    :return: Spring_bout_croix_1: cas.MX((Nb_ressorts_croix, 3)): bout 1 de chaque ressort oblique
    :return: Spring_bout_croix_2: cas.MX((Nb_ressorts_croix, 3)): bout 2 de chaque ressort oblique
    """
    # RESSORTS OBLIQUES : il n'y en a pas entre le cadre et la toile
    Spring_bout_croix_1 = np.zeros((Nb_ressorts_croix, 3))

    # Pour spring_bout_1 on prend uniquement les points de droite des ressorts obliques
    k = 0
    for i in range((m - 1) * n):
        Spring_bout_croix_1[k, :] = Pt[i, :]
        k += 1
        # a part le premier et le dernier de chaque colonne, chaque point est relie a deux ressorts obliques
        if (i + 1) % n != 0 and i % n != 0:
            Spring_bout_croix_1[k, :] = Pt[i, :]
            k += 1

    Spring_bout_croix_2 = np.zeros((Nb_ressorts_croix, 3))
    # Pour spring_bout_2 on prend uniquement les points de gauche des ressorts obliques
    # pour chaue carre on commence par le point en haut a gauche, puis en bas a gauche
    # cetait un peu complique mais ca marche, faut pas le changer
    j = 1
    k = 0
    while j < m:
        for i in range(j * n, (j + 1) * n - 2, 2):
            Spring_bout_croix_2[k, :] = Pt[i + 1, :]
            Spring_bout_croix_2[k + 1, :] = Pt[i, :]
            Spring_bout_croix_2[k + 2, :] = Pt[i + 2, :]
            Spring_bout_croix_2[k + 3, :] = Pt[i + 1, :]
            k += 4
        j += 1

    return Spring_bout_croix_1, Spring_bout_croix_2

def Force_calc(Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2, k,k_oblique, M, dict_fixed_params):  # force dans chaque ressort
    """

    :param Spring_bout_1: cas.MX((Nb_ressorts, 3)): bout 1 de chaque ressort non oblique
    :param Spring_bout_2: cas.MX((Nb_ressorts, 3)): bout 2 de chaque ressort non oblique
    :param Spring_bout_croix_1: cas.MX((Nb_ressorts_croix, 3)): bout 1 de chaque ressort oblique
    :param Spring_bout_croix_2: cas.MX((Nb_ressorts_croix, 3)): bout 2 de chaque ressort oblique
    :param k: cas.MX(Nb_ressorts): raideurs de tous les ressorts non obliques
    :param k_oblique: cas.MX(Nb_ressorts_croix): raideurs de tous les ressorts obliques
    :param dict_fixed_params: dictionnaire contenant les paramètres fixes

    :return: F_spring: cas.MX(Nb_ressorts, 3): force élastique de chaque ressort non oblique (dont ressorts du cadre)
    :return: F_spring_croix: cas.MX(Nb_ressorts_croix, 3): force élastique de chaque ressort oblique
    :return: F_masses: cas.MX(n*m,3): force de gravité appliquée à chaque point
    """

    l_repos = dict_fixed_params['l_repos']
    l_repos_croix = dict_fixed_params['l_repos_croix']

    F_spring = np.zeros((Nb_ressorts, 3))
    Vect_unit_dir_F = np.zeros((Nb_ressorts, 3))
    for i in range(Nb_ressorts):
        Vect_unit_dir_F[i, :] = (Spring_bout_2[i, :] - Spring_bout_1[i, :]) / np.linalg.norm(Spring_bout_2[i, :] - Spring_bout_1[i, :])

    for ispring in range(Nb_ressorts):
        F_spring[ispring, :] = Vect_unit_dir_F[ispring, :] * k[ispring] * (
                np.linalg.norm(Spring_bout_2[ispring, :] - Spring_bout_1[ispring, :]) - l_repos[ispring])

    # F_spring_croix = np.zeros((Nb_ressorts_croix, 3))
    F_spring_croix = np.zeros((Nb_ressorts_croix, 3))
    Vect_unit_dir_F_croix = np.zeros((Nb_ressorts, 3))
    for i in range(Nb_ressorts_croix):
        Vect_unit_dir_F_croix[i, :] = (Spring_bout_croix_2[i, :] - Spring_bout_croix_1[i, :]) / np.linalg.norm(Spring_bout_croix_2[i, :] - Spring_bout_croix_1[i, :])

    for ispring in range(Nb_ressorts_croix):
        F_spring_croix[ispring, :] = Vect_unit_dir_F_croix[ispring, :] * k_oblique[ispring] * (
                np.linalg.norm(Spring_bout_croix_2[ispring, :] - Spring_bout_croix_1[ispring, :]) - l_repos_croix[ispring])

    F_masses = np.zeros((n*m, 3))
    F_masses[:, 2] = - M * 9.81

    return F_spring, F_spring_croix, F_masses

def Force_point(F_spring, F_spring_croix, F_masses):  # --> resultante des forces en chaque point a un instant donne
    """

    :param F_spring: cas.MX(Nb_ressorts, 3): force élastique de chaque ressort non oblique (dont ressorts du cadre)
    :param F_spring_croix: cas.MX(Nb_ressorts_croix, 3): force élastique de chaque ressort oblique
    :param F_masses: cas.MX(n*m,3): force de gravité appliquée à chaque point

    :return: F_point: cas.MX(n*m,3): résultantes des forces en chaque point
    """

    # forces elastiques
    F_spring_points = np.zeros((n * m, 3))

    # - points des coin de la toile : VERIFIE CEST OK
    F_spring_points[0, :] = F_spring[0, :] + \
                            F_spring[Nb_ressorts_cadre - 1, :] - \
                            F_spring[Nb_ressorts_cadre, :] - \
                            F_spring[Nb_ressorts_cadre + Nb_ressorts_horz, :] - \
                            F_spring_croix[0,:]  # en bas a droite : premier ressort du cadre + dernier ressort du cadre + premiers ressorts horz, vert et croix
    F_spring_points[n - 1, :] = F_spring[n - 1, :] + \
                                F_spring[n, :] - \
                                F_spring[Nb_ressorts_cadre + n - 1, :] + \
                                F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + Nb_ressorts_vert - m, :] - \
                                F_spring_croix[2 * (n - 1) - 1, :]  # en haut a droite
    F_spring_points[(m - 1) * n, :] = F_spring[2 * n + m - 1, :] + \
                                      F_spring[2 * n + m, :] + \
                                      F_spring[Nb_ressorts_cadre + (m - 2) * n, :] - \
                                      F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + m - 1, :] + \
                                      F_spring_croix[Nb_ressorts_croix - 2 * (n - 1) + 1, :]  # en bas a gauche
    F_spring_points[m * n - 1, :] = F_spring[n + m - 1, :] + \
                                    F_spring[n + m, :] + \
                                    F_spring[Nb_ressorts_cadre + Nb_ressorts_horz - 1, :] + \
                                    F_spring[Nb_ressorts - 1, :] + \
                                    F_spring_croix[Nb_ressorts_croix - 2, :]  # en haut a gauche


    # - points du bord de la toile> Pour lordre des termes de la somme, on part du ressort cadre puis sens trigo
    # - cote droit VERIFIE CEST OK
    for i in range(1, n - 1):
        F_spring_points[i, :] = F_spring[i, :] - \
                                F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + m * i, :] - \
                                F_spring_croix[2 * (i - 1) + 1, :] - \
                                F_spring[Nb_ressorts_cadre + i, :] - \
                                F_spring_croix[2 * (i - 1) + 2, :] + \
                                F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + m * (i - 1), :]

        # - cote gauche VERIFIE CEST OK
    j = 0
    for i in range((m - 1) * n + 1, m * n - 1):
        F_spring_points[i, :] = F_spring[Nb_ressorts_cadre - m - (2 + j), :] + \
                                F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + (j + 1) * m - 1, :] + \
                                F_spring_croix[Nb_ressorts_croix - 2 * n + 1 + 2 * (j + 2), :] + \
                                F_spring[Nb_ressorts_cadre + Nb_ressorts_horz - n + j + 1, :] + \
                                F_spring_croix[Nb_ressorts_croix - 2 * n + 2 * (j + 1), :] - \
                                F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + (j + 2) * m - 1, :]
        j += 1

        # - cote haut VERIFIE CEST OK
    j = 0
    for i in range(2 * n - 1, (m - 1) * n, n):
        F_spring_points[i, :] = F_spring[n + 1 + j, :] - \
                                F_spring[Nb_ressorts_cadre + i, :] - \
                                F_spring_croix[(j + 2) * (n - 1) * 2 - 1, :] + \
                                F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + (Nb_ressorts_vert + 1) - (m - j),
                                :] + \
                                F_spring_croix[(j + 1) * (n - 1) * 2 - 2, :] + \
                                F_spring[Nb_ressorts_cadre + i - n, :]
        j += 1


        # - cote bas VERIFIE CEST OK
    j = 0
    for i in range(n, (m - 2) * n + 1, n):
        F_spring_points[i, :] = F_spring[Nb_ressorts_cadre - (2 + j), :] + \
                                F_spring[Nb_ressorts_cadre + n * j, :] + \
                                F_spring_croix[1 + 2 * (n - 1) * j, :] - \
                                F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + j + 1, :] - \
                                F_spring_croix[2 * (n - 1) * (j + 1), :] - \
                                F_spring[Nb_ressorts_cadre + n * (j + 1), :]
        j += 1


    # Points du centre de la toile (tous les points qui ne sont pas en contact avec le cadre)
    # on fait une colonne puis on passe a la colonne de gauche etc
    # dans lordre de la somme : ressort horizontal de droite puis sens trigo
    for j in range(1, m - 1):
        for i in range(1, n - 1):
            F_spring_points[j * n + i, :] = F_spring[Nb_ressorts_cadre + (j - 1) * n + i, :] + \
                                            F_spring_croix[2 * j * (n - 1) - 2 * n + 3 + 2 * i, :] - \
                                            F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + m * i + j, :] - \
                                            F_spring_croix[j * 2 * (n - 1) + i * 2, :] - \
                                            F_spring[Nb_ressorts_cadre + j * n + i, :] - \
                                            F_spring_croix[j * 2 * (n - 1) + i * 2 - 1, :] + \
                                            F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + m * (i - 1) + j, :] + \
                                            F_spring_croix[j * 2 * (n - 1) - 2 * n + 2 * i, :]

    F_point = F_masses - F_spring_points

    return F_point

def Resultat_PF_collecte(participant, vide_name, trial_name, frame) :
    """
    Regroupe tous les calculs pour traiter les données des PF et des caméras et les retourner afin de les utiliser pour l'optimisation
    :param participant: int: numéro du participant correspondant à l'essai
    :param vide_name: str: nom de l'essai vide correspondant à l'essai statique
    :param trial_name: str: nom de l'essai dynamique étudié
    :param frame: int: numéro du frame étudié

    :return: F_out_cas: DM(3,1): somme des forces Fx,Fy,Fz de chaque point
    :return: Pt_out_cas: DM(3,127): position des points labellés de l'essai
    :return: labels: list(nombre_de_points_labellés_de_l_essai): utile pour la simu pour retrouver les bons indices
    :return: ind_marqueur_min: str: permet de retrouver l'indice du point d'application de la masse (utile pour la simu)
    """

    def open_c3d(participant, trial_name):
        """
      Trouvre et ouvre le bon fichier de la collecte
      :param participant: int: numéro du participant (0 si statique)
      :param trial_name: str: nom de l'essai que l'on veut ouvrir
      :return: c3d_file: c3d: fichier c3d
      """
        dossiers = ['statique', 'participant_01', 'participant_02']
        file_path = '/home/lim/Documents/Thea/Thea_final/Collecte/c3d_files/' + dossiers[participant]
        c3d_file = c3d(file_path + '/' + trial_name + '.c3d')
        return c3d_file

    def matrices():
        """
        :return: 4 matrices de calibrage des PF: list6x6
        """
        # M1,M2,M4 sont les matrices obtenues apres la calibration sur la plateforme 3
        M4_new = [[5.4526, 0.1216, 0.0937, -0.0001, -0.0002, 0.0001],
                  [0.4785, 5.7700, 0.0178, 0.0001, 0.0001, 0.0001],
                  [-0.1496, -0.1084, 24.6172, 0.0000, -0.0000, 0.0002],
                  [12.1726, -504.1483, -24.9599, 3.0468, 0.0222, 0.0042],
                  [475.4033, 10.6904, -4.2437, -0.0008, 3.0510, 0.0066],
                  [-6.1370, 4.3463, -4.6699, -0.0050, 0.0038, 1.4944]]

        M1_new = [[2.4752, 0.1407, 0.0170, -0.0000, -0.0001, 0.0001],
                  [0.3011, 2.6737, -0.0307, 0.0000, 0.0001, 0.0000],
                  [0.5321, 0.3136, 11.5012, -0.0000, -0.0002, 0.0011],
                  [20.7501, -466.7832, -8.4437, 1.2666, -0.0026, 0.0359],
                  [459.7415, 9.3886, -4.1276, -0.0049, 1.2787, -0.0057],
                  [265.6717, 303.7544, -45.4375, -0.0505, -0.1338, 0.8252]]

        M2_new = [[2.9967, -0.0382, 0.0294, -0.0000, 0.0000, -0.0000],
                  [-0.1039, 3.0104, -0.0324, -0.0000, -0.0000, 0.0000],
                  [-0.0847, -0.0177, 11.4614, -0.0000, -0.0000, -0.0000],
                  [13.6128, 260.5267, 17.9746, 1.2413, 0.0029, 0.0158],
                  [-245.7452, -7.5971, 11.5052, 0.0255, 1.2505, -0.0119],
                  [-10.3828, -0.9917, 15.3484, -0.0063, -0.0030, 0.5928]]

        M3_new = [[2.8992, 0, 0, 0, 0, 0],
                  [0, 2.9086, 0, 0, 0, 0],
                  [0, 0, 11.4256, 0, 0, 0],
                  [0, 0, 0, 1.2571, 0, 0],
                  [0, 0, 0, 0, 1.2571, 0],
                  [0, 0, 0, 0, 0, 0.5791]]

        return M1_new, M2_new, M3_new, M4_new

    def matrices_rotation():
        """

        :return: les matrices de rotation des PF142 par rapport à PF3: np.array((2,2))
        """
        theta31 = 0.53 * np.pi / 180
        rot31 = np.array([[np.cos(theta31), -np.sin(theta31)],
                          [np.sin(theta31), np.cos(theta31)]])

        theta34 = 0.27 * np.pi / 180
        rot34 = np.array([[np.cos(theta34), -np.sin(theta34)],
                          [np.sin(theta34), np.cos(theta34)]])

        theta32 = 0.94 * np.pi / 180
        rot32 = np.array([[np.cos(theta32), -np.sin(theta32)],
                          [np.sin(theta32), np.cos(theta32)]])

        return rot31,rot34,rot32

    def plateformes_separees_rawpins(c3d):
        """
        Sépare les données non calibrées du fichier c3d selon la plateforme correspondante
        :param c3d: c3d: fichier c3d de l'essai
        :return: platform: np.array(4,6,nombre_de_points_de_l_essai)
        """
        # on garde seulement les Raw pins
        force_labels = c3d['parameters']['ANALOG']['LABELS']['value']
        ind = []
        for i in range(len(force_labels)):
            if 'Raw' in force_labels[i]:
                ind = np.append(ind, i)
        # ind_stop=int(ind[0])
        indices = np.array([int(ind[i]) for i in range(len(ind))])
        ana = c3d['data']['analogs'][0, indices, :]
        platform1 = ana[0:6, :]  # pins 10 a 15
        platform2 = ana[6:12, :]  # pins 19 a 24
        platform3 = ana[12:18, :]  # pins 28 a 33
        platform4 = ana[18:24, :]  # pins 1 a 6

        platform = np.array([platform1, platform2, platform3, platform4])
        return platform

    def soustraction_zero(platform):  # soustrait juste la valeur du debut aux raw values
        """
        Soustrait aux données de chaque plateforme la moyenne de leurs premiers points
        :param platform: np.array(4,6,nombre_de_points_de_l_essai)
        :return: platform: np.array(4,6,nombre_de_points_de_l_essai)
        """
        longueur = np.size(platform[0, 0])
        zero_variable = np.zeros((4, 6))
        for i in range(6):
            for j in range(4):
                zero_variable[j, i] = np.mean(platform[j, i, 0:100])
                platform[j, i, :] = platform[j, i, :] - zero_variable[j, i] * np.ones(longueur)
        return platform

    def plateforme_calcul(platform):  # prend les plateformes separees, passe les N en Nmm, calibre, multiplie par mat rotation, met dans la bonne orientation
        """
        Ensemble de calcul sur les données des plateformes :
        - Transformer les données de moment des PF de Nm en Nmm
        - Multiplier chaque PF par sa matrice de calibrage et son facteur de correction
        - Mutiplier chaque PF par sa matrice de rotation
        - Donner a chaque PF la bonne orientation (x,y) comme sur la simulation
        - Diminuer par 4 la fréquence d'acquisition (on garde seulement 1 point sur 4)
        :param platform: np.array(4,6,nombre_de_points_de_l_essai)
        :return: platform_new : np.array(4,6,nombre_de_points_de_l_essai)
        """

        M1, M2, M3, M4 = matrices()
        rot31, rot34, rot32 = matrices_rotation()


        # Nm--> Nmm
        platform[:, 3:6] = platform[:, 3:6] * 1000

        # calibration
        platform[0] = np.matmul(M1, platform[0]) * 100
        platform[1] = np.matmul(M2, platform[1]) * 200
        platform[2] = np.matmul(M3, platform[2]) * 100
        platform[3] = np.matmul(M4, platform[3]) * 25

        # matrices de rotation ; nouvelle position des PF
        platform[0, 0:2] = np.matmul(rot31, platform[0, 0:2])
        platform[1, 0:2] = np.matmul(rot32, platform[1, 0:2])
        platform[3, 0:2] = np.matmul(rot34, platform[3, 0:2])

        # bonne orientation ; nouvelle position des PF (pas sure si avant ou apres)
        platform[0, 1] = -platform[0, 1]
        platform[1, 0] = -platform[1, 0]
        platform[2, 0] = -platform[2, 0]
        platform[3, 1] = -platform[3, 1]

        # prendre un point sur 4 pour avoir la même fréquence que les caméras
        platform_new = np.zeros((4, 6, int(np.shape(platform)[2] / 4)))
        for i in range(np.shape(platform)[2]):
            if i % 4 == 0:
                platform_new[:, :, i // 4] = platform[:, :, i]

        return platform_new

    def soustraction_vide(c3d_statique, c3d_vide):  # pour les forces calculees par Vicon
        """
        Pour un essai statique, soustrait les valeurs à vide des plateformes
        :param c3d_statique: c3d: données de l'essai statique
        :param c3d_vide: c3d: données de l'essai à vide correspondant (en général c'est labeled_statique_centrefront_vide)
        :return: platform: np.array(4,6,nombre_de_points_de_l_essai)
        """
        platform_statique = plateformes_separees_rawpins(c3d_statique)
        platform_vide = plateformes_separees_rawpins(c3d_vide)
        platform = np.copy(platform_statique)

        # on soustrait les valeurs du fichier a vide
        for j in range(6):
            for i in range(4):
                platform[i, j, :] = platform_statique[i, j, :] - np.mean(platform_vide[i, j, :])
        platform = plateforme_calcul(platform)
        return platform

    def dynamique(c3d_experimental):
        """
        Applique toutes les transformations nécessaires aux données des essais dynamiques
        :param c3d_experimental: données de l'essai dynamique
        :return:  platform: np.array(4,6,nombre_de_points_de_l_essai)
        """
        platform = plateformes_separees_rawpins(c3d_experimental)
        platform = soustraction_zero(platform)
        platform = plateforme_calcul(platform)
        return platform

    def Named_markers(c3d_experimental):
        """
        Trie les indices et les coordonnées des points qui ont été labellés
        :param c3d_experimental: données de l'essai dynamique
        :return: labels: list(nombre_de_points_labellés): liste des labels existants dans l'essai
        :return: moyenne_milieu: np.array(3): position moyenne du point du milieu de la toile (t67) sur les 100 premiers instants
        :return: named_positions: np.array(3, nombre_de_points_labellés, nombre_instants_de_l_essai): ensemble des positions correspondants aux labels
        """
        labels = c3d_experimental['parameters']['POINT']['LABELS']['value']

        indices_supp = []
        for i in range(len(labels)):
            if '*' in labels[i]:
                indices_supp = np.append(indices_supp, i)

        if len(indices_supp) == 0:
            ind_stop = int(len(labels))
        if len(indices_supp) != 0:
            ind_stop = int(indices_supp[0])

        labels = c3d_experimental['parameters']['POINT']['LABELS']['value'][0:ind_stop]
        named_positions = c3d_experimental['data']['points'][0:3, 0:ind_stop, :]

        ind_milieu = labels.index('t67')
        moyenne_milieu = np.array([np.mean(named_positions[i, ind_milieu, :100]) for i in range(3)])
        return labels,moyenne_milieu, named_positions

    def position_statique(named_positions,moyenne_vide):
        """
        Applique toutes les transformations pour calculer les données de l'essai statique :
        - soustraire à chaque point la moyenne des points de l'essai à vide (pour que t67 soit le point d'origine)
        - remettre les axes dans le meme sens que dans la modelisation
        - on convertit les mm en m
        :param named_positions: np.array(3, nombre_de_points_labellés, nombre_instants_de_l_essai)
        :param moyenne_vide: np.array(3): position moyenne du point du milieu de l'essai a vide
        :return: named_positions_bonsens: np.array(3, nombre_de_points_labellés, nombre_instants_de_l_essai)
        """
        # on soustrait la moyenne de la position du milieu
        for i in range(3):
            named_positions[i, :, :] = named_positions[i, :, :] - moyenne_vide[i]

        # on remet les axes dans le meme sens que dans la modelisation
        named_positions_bonsens = np.copy(named_positions)
        named_positions_bonsens[0, :, :] = - named_positions[1, :, :]
        named_positions_bonsens[1, :, :] = named_positions[0, :, :]

        #passage de mm en m :
        named_positions_bonsens *= 0.001

        return named_positions_bonsens

    def position_dynamique(named_positions, moyenne_milieu):
        """
        Applique toutes les transformations pour calculer les données de l'essai dynamique :
        - soustraire à chaque point la moyenne des points du début du même essai (pour que t67 soit le point d'origine)
        - remettre les axes dans le meme sens que dans la modelisation
        - on convertit les mm en m
        :param named_positions: np.array(3, nombre_de_points_labellés, nombre_instants_de_l_essai)
        :param moyenne_milieu: np.array(3): position moyenne du point du milieu de l'essai dynamique
        :return: named_positions_bonsens: np.array(3, nombre_de_points_labellés, nombre_instants_de_l_essai)
        """
        # on soustrait la moyenne de la position du milieu sur les 100 premiers points
        for i in range(3):
            named_positions[i, :, :] = named_positions[i, :, :] - moyenne_milieu[i]

        # on remet les axes dans le meme sens que dans la modelisation
        named_positions_bonsens = np.copy(named_positions)
        named_positions_bonsens[0, :, :] = - named_positions[1, :, :]
        named_positions_bonsens[1, :, :] = named_positions[0, :, :]

        #passage de mm en m :
        named_positions_bonsens*= 0.001

        return named_positions_bonsens

    def point_le_plus_bas(points, labels, frame):
        """

        :param points: np.array(3,nombre_de_points_labellés, nombre_instants_de_l_essai): points ayant déjà été modifiés (mis dans la bonne base, passage en m etc)
        :param labels: list(nombre_de_points_labellés): liste des labels existants dans l'essai
        :param frame: int: numéro du frame étudié
        :return: label_min: str: nom du label qui correspond au point d'altitude minimal sur le frame
        """


        indice_minimum = np.nanargmin(points[2, :, frame])
        label_min = labels[indice_minimum]
        # # garder seulement les tX et supprimer les points en C (cadre) et en M (milieu)
        while 'M' in label_min or 'C' in label_min :
            labels=np.delete(labels,indice_minimum)
            points = np.delete(points, indice_minimum, 1)
            indice_minimum = np.nanargmin(points[2, :, frame])
            label_min = labels[indice_minimum]

        return label_min

    if participant == 0 :
        c3d_vide = open_c3d(0, vide_name)
        c3d_statique = open_c3d(0, trial_name)
        platform = soustraction_vide(c3d_statique,c3d_vide)  # plateforme statique a laquelle on a soustrait la valeur de la plateforme a vide
        labels,moyenne_milieu, named_positions = Named_markers(c3d_statique)
        labels_vide,moyenne_milieu_vide, named_positions_vide = Named_markers(c3d_vide)
        Pt_collecte = position_statique(named_positions, moyenne_milieu_vide)

    else :
        c3d_experimental = open_c3d(participant, trial_name)
        platform = dynamique(c3d_experimental)
        labels,moyenne_milieu, named_positions = Named_markers(c3d_experimental)
        Pt_collecte = position_dynamique(named_positions, moyenne_milieu)

    longueur = np.size(platform[0, 0])
    F_totale_collecte = np.zeros((longueur,3))
    for i in range (3) :
        for x in range (longueur) :
            F_totale_collecte[x,i] = platform[0,i,x] + platform[1,i,x] + platform[2,i,x] + platform[3,i,x]

    # position_instant = Pt_collecte[:, :, int(7050)]
    label_min = point_le_plus_bas(Pt_collecte, labels,frame) #coordonnée du marqueur le plus bas dans labels
    ind_marqueur_min = int(label_min[1:])#coordonnées de ce marqueur adaptées à la simulation
    print('Point le plus bas sur l\'essai ' + trial_name + ' au frame ' + str(frame) + ' : ' + str(label_min))

    #on veut un seul frame donné :
    Pt_out = Pt_collecte[:,:,frame]
    F_out = F_totale_collecte[frame,:]

    # retourner des tableaux casadi
    F_out_cas = np.zeros(np.shape(F_out)[0])
    F_out_cas[:] = F_out[:]

    Pt_out_cas = np.zeros(np.shape(Pt_out))
    Pt_out_cas[:,:] = Pt_out[:,:]

    return F_out_cas,Pt_out_cas,labels,ind_marqueur_min

def interpolation_collecte(Pt_collecte, Pt_ancrage, labels) :
    """
    Interpoler lespoints manquants de la collecte pour les utiliser dans l'initial guess
    :param Pt_collecte: DM(3,135)
    :param labels: list(nombre de labels)
    :return: Pt_interpole: DM(3,135) (même dimension que Pos_repos)
    """
    #liste avec les bons points aux bons endroits, et le reste vaut 0
    Pt_interpole = np.zeros((3,135))
    Pt_interpole[:] = np.nan
    for ind in range (135) :
        if 't' + str(ind) in labels and np.isnan(Pt_collecte[0, labels.index('t' + str(ind))])==False :
            Pt_interpole[:,ind] = Pt_collecte[:, labels.index('t' + str(ind))]

    return Pt_interpole

def list2tab (list) :
    tab =np.zeros((135, 3))
    for i in range(135):
        tab[i, :] = list[:,i]
    return tab

def Calcul_Pt_F(X, Pt_ancrage, dict_fixed_params, K, ind_masse) :
    k, k_croix, M = Param_variable(K, ind_masse, K[12:])
    Pt = list2tab(X)

    Spring_bout_1, Spring_bout_2 = Spring_bouts(Pt, Pt_ancrage)
    Spring_bout_croix_1, Spring_bout_croix_2 = Spring_bouts_croix(Pt)
    F_spring, F_spring_croix, F_masses = Force_calc(Spring_bout_1, Spring_bout_2, Spring_bout_croix_1,
                                                    Spring_bout_croix_2, k, k_croix, M, dict_fixed_params, ind_masse)
    F_point = Force_point(F_spring, F_spring_croix, F_masses)

    #a verifier :
    F_totale = np.zeros(3)
    for ind in range (F_point.shape[0]) :
        for i in range (3) :
            F_totale[i] += F_point[ind,i]

    return F_totale, F_point

##########################################################################################################################

#ESSAI C3D A VERIFIER#
participant = 0
frame = 700
nb_disques = 8
trial_name = 'labeled_statique_centrefront_D' + str(nb_disques)
vide_name = 'labeled_statique_centrefront_vide'
if 'front' not in trial_name:
    vide_name = 'labeled_statique_vide'

masses = [0, 27.0, 47.1, 67.3, 87.4, 102.5, 122.6, 142.8, 163.0, 183.1, 203.3, 228.6]
Masse_centre = masses[nb_disques]

#raideurs obtenues
k1 = 1.21175669e+05
k2 = 3.20423906e+03
k3 = 4.11963416e+03
k4 = 2.48125477e+03
k5 = 7.56820743e+03
k6 = 4.95811865e+05
k7 = 1.30776275e-03
k8 = 3.23131678e+05
k1ob = 7.48735556e+02
k2ob = 1.08944449e-04
k3ob = 3.89409909e+03
k4ob = 1.04226031e-04
k=[k1, k2, k3, k4, k5, k6, k7, k8, k1ob, k2ob, k3ob, k4ob]

#masse réparties de maniere homogene
#M = [Masse_centre/5]*5
K=k
Masse_repartie = [4.71, 11.09, 7.04, 10.11, 14.12]
for i in range(0, len(Masse_repartie)):
    K.append(Masse_repartie[i])

#récupération des points de l'essais collecté
F_totale_collecte, Pt_collecte, labels, ind_masse = Resultat_PF_collecte(participant, vide_name, trial_name, frame)

#dict
dict_fixed_params = Param_fixe(ind_masse, Masse_centre)
Pt_ancrage, Pos_repos = Points_ancrage_repos(dict_fixed_params)

Pt_interpolés = interpolation_collecte(Pt_collecte, Pt_ancrage, labels) #liste des 135 points avec des 0 la ou il n'y a pas de label

#Calcul des forces en chaque points présents a la frame 700
k, k_croix, M = Param_variable(k, ind_masse, Masse_repartie)
Pt = list2tab(Pt_interpolés)
Spring_bout_1, Spring_bout_2 = Spring_bouts(Pt, Pt_ancrage)
Spring_bout_croix_1, Spring_bout_croix_2 = Spring_bouts_croix(Pt)

F_spring, F_spring_croix, F_masses = Force_calc(Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2, k, k_croix, M, dict_fixed_params)
F_point = Force_point(F_spring, F_spring_croix, F_masses)

#calcul de l'erreur :
#sur la position/force
    #on met tous les element de F_point au carre
for ind in range (n*m) :
    for i in range (3) :
        F_point[ind,i] = ((F_point[ind, i]) ** 2)

    #on somme sans les nan
err = np.nansum(F_point)

print('Erreur sur la force essai : ' + str(trial_name) + ' = ' + str(err) + ' N')

#ERREUR MOYENNE PAR POINT
#on divise l'erreur de force par le nombre de point de l'essai


"""Il faudrait verifier l'erreur de position entre les points de collecte et les points optimisés.
Mais pour les essais de verification, on ne dispose pas des positions des points optimisés.
On peut lancer une optimisation des essais de verification en fixant les k, et en optimisant seulement les points
"""

