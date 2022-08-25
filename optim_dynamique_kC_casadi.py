"""
Attention on a remis l'influence des ressorts obliques !!

A mettre en MX :
- Param_variable
- Spring_bouts
- Force_calc
- Force_point (voir pour lier Force_point et Force_calc)
- Etat initial
- rotation_points
"""

import casadi as cas
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
from ezc3d import c3d
import time


n = 15  # nombre de mailles sur le grand cote
m = 9  # nombre de mailles sur le petit cote

Nb_ressorts = 2 * n * m + n + m  # nombre de ressorts non obliques total dans le modele
Nb_ressorts_cadre = 2 * n + 2 * m  # nombre de ressorts entre le cadre et la toile
Nb_ressorts_croix = 2 * (m - 1) * (n - 1)  # nombre de ressorts obliques dans la toile
Nb_ressorts_horz = n * (m - 1)  # nombre de ressorts horizontaux dans la toile (pas dans le cadre)
Nb_ressorts_vert = m * (n - 1)  # nombre de ressorts verticaux dans la toile (pas dans le cadre)

#FONCTIONS AVEC LES PARAMÈTRES FIXES :

def Param_fixe(ind_masse,Masse_centre):
    #ESPACES EN TRE LES MARQUEURS :

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


    ##################################################################################################################
    # MASSES (pris en compte la masse ajoutee par lathlete)
    mcoin = 1.803  # masse d'un point se trouvant sur un coin de la toile
    mpetit = 0.5 * 5.695 / (m - 2)  # masse d'un point se trouvant sur le petit cote de la toile
    mgrand = 0.5 * 9.707 / (n - 2)  # masse d'un point se trouvant sur le grand cote de la toile
    mmilieu = 3 * 0.650 / ((n - 2) * (m - 2))  # masse d'un point se trouvant au milieu de la toile

    M = mmilieu * np.ones(n * m)  # on initialise toutes les masses a celle du centre
    M[0], M[n - 1], M[n * (m - 1)], M[n * m - 1] = mcoin, mcoin, mcoin, mcoin
    M[n:n * (m - 1):n] = mpetit  # masses du cote bas
    M[2 * n - 1:n * m - 1:n] = mpetit  # masses du cote haut
    M[1:n - 1] = mgrand  # masse du cote droit
    M[n * (m - 1) + 1:n * m - 1] = mgrand  # masse du cote gauche
    # M[int((m * n - 1) / 2)] += Masse_centre
    M[ind_masse] += Masse_centre
    # M[ind_masse] += Masse_centre/5
    # M[ind_masse + 1]+=Masse_centre/5
    # M[ind_masse - 1] += Masse_centre / 5
    # M[ind_masse + 15] += Masse_centre / 5
    # M[ind_masse - 15] += Masse_centre / 5

    dict_fixed_params = {'dL': dL,
                         'dLmilieu': dLmilieu,
                         'dl': dl,
                         'dlmilieu': dlmilieu,
                         'l_droite': l_droite,
                         'l_gauche': l_gauche,
                         'L_haut': L_haut,
                         'L_bas': L_bas,
                         'l_repos': l_repos,
                         'l_repos_croix': l_repos_croix,
                         'Masse_centre': Masse_centre,
                         'M': M}

    # return dL, dLmilieu, dl, dlmilieu, l_droite, l_gauche, L_haut, L_bas, l_repos ,l_repos_croix, M  # np
    return dict_fixed_params  # np

def Points_ancrage_repos_cas(dict_fixed_params):
    dL = dict_fixed_params['dL']
    dl = dict_fixed_params['dl']
    l_droite = dict_fixed_params['l_droite']
    l_gauche = dict_fixed_params['l_gauche']
    L_haut = dict_fixed_params['L_haut']
    L_bas = dict_fixed_params['L_bas']

    # repos :
    Pos_repos = cas.DM.zeros((n * m, 3))

    # on dit que le point numero 0 est a l'origine
    for j in range(m):
        for i in range(n):
            # Pos_repos[i + j * n] = np.array([-np.sum(dl[:j + 1]), np.sum(dL[:i + 1]), 0])
            Pos_repos[i + j * n,:] = cas.DM([-np.sum(dl[:j + 1]), np.sum(dL[:i + 1]), 0])

    Pos_repos_new = cas.DM.zeros((n * m, 3))
    for j in range(m):
        for i in range(n):
            Pos_repos_new[i + j * n,:] = Pos_repos[i + j * n,:] - Pos_repos[67,:]


    # ancrage :
    Pt_ancrage = cas.DM.zeros((2 * (n + m), 3))
    # cote droit :
    for i in range(n):
        Pt_ancrage[i, 1:2] = Pos_repos_new[i, 1:2]
        Pt_ancrage[i, 0] = l_droite
    # cote haut : on fait un truc complique pour center autour de l'axe vertical
    Pt_ancrage[n + 4, :] = cas.DM([0,L_haut, 0])
    for j in range(n, n + 4):
        Pt_ancrage[j, :] = cas.DM([0,L_haut, 0]) + cas.DM([np.sum(dl[1 + j - n:5]), 0, 0])
    for j in range(n + 5, n + m):
        Pt_ancrage[j, :] = cas.DM([0,L_haut, 0]) - cas.DM([np.sum(dl[5: j - n + 1]), 0, 0])
    # cote gauche :
    for k in range(n + m, 2 * n + m):
        Pt_ancrage[k, 1:2] = - Pos_repos_new[k - n - m, 1:2]
        Pt_ancrage[k, 0] = -l_gauche
    # cote bas :
    Pt_ancrage[2 * n + m + 4, :] = cas.DM([0, -L_bas, 0])

    Pt_ancrage[2 * n + m, :] = cas.DM([0, -L_bas, 0]) - cas.DM([np.sum(dl[5:9]), 0, 0])
    Pt_ancrage[2 * n + m + 1, :] = cas.DM([0, -L_bas, 0]) - cas.DM([np.sum(dl[5:8]), 0, 0])
    Pt_ancrage[2 * n + m + 2, :] = cas.DM([0, -L_bas, 0]) - cas.DM([np.sum(dl[5:7]), 0, 0])
    Pt_ancrage[2 * n + m + 3, :] = cas.DM([0, -L_bas, 0]) - cas.DM([np.sum(dl[5:6]), 0, 0])

    Pt_ancrage[2 * n + m + 5, :] = cas.DM([0, -L_bas, 0]) + cas.DM([np.sum(dl[4:5]), 0, 0])
    Pt_ancrage[2 * n + m + 6, :] = cas.DM([0, -L_bas, 0]) + cas.DM([np.sum(dl[3:5]), 0, 0])
    Pt_ancrage[2 * n + m + 7, :] = cas.DM([0, -L_bas, 0]) + cas.DM([np.sum(dl[2:5]), 0, 0])
    Pt_ancrage[2 * n + m + 8, :] = cas.DM([0, -L_bas, 0]) + cas.DM([np.sum(dl[1:5]), 0, 0])

    Pt_ancrage,Pos_repos_new = rotation_points(Pos_repos_new,Pt_ancrage)
    return Pt_ancrage, Pos_repos_new #np

def Points_ancrage_repos(dict_fixed_params):
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

    Pt_ancrage_cas = cas.DM(Pt_ancrage)
    Pos_repos_cas = cas.DM(Pos_repos)
    return Pt_ancrage_cas, Pos_repos_cas #np

def Spring_bouts_repos(Pos_repos, Pt_ancrage): #np
    # Definition des ressorts (position, taille)
    Spring_bout_1 = cas.MX.zeros((Nb_ressorts, 3))

    # RESSORTS ENTRE LE CADRE ET LA TOILE
    for i in range(0, Nb_ressorts_cadre):
        Spring_bout_1[i, :] = Pt_ancrage[i, :]

    # RESSORTS HORIZONTAUX : il y en a n*(m-1)
    for i in range(Nb_ressorts_horz):
        Spring_bout_1[Nb_ressorts_cadre + i, :] = Pos_repos[i, :]

    # RESSORTS VERTICAUX : il y en a m*(n
    k = 0
    for i in range(n - 1):
        for j in range(m):
            Spring_bout_1[Nb_ressorts_cadre + Nb_ressorts_horz + k, :] = Pos_repos[i + n * j, :]
            k += 1
    ####################################################################################################################
    Spring_bout_2 = cas.MX.zeros((Nb_ressorts, 3))

    # RESSORTS ENTRE LE CADRE ET LA TOILE
    for i in range(0, n):  # points droite du bord de la toile
        Spring_bout_2[i, :] = Pos_repos[i, :]

    k = 0
    for i in range(n - 1, m * n, n):  # points hauts du bord de la toile
        Spring_bout_2[n + k, :] = Pos_repos[i, :]
        k += 1

    k = 0
    for i in range(m * n - 1, n * (m - 1) - 1, -1):  # points gauche du bord de la toile
        Spring_bout_2[n + m + k, :] = Pos_repos[i, :]
        k += 1

    k = 0
    for i in range(n * (m - 1), -1, -n):  # points bas du bord de la toile
        Spring_bout_2[2 * n + m + k, :] = Pos_repos[i, :]
        k += 1

    # RESSORTS HORIZONTAUX : il y en a n*(m-1)
    k = 0
    for i in range(n, n * m):
        Spring_bout_2[Nb_ressorts_cadre + k, :] = Pos_repos[i, :]
        k += 1

    # RESSORTS VERTICAUX : il y en a m*(n-1)
    k = 0
    for i in range(1, n):
        for j in range(m):
            Spring_bout_2[Nb_ressorts_cadre + Nb_ressorts_horz + k, :] = Pos_repos[i + n * j, :]
            k += 1

    return (Spring_bout_1, Spring_bout_2)

def Spring_bouts_croix_repos(Pos_repos): #np
    # RESSORTS OBLIQUES : il n'y en a pas entre le cadre et la toile
    Spring_bout_croix_1 = cas.MX.zeros((Nb_ressorts_croix, 3))

    # Pour spring_bout_1 on prend uniquement les points de droite des ressorts obliques
    k = 0
    for i in range((m - 1) * n):
        Spring_bout_croix_1[k, :] = Pos_repos[i, :]
        k += 1
        # a part le premier et le dernier de chaque colonne, chaque point est relie a deux ressorts obliques
        if (i + 1) % n != 0 and i % n != 0:
            Spring_bout_croix_1[k, :] = Pos_repos[i, :]
            k += 1

    Spring_bout_croix_2 = cas.MX.zeros((Nb_ressorts_croix, 3))
    # Pour spring_bout_2 on prend uniquement les points de gauche des ressorts obliques
    # pour chaue carre on commence par le point en haut a gauche, puis en bas a gauche
    # cetait un peu complique mais ca marche, faut pas le changer
    j = 1
    k = 0
    while j < m:
        for i in range(j * n, (j + 1) * n - 2, 2):
            Spring_bout_croix_2[k, :] = Pos_repos[i + 1, :]
            Spring_bout_croix_2[k + 1, :] = Pos_repos[i, :]
            Spring_bout_croix_2[k + 2, :] = Pos_repos[i + 2, :]
            Spring_bout_croix_2[k + 3, :] = Pos_repos[i + 1, :]
            k += 4
        j += 1

    return Spring_bout_croix_1, Spring_bout_croix_2


#FONCTIONS AVEC LES PARAMÈTRES VARIABLES :

def param_type() :
    #donne les 40 C et les 12 k à optimiser, avant leur répartition sur les points et ressorts :
    k_type=cas.MX.sym('k',(1,12))
    C_symetrie=cas.MX.sym('C',5*8) #nombre de points sur un quart de la toile en comptant le milieu
    return k_type,C_symetrie

def Param_variable(k_type,C_symetrie):
    # k_type,C_symetrie = param_type()
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


    # # # k trouves a partir du programme 5x3:
    # k1 = (5 / n) * 3266.68  # un type de coin (ressort horizontal)
    # k2 = k1 * 2  # ressorts horizontaux du bord (bord vertical) : relient le cadre et la toile
    # k3 = (3 / m) * 3178.4  # un type de coin (ressort vertical)
    # k4 = k3 * 2  # ressorts verticaux du bord (bord horizontal) : relient le cadre et la toile
    # k5 = 4 / (n - 1) * 22866.79  # ressorts horizontaux du bord horizontal de la toile
    # k6 = 2 * k5  # ressorts horizontaux
    # k7 = 2 / (m - 1) * 23308.23  # ressorts verticaux du bord vertical de la toile
    # k8 = 2 * k7  # ressorts verticaux
    # # VALEURS INVENTÉES :
    # k_oblique_1 = np.mean((k1, k3))  # 4 ressorts des coins
    # k_oblique_2 = np.mean((k6,k7))  # ressorts des bords verticaux
    # k_oblique_3 = np.mean((k5, k4))  # ressorts des bords horizontaux
    # k_oblique_4 = np.mean((k6, k7))  # ressorts obliques quelconques

    # ressorts entre le cadre du trampoline et la toile : k1,k2,k3,k4
    # k_bord = np.zeros(Nb_ressorts_cadre)
    k_bord = cas.MX.zeros(Nb_ressorts_cadre)
    # cotes verticaux de la toile :
    k_bord[0:n], k_bord[n + m:2 * n + m] = k2, k2
    # cotes horizontaux :
    k_bord[n:n + m], k_bord[2 * n + m:2 * n + 2 * m] = k4, k4
    # coins :
    k_bord[0], k_bord[n - 1], k_bord[n + m], k_bord[2 * n + m - 1] = k1, k1, k1, k1
    k_bord[n], k_bord[n + m - 1], k_bord[2 * n + m], k_bord[2 * (n + m) - 1] = k3, k3, k3, k3

    # ressorts horizontaux dans la toile
    # k_horizontaux = k6 * np.ones(n * (m - 1))
    k_horizontaux = k6 * cas.MX.ones(n * (m - 1))
    k_horizontaux[0:n*(m-1):n] = k5  # ressorts horizontaux du bord DE LA TOILE en bas
    k_horizontaux[n - 1:n * (m - 1):n] = k5  # ressorts horizontaux du bord DE LA TOILE en haut

    # ressorts verticaux dans la toile
    # k_verticaux = k8 * np.ones(m * (n - 1))
    k_verticaux = k8 * cas.MX.ones(m * (n - 1))
    k_verticaux[0:m * (n - 1):m] = k7  # ressorts verticaux du bord DE LA TOILE a droite
    k_verticaux[m - 1:n * m - m:m] = k7  # ressorts verticaux du bord DE LA TOILE a gauche

    k = cas.vertcat(k_horizontaux, k_verticaux)
    k = cas.vertcat(k_bord, k)

######################################################################################################################

    # RESSORTS OBLIQUES
    #milieux :
    k_oblique = cas.MX.zeros(Nb_ressorts_croix)

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

    # k_oblique=np.reshape(k_oblique,(8,28)) #pour visualiser

#######################################################################################################################

    #COEFFICIENTS D'AMORTISSEMENT : la toile est séparéee en 4 quarts dont les C sont les mêmes par symétrie avec le centre
    # C_symetrie = 0.3*np.ones(5*8)
    C=cas.MX.zeros(n*m)

    # coin en bas a droite de la toile : c'est le coin qui définit tous les autres
    C[0:8] = C_symetrie[0:8]
    C[15:23] = C_symetrie[8:16]
    C[30:38] = C_symetrie[16:24]
    C[45:53] = C_symetrie[24:32]
    C[60:68] = C_symetrie[32:40]

    # coin en bas a gauche de la toile :
    C[75:83] = C_symetrie[24:32]
    C[90:98] = C_symetrie[16:24]
    C[105:113] = C_symetrie[8:16]
    C[120:128] = C_symetrie[0:8]

    #coin en haut a droite de la toile :
    C[14:7:-1] = C_symetrie[0:7]
    C[29:22:-1] = C_symetrie[8:15]
    C[44:37:-1] = C_symetrie[16:23]
    C[59:52:-1] = C_symetrie[24:31]
    C[74:67:-1] = C_symetrie[32:39]

    # coin en haut a gauche de la toile :
    C[89:82:-1] = C_symetrie[24:31]
    C[104:97:-1] = C_symetrie[16:23]
    C[119:112:-1] = C_symetrie[8:15]
    C[134:127:-1] = C_symetrie[0:7]

    return k,k_oblique, C  #  sx

def Spring_bouts(Pt, Pt_ancrage): #sx
    # Definition des ressorts (position, taille)
    Spring_bout_1 = cas.MX.zeros((Nb_ressorts, 3))

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
    Spring_bout_2 = cas.MX.zeros((Nb_ressorts, 3))

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
    # RESSORTS OBLIQUES : il n'y en a pas entre le cadre et la toile
    Spring_bout_croix_1 = cas.MX.zeros((Nb_ressorts_croix, 3))

    # Pour spring_bout_1 on prend uniquement les points de droite des ressorts obliques
    k = 0
    for i in range((m - 1) * n):
        Spring_bout_croix_1[k, :] = Pt[i, :]
        k += 1
        # a part le premier et le dernier de chaque colonne, chaque point est relie a deux ressorts obliques
        if (i + 1) % n != 0 and i % n != 0:
            Spring_bout_croix_1[k, :] = Pt[i, :]
            k += 1

    Spring_bout_croix_2 = cas.MX.zeros((Nb_ressorts_croix, 3))
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

def Etat_initial(Pt_ancrage, Pos_repos):
    Spring_bout_1, Spring_bout_2 = Spring_bouts_repos(Pos_repos, Pt_ancrage)
    Spring_bout_croix_1, Spring_bout_croix_2 = Spring_bouts_croix_repos(Pos_repos)
    return Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2

def Force_calc(Spring_bout_1, Spring_bout_2, Spring_bout_croix_1, Spring_bout_croix_2, k,k_oblique,dict_fixed_params):  # force dans chaque ressort

    M = dict_fixed_params['M']
    Masse_centre = dict_fixed_params['Masse_centre']
    l_repos = dict_fixed_params['l_repos']
    l_repos_croix = dict_fixed_params['l_repos_croix']

    # F_spring = np.zeros((Nb_ressorts, 3))
    F_spring = cas.MX.zeros((Nb_ressorts, 3))
    Vect_unit_dir_F = (Spring_bout_2 - Spring_bout_1) / cas.norm_fro(Spring_bout_2 - Spring_bout_1)
    for ispring in range(Nb_ressorts):
        F_spring[ispring, :] = Vect_unit_dir_F[ispring, :] * k[ispring] * (
                cas.norm_fro(Spring_bout_2[ispring, :] - Spring_bout_1[ispring, :]) - l_repos[ispring])

    # F_spring_croix = np.zeros((Nb_ressorts_croix, 3))
    F_spring_croix = cas.MX.zeros((Nb_ressorts_croix, 3))
    Vect_unit_dir_F_croix = (Spring_bout_croix_2 - Spring_bout_croix_1) / cas.norm_fro(
        Spring_bout_croix_2 - Spring_bout_croix_1)
    for ispring in range(Nb_ressorts_croix):
        F_spring_croix[ispring, :] = Vect_unit_dir_F_croix[ispring, :] * k_oblique[ispring] * (
                cas.norm_fro(Spring_bout_croix_2[ispring, :] - Spring_bout_croix_1[ispring, :]) - l_repos_croix[
            ispring])

    F_masses = cas.MX.zeros((n*m, 3))
    F_masses[:, 2] = - M * 9.81

    return F_spring, F_spring_croix, F_masses

def Force_point(F_spring, F_spring_croix, F_masses):  # --> resultante des forces en chaque point a un instant donne

    # forces elastiques
    F_spring_points = cas.MX.zeros((n * m, 3))

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

def rotation_points_cas (Pos_repos,Pt_ancrage) :
    mat_base_collecte = np.array([[ 0.99964304, -0.02650231,  0.00338079],
               [ 0.02650787,  0.99964731, -0.00160831],
               [-0.00333697,  0.00169736,  0.99999299]])
    #calcul inverse puis passage en casadi
    mat_base_inv_np = np.linalg.inv(mat_base_collecte)
    mat_base_inv = cas.MX(mat_base_inv_np)

    Pt_ancrage_new = cas.DM.zeros((Nb_ressorts_cadre,3))
    for index in range (Nb_ressorts_cadre) :
        Pt_ancrage_new[index,:] = cas.mtimes(Pt_ancrage[index,:], mat_base_inv) #multplication de matrices en casadi

    Pos_repos_new = cas.DM.zeros((n*m, 3))
    for index in range(n*m):
        Pos_repos_new[index, :] = cas.mtimes(Pos_repos[index, :], mat_base_inv)  # multplication de matrices en casadi

    return Pos_repos_new,Pt_ancrage_new

def rotation_points (Pos_repos,Pt_ancrage) :
    mat_base_collecte = np.array([[ 0.99964304, -0.02650231,  0.00338079],
               [ 0.02650787,  0.99964731, -0.00160831],
               [-0.00333697,  0.00169736,  0.99999299]])
    #calcul inverse
    mat_base_inv_np = np.linalg.inv(mat_base_collecte)

    Pt_ancrage_new = np.zeros((Nb_ressorts_cadre,3))
    for index in range (Nb_ressorts_cadre) :
        Pt_ancrage_new[index,:] = np.matmul(Pt_ancrage[index,:], mat_base_inv_np) #multplication de matrices en casadi

    Pos_repos_new = np.zeros((n*m, 3))
    for index in range(n*m):
        Pos_repos_new[index, :] = np.matmul(Pos_repos[index, :], mat_base_inv_np)  # multplication de matrices en casadi

    return Pos_repos_new,Pt_ancrage_new


def tab2list (tab) :
    list = cas.MX.zeros(135 * 3)
    for i in range(135):
        for j in range(3):
            list[j + 3 * i] = tab[i, j]
    return list

def list2tab (list) :
    tab = cas.MX.zeros(135, 3)
    for ind in range(135):
        for i in range(3):
            tab[ind, i] = list[i + 3 * ind]
    return tab

def Integration(x, u, dt, dict_fixed_params,Pt_ancrage,vitesse):

    M=dict_fixed_params['M']
    accel=cas.MX.zeros(135,3)
    # distribuer les variables x et u :

    #u = k_type ,C_symetrie :
    k_type = u[:12]
    C_symetrie = u[12:]
    k,k_oblique, C = Param_variable(k_type, C_symetrie)

    Pt=list2tab(x)

    # besoin de ça pour utiliser dans F
    Spb1, Spb2 = Spring_bouts(Pt, Pt_ancrage)
    Spbc1, Spbc2 = Spring_bouts_croix(Pt)

################################################################################################################
    #calcul de la force actuelle à partir des positions précédentes

    F_spring, F_spring_croix, F_masses = Force_calc(Spb1, Spb2, Spbc1, Spbc2, k,k_oblique, dict_fixed_params)  # calcule la force de chaque ressort
    F_point = Force_point(F_spring, F_spring_croix, F_masses)
    Force_amortissement = cas.MX.zeros(135, 3) #on l'initialise car on travaille sur ses indices

    F_totale = cas.MX.zeros(3)


    #calcul de la position actuelle à partir de la force actuelle
    for index in range(135):  # schema d'Euler pour l'integration
        Force_amortissement[index, :] = -C[index] * vitesse[index, :]  #vitesse de la boucle précédente
        accel[index, :] = (F_point[index, :] + Force_amortissement[index, :]) / M[index]
        vitesse[index, :] = dt * accel[index, :] + vitesse[index, :]
        Force_amortissement[index, :] = -C[index] * vitesse[index, :] #vitesse de la boucle actuelle
        Pt[index, :] = dt * vitesse[index, :] + Pt[index, :]
        for i in range (3) :
            F_totale[i] += Force_amortissement[index, i] + F_point[index, i]

    Pt_new = tab2list(Pt)

    func = cas.Function('F', [x, u], [Pt_new, F_totale, vitesse])

    return func

def Resultat_PF_collecte(participant,statique_name, vide_name, trial_name, intervalle_dyna) :
    def open_c3d(participant, trial_name):
        dossiers = ['c3d/statique', 'c3d/participant_01', 'c3d/participant_02', 'c3d/', 'c3d/test_plateformes']
        file_path = '/home/lim/Documents/UDEM_S2M_Thea/collecte/' + dossiers[participant]
        c3d_file = c3d(file_path + '/' + trial_name + '.c3d')
        return c3d_file

    def matrices():
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

        # zeros donnes par Nexus
        zeros1 = np.array([1.0751899, 2.4828501, -0.1168980, 6.8177500, -3.0313399, -0.9456340])
        zeros2 = np.array([0., -2., -2., 0., 0., 0.])
        zeros3 = np.array([0.0307411, -5., -4., -0.0093422, -0.0079338, 0.0058189])
        zeros4 = np.array([-0.1032560, -3., -3., 0.2141770, 0.5169040, -0.3714130])

        return M1_new, M2_new, M3_new, M4_new, zeros1, zeros2, zeros3, zeros4

    def matrices_rotation():
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
        longueur = np.size(platform[0, 0])
        zero_variable = np.zeros((4, 6))
        for i in range(6):
            for j in range(4):
                zero_variable[j, i] = np.mean(platform[j, i, 0:100])
                platform[j, i, :] = platform[j, i, :] - zero_variable[j, i] * np.ones(longueur)
        return platform

    def plateforme_calcul(platform,intervalle_dyna,participant):  # prend les plateformes separees, passe les N en Nmm, calibre, multiplie par mat rotation, met dans la bonne orientation

        M1, M2, M3, M4, zeros1, zeros2, zeros3, zeros4 = matrices()
        rot31, rot34, rot32 = matrices_rotation()


        # N--> Nmm
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

        if participant !=0 :
            platform_new = platform_new[:,:,intervalle_dyna[0]:intervalle_dyna[1]]

        return platform_new

    def soustraction_vide(c3d_statique, c3d_vide):  # pour les forces calculees par Vicon
        platform_statique = plateformes_separees_rawpins(c3d_statique)
        platform_vide = plateformes_separees_rawpins(c3d_vide)
        platform = np.copy(platform_statique)

        # on soustrait les valeurs du fichier a vide
        for j in range(6):
            for i in range(4):
                platform[i, j, :] = platform_statique[i, j, :] - np.mean(platform_vide[i, j, :])
        platform = plateforme_calcul(platform,0,0)
        return platform

    def dynamique(c3d_experimental,intervalle_dyna,participant):
        platform = plateformes_separees_rawpins(c3d_experimental)
        platform = soustraction_zero(platform)
        platform = plateforme_calcul(platform,intervalle_dyna,participant)
        return platform

    def Named_markers(c3d_experimental):
        labels = c3d_experimental['parameters']['POINT']['LABELS']['value']

        indices_supp = []
        for i in range(len(labels)):
            if '*' in labels[i]:
                indices_supp = np.append(indices_supp, i)
        ind_stop = int(indices_supp[0])

        labels = c3d_experimental['parameters']['POINT']['LABELS']['value'][0:ind_stop]
        named_positions = c3d_experimental['data']['points'][0:3, 0:ind_stop, :]

        ind_milieu = labels.index('t67')
        moyenne_milieu = np.array([np.mean(named_positions[i, ind_milieu, :100]) for i in range(3)])
        return labels,moyenne_milieu, named_positions

    def position_statique(named_positions, moyenne,named_positions_vide,moyenne_vide):
        # on soustrait la moyenne de la position du milieu
        for i in range(3):
            named_positions[i, :, :] = named_positions[i, :, :] - moyenne_vide[i]

        # on remet les axes dans le meme sens que dans la modelisation
        named_positions_bonsens = np.copy(named_positions)
        named_positions_bonsens[0, :, :] = - named_positions[1, :, :]
        named_positions_bonsens[1, :, :] = named_positions[0, :, :]

        position_moyenne = np.zeros((3, np.shape(named_positions_bonsens)[1]))
        for ind_print in range(np.shape(named_positions_bonsens)[1]):
            position_moyenne[:, ind_print] = np.array(
                [np.mean(named_positions_bonsens[i, ind_print, :100]) for i in range(3)])

        #passage de mm en m :
        named_positions_bonsens *= 0.001

        return named_positions_bonsens

    def position_dynamique(named_positions, moyenne_milieu, intervalle_dyna):
        # on soustrait la moyenne de la position du milieu sur les 100 premiers points
        for i in range(3):
            named_positions[i, :, :] = named_positions[i, :, :] - moyenne_milieu[i]

        # on remet les axes dans le meme sens que dans la modelisation
        named_positions_bonsens = np.copy(named_positions)
        named_positions_bonsens[0, :, :] = - named_positions[1, :, :]
        named_positions_bonsens[1, :, :] = named_positions[0, :, :]

        position_moyenne = np.zeros((3, np.shape(named_positions_bonsens)[1]))
        for ind_print in range(np.shape(named_positions_bonsens)[1]):
            position_moyenne[:, ind_print] = np.array(
                [np.mean(named_positions_bonsens[i, ind_print, :100]) for i in range(3)])

        #passage de mm en m :
        named_positions_bonsens*= 0.001

        positions_new = named_positions_bonsens[:,:,intervalle_dyna[0] : intervalle_dyna[1]]

        return positions_new

    def point_le_plus_bas(points, labels,intervalle_dyna):
        # on cherche le z min obtenu sur notre intervalle

        #compter le nombre de nan par marqueur au cours de l'intervalle :
        isnan_marqueur = np.zeros(len(labels))
        for i in range (len(labels)) :
            # for time in range (intervalle_dyna[0], intervalle_dyna[1]) :
            for time in range(intervalle_dyna[1] - intervalle_dyna[0]):
                if np.isnan(points[2, i,time])==True :
                    isnan_marqueur[i] += 1

        #trier les marqueurs en t qui sont des nan sur tout l'intervalle
        labels_notnan = []
        for i in range (len(labels)) :
            #s'il y a autant de nan que de points dans l'intervalle, alors on n'en veut pas :
            if isnan_marqueur[i] != intervalle_dyna[1] - intervalle_dyna[0] :
                labels_notnan += [labels[i]]

        indice_notnan= []
        for i in range (len (labels_notnan)) :
            indice_notnan += [labels.index(labels_notnan[i])]

        labels_modified=[labels[i] for i in indice_notnan]
        points_modified = points[:,indice_notnan,:]

        #on peut enfin calculer le minimum :
        # on cherche le z min de chaque marqueur (on en profite pour supprimer les nan)
        # minimum_marqueur = [np.nanmin(points_modified[2, i,intervalle_dyna[0] : intervalle_dyna[1]]) for i in range(len(indice_notnan))]
        minimum_marqueur = [np.nanmin(points_modified[2, i, :]) for i in range(len(indice_notnan))]

        #indice du marqueur ayant le plus petit z sur tout l'intervalle
        argmin_marqueur = np.argmin(minimum_marqueur)
        label_min = labels_modified[argmin_marqueur]

        return argmin_marqueur,label_min

    if participant == 0 :
        c3d_vide = open_c3d(0, vide_name)
        c3d_statique = open_c3d(0, statique_name)
        platform = soustraction_vide(c3d_statique,c3d_vide)  # plateforme statique a laquelle on a soustrait la valeur de la plateforme a vide
        labels,moyenne_milieu, named_positions = Named_markers(c3d_statique)
        labels_vide,moyenne_milieu_vide, named_positions_vide = Named_markers(c3d_vide)
        Pt_collecte = position_statique(named_positions,moyenne_milieu, named_positions_vide, moyenne_milieu_vide)

    else :
        c3d_experimental = open_c3d(participant, trial_name)
        platform = dynamique(c3d_experimental,intervalle_dyna,participant)
        labels,moyenne_milieu, named_positions = Named_markers(c3d_experimental)
        Pt_collecte = position_dynamique(named_positions, moyenne_milieu,intervalle_dyna)

    longueur = np.size(platform[0, 0])
    F_totale_collecte = np.zeros((longueur,3))
    for i in range (3) :
        for x in range (longueur) :
            F_totale_collecte[x,i] = platform[0,i,x] + platform[1,i,x] + platform[2,i,x] + platform[3,i,x]

    # position_instant = Pt_collecte[:, :, int(7050)]
    argmin_marqueur,label_min = point_le_plus_bas(Pt_collecte, labels,intervalle_dyna) #coordonnée du marqueur le plus bas dans labels
    ind_marqueur_min = int(label_min[1:])#coordonnées de ce marqueur adaptées à la simulation
    print('Point le plus bas sur l\'intervalle ' + str(intervalle_dyna) + ' : ' + str(label_min))

    #retourner des tableaux casadi
    F_collecte_cas = cas.DM.zeros(np.shape(F_totale_collecte))
    F_collecte_cas[:,:] = F_totale_collecte[:,:]

    Pt_collecte_tab=[0 for i in range (np.shape(Pt_collecte)[2])]
    for time in range (np.shape(Pt_collecte)[2]) :
        #passage en casadi
        Pt_time = cas.DM.zeros(3,np.shape(Pt_collecte)[1])
        Pt_time[:,:] = Pt_collecte[:,:,time] #attention pas dans le même ordre que Pt_simu
        #séparation en Nb_increments tableaux
        Pt_collecte_tab[time] = Pt_time

    return F_collecte_cas,Pt_collecte_tab,labels,ind_marqueur_min

#####################################################################################################################

def Optimisation() :  # main

    def kC_bounds (Uk_K,Uk_C) : #initial guess pour les k et les C
        k1 = (5 / n) * 3266.68  # un type de coin (ressort horizontal)
        k2 = k1 * 2  # ressorts horizontaux du bord (bord vertical) : relient le cadre et la toile
        k3 = (3 / m) * 3178.4  # un type de coin (ressort vertical)
        k4 = k3 * 2  # ressorts verticaux du bord (bord horizontal) : relient le cadre et la toile
        k5 = 4 / (n - 1) * 22866.79  # ressorts horizontaux du bord horizontal de la toile
        k6 = 2 * k5  # ressorts horizontaux
        k7 = 2 / (m - 1) * 23308.23  # ressorts verticaux du bord vertical de la toile
        k8 = 2 * k7  # ressorts verticaux
        # VALEURS INVENTÉES :
        k_oblique1 = np.mean((k1, k3))  # 4 ressorts des coins
        k_oblique2 = k5  # ressorts des bords verticaux
        k_oblique3 = np.mean((k5, k8))  # ressorts des bords horizontaux
        k_oblique4 = np.mean((k6, k7))  # ressorts obliques quelconques
        k_croix = 3000  # je sais pas

        # A CHANGER :
        C = [0.2]*Uk_C.shape[0]

        w0_k = [k1, k2, k3, k4, k5, k6, k7, k8, k_oblique1, k_oblique2, k_oblique3, k_oblique4]
        w0_C = C

        lbw_k = [100]*Uk_K.shape[0]
        ubw_k = [100000]*Uk_K.shape[0] # bornes très larges

        lbw_C = [1e-3]*Uk_C.shape[0]
        ubw_C = [1e+3]*Uk_C.shape[0]

        return w0_k, w0_C, lbw_k, ubw_k, lbw_C, ubw_C

    def Pt_bounds(x,Pos) :
        #bounds and initial guess
        lbw_Pt = []
        ubw_Pt = []
        w0_Pt = []

        for k in range (x.shape[0]) :
            if k % 3 == 0 :#limites et guess en x
                lbw_Pt +=[Pos[int(k//3),0] - 0.3]
                ubw_Pt += [Pos[int(k // 3), 0] + 0.3]
                w0_Pt += [Pos[int(k // 3), 0]]
            if k % 3 == 1: #limites et guess en y
                lbw_Pt += [Pos[int(k // 3), 1] - 0.3]
                ubw_Pt += [Pos[int(k // 3), 1] + 0.3]
                w0_Pt += [Pos[int(k // 3), 1]]
            if k % 3 == 2: #limites et guess en z
                lbw_Pt += [-2]
                ubw_Pt += [0.5]
                w0_Pt += [Pos[int(k // 3), 2]]

        return lbw_Pt, ubw_Pt,w0_Pt

    def Difference_func(Pt, F_totale, Pt_collecte, F_totale_collecte) :
        Difference = cas.MX.zeros(1)
        Pt=list2tab(Pt)
        for i in range (3) :
            Difference += (F_totale_collecte[i] - F_totale[i]) ** 2
            for ind in range (n*m) :
                if 't'+ str(ind) in labels :
                    ind_collecte = labels.index('t'+ str(ind)) # ATTENTION gérer les nans
                    if np.isnan(Pt_collecte[i, ind_collecte])== False :
                        Difference += (Pt[ind,i] - Pt_collecte[i, ind_collecte])**2

        return Difference



    # RÉSULTATS COLLECTE :
    participant = 1
    statique_name = 'labeled_statique_leftfront_D7'
    trial_name  = 'labeled_p1_sauthaut_01'
    intervalle_dyna = [7030, 7164] #dépend de l'essai (utiliser plateforme_verification_toutesversions pour toruver l'intervalle)
    # intervalle_dyna = [0, 11000]
    # intervalle_dyna = [7005, 7180]
    intervalle_dyna = [0,2]
    vide_name = 'labeled_statique_centrefront_vide'
    F_totale_collecte, Pt_collecte_tab, labels, ind_masse = Resultat_PF_collecte(participant, statique_name,vide_name,trial_name,intervalle_dyna)


    #VERIF COLLECTE :
    # plt.figure(0)
    # ind = labels.index('t55')
    # Pt_collecte = np.zeros((len(Pt_collecte_tab),3,len(labels)))
    # for time in range (0,len(Pt_collecte_tab)) :
    #     Pt_collecte[time] = np.array(Pt_collecte_tab[time])
    # for i in range (3) :
    #     plt.subplot(3,1,i+1)
    #     plt.plot(Pt_collecte[:,i,ind])
    #
    # plt.figure(1)
    # F_collecte = np.zeros((len(Pt_collecte_tab),3))
    # for time in range (len(Pt_collecte_tab)) :
    #     F_collecte[time] = np.array(F_totale_collecte[time])
    # for i in range (3) :
    #     plt.subplot(3,1,i+1)
    #     plt.plot(np.array(F_totale_collecte[:,i]))
    #
    # plt.show()

    # F_totale_collecte_inter = F_totale_collecte[intervalle_dyna[0] : intervalle_dyna[1], :]
    # Pt_collecte_tab_inter = Pt_collecte_tab[intervalle_dyna[0] : intervalle_dyna[1]]


    #PARAM A FAIRE VARIER
    masses=[64.5,87.2]
    Masse_centre = 140
    Masse_centre = masses[0]
    # Nb_frame = 4 # nb de frames de l'essai
    Nb_frame = intervalle_dyna[1] - intervalle_dyna[0]
    dt = 1/500  # fréquence des caméras

    #PARAM FIXES
    n = 15
    m = 9
    dict_fixed_params = Param_fixe(ind_masse, Masse_centre)
    Pt_ancrage, Pos_repos = Points_ancrage_repos(dict_fixed_params)

    #VARIABLES A OPTIMISER :
    # Xk = position et vitesse des points, Uk = K et C
    X_sym = cas.MX.sym('X', 135 * 3)
    U_sym = cas.MX.sym('U', 52)

    #INITIALISATION :
    # Spb1, Spb2, Spbc1, Spbc2 = Etat_initial(Pt_ancrage, Pos_repos)  # --> actualise Pt[0,:,:]

    vitesse = cas.MX.zeros(135,3)
    F = Integration(X_sym, U_sym, dt, dict_fixed_params,Pt_ancrage,vitesse)

    #OPTIMISATION :
    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []

    #NLP VALUES
    Uk_K = cas.MX.sym('K', 12)
    Uk_C = cas.MX.sym('C', 40)
    Xk = cas.MX.sym('X_0', 135 * 3)  # xyz pour chaque point (xyz_0, xyz_1, ...)
    lbw_Pt0, ubw_Pt0, w0_Pt0 = Pt_bounds(Xk, Pos_repos)
    w0_k, w0_C, lbw_k, ubw_k, lbw_C, ubw_C = kC_bounds(Uk_K,Uk_C)

    # w=[k,C,Pt] :
    w += [Uk_K]
    w += [Uk_C]
    w += [Xk]
    lbw += lbw_k
    lbw += lbw_C
    lbw += lbw_Pt0
    ubw += ubw_k
    ubw += ubw_C
    ubw += ubw_Pt0
    w0 += w0_k
    w0 += w0_C
    w0 += w0_Pt0

    for k in range(Nb_frame - 1):
        Pt_collecte = Pt_collecte_tab[k] #voir si bon indice
        F_collecte = F_totale_collecte[k,:]

        # fonction
        Pt, F_totale, vitesse = F(Xk, cas.vertcat(Uk_K, Uk_C))
        J = Difference_func(Pt, F_totale, Pt_collecte, F_collecte) # fonction a modiffier

        # New NLP variable for state at end of interval
        Xk = cas.MX.sym('X_' + str(k+1), 135 * 3)  # xyz poir chaque point (xyz_0, xyz_1, ...)
        w += [Xk]
        # lbw_Ptk, ubw_Ptk, w0_Ptk = Pt_bounds(Xk, Pt_collecte.T) # probleme il faudraiyt interpoler les Pt_collecte sur les nan
        lbw_Ptk, ubw_Ptk, w0_Ptk = Pt_bounds(Xk, Pos_repos)
        lbw += lbw_Ptk
        ubw += ubw_Ptk
        w0 += w0_Ptk

        # Add equality constraint
        g += [Pt - Xk]
        lbg += [0] * 135 * 3
        ubg += [0] * 135 * 3
        print('fin de boucle : k=' + str(k))

    # Create an NLP solver
    prob = {'f': J, 'x': cas.vertcat(*w), 'g': cas.vertcat(*g)}
    print('définition du problème')
    start_solver = time.time()
    # opts={"ipopt" : {"linear_solver" : "ma57", "tol" : 1e-4, "constr_viol_tol" : 1e-4, "constr_inf_tol" : 1e-4, "hessian_approximation" : "limited-memory"}}
    opts = {"ipopt": {"linear_solver": "ma57", "tol": 1e-4, "hessian_approximation" : "limited-memory"}}
    solver = cas.nlpsol('solver', 'ipopt', prob, opts)
    end_solver = time.time()

    # Solve the NLP
    # sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    print('solver après ' + str(end_solver-start_solver) + ' s.')
    sol = solver(x0=cas.vertcat(*w0), lbg=cas.vertcat(*lbg), ubg=cas.vertcat(*ubg), lbx=cas.vertcat(*lbw),
           ubx=cas.vertcat(*ubw))
    w_opt = sol['x'].full().flatten()

    return w_opt





start_main = time.time()
Solution = Optimisation()
print ('k = ' +str(Solution[:12]))
print ('C = ' +str(Solution[12:52]))
Pt = Solution[52:]
Pt_time = np.zeros((int(np.shape(Pt)[0]/(135*3)),135,3))
for i in range (int(np.shape(Pt)[0]/(135*3))) :
    Pt_time[i] = np.reshape(Pt[i*135*3:(i+1)*135*3],(135,3))
print ('Pt = ' + str(Pt_time))
end_main = time.time()
print ('Temps total : ' + str(end_main - start_main))

# Pt_save2d = np.zeros((int(np.shape(Pt)[0]/(135*3)),135*3))
# for i in range (int(np.shape(Pt)[0]/(135*3))) :
#     Pt_save2d[i] = np.reshape(Pt[i*135*3:(i+1)*135*3],(135*3))

np.savetxt('Pt_liste', Pt)