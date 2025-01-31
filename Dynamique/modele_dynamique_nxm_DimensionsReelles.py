"""
Modele dynamique avec deux possibilités d'affichage : subplot en plusieurs instants ou enregistrement de video
Les ecarts entre les points du maillage (positions au repos) sont ceux mesurés sur le trampo réel
k issus de la these de Jacques (2008)
On affiche aussi la force, la position, la vitesse et l'accélération du point auquel on met la masse
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from ezc3d import c3d
import seaborn as sns
from scipy import signal

#####################################################################################################################
# Le programme dynamique totalement simule fonctionne, mais on veut maintenant
# mettre les vraies valeurs des parametres du trampo>
# FAIT - ecart entre les points de la toile
# - ecarts entre les points de la toile et ceux du cadre
# - 8 points du maillage plus fin au centre
# - vraies longueurs au repos
# - vraies raideurs et longueurs au repos de la toile
# - vraies raideurs et longueurs au repos des ressorts du cadre
# - vraies masses en chaque point
######################################################################################################################
"""
Ce programme calcule et affiche les positions des points de la toile de trampoline 
On a utilisé les mêmes mesures que sur le trampo réel (positions des marqueurs)
"""

#ACTION :
affichage= 'subplot' #'subplot' #'animation'
masse_type = 'repartie' #'repartie' #'ponctuelle'

#PARAMETRES :
n=15 #nombre de mailles sur le grand cote
m=9 #nombre de mailles sur le petit cote
Masse_centre=80
#PARAMETRES POUR LA DYNAMIQUE :
dt = 0.002 #fonctionne pour dt<0.004

#NE PAS CHANGER :
Nb_ressorts=2*n*m+n+m #nombre de ressorts non obliques total dans le modele
Nb_ressorts_cadre=2*n+2*m #nombre de ressorts entre le cadre et la toile
Nb_ressorts_croix=2*(m-1)*(n-1) #nombre de ressorts obliques dans la toile
Nb_ressorts_horz=n * (m - 1) #nombre de ressorts horizontaux dans la toile (pas dans le cadre)
Nb_ressorts_vert=m * (n - 1) #nombre de ressorts verticaux dans la toile (pas dans le cadre)


def longueurs() :
    #de bas en haut :
    dL = np.array([510.71703748, 261.87522103, 293.42186099, 298.42486747, 300.67352585,
                   298.88879749, 299.6946861, 300.4158083, 304.52115312, 297.72780618,
                   300.53723415, 298.27144226, 298.42486747, 293.42186099, 261.87522103,
                   510.71703748]) * 0.001  # ecart entre les lignes, de bas en haut
    dL = np.array([np.mean([dL[i], dL[-(i+1)]]) for i in range (16)])

    dLmilieu = np.array([151.21983556, 153.50844775]) * 0.001
    dLmilieu = np.array([np.mean([dLmilieu[i], dLmilieu[-(i + 1)]]) for i in range(2)])
    #de droite a gauche :
    dl = np.array(
        [494.38703513, 208.96708367, 265.1669672, 254.56358938, 267.84760997, 268.60351809, 253.26974254, 267.02823864,
         208.07894712, 501.49013437]) * 0.001
    dl = np.array([np.mean([dl[i], dl[-(i + 1)]]) for i in range(10)])

    dlmilieu = np.array([126.53897435, 127.45173517]) * 0.001
    dlmilieu = np.array([np.mean([dlmilieu[i], dlmilieu[-(i + 1)]]) for i in range(2)])

    l_droite = np.sum(dl[:5])
    l_gauche = np.sum(dl[5:])

    L_haut = np.sum(dL[:8])
    L_bas = np.sum(dL[8:])

    ################################
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

    # ressorts obliques internes a la toile :
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

    return dict_fixed_params

def Param():

    #k multistart

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

    # ressorts entre le cadre du trampoline et la toile : k1,k2,k3,k4
    k_bord = np.zeros(Nb_ressorts_cadre)

    # cotes verticaux de la toile :
    k_bord[0:n], k_bord[n + m:2 * n + m] = k2, k2

    # cotes horizontaux :
    k_bord[n:n + m], k_bord[2 * n + m:2 * n + 2 * m] = k4, k4

    # coins :
    k_bord[0], k_bord[n - 1], k_bord[n + m], k_bord[2 * n + m - 1] = k1, k1, k1, k1
    k_bord[n], k_bord[n + m - 1], k_bord[2 * n + m], k_bord[2 * (n + m) - 1] = k3, k3, k3, k3

    #ressorts horizontaux dans la toile
    k_horizontaux = k6 * np.ones(n * (m - 1))
    k_horizontaux[0:n * m - 1:n] = k5  # ressorts horizontaux du bord DE LA TOILE en bas
    k_horizontaux[n - 1:n * (m - 1):n] = k5  # ressorts horizontaux du bord DE LA TOILE en haut

    # ressorts verticaux dans la toile
    k_verticaux = k8 * np.ones(m * (n - 1))
    k_verticaux[0:m * (n - 1):m] = k7  # ressorts verticaux du bord DE LA TOILE a droite
    k_verticaux[m - 1:n * m - 1:m] = k7  # ressorts verticaux du bord DE LA TOILE a gauche


    k = np.append(k_horizontaux, k_verticaux)
    k = np.append(k_bord, k)

    # ressorts obliques dans la toile
    k_oblique = np.ones(Nb_ressorts_croix)
    k_oblique[0], k_oblique[1] = k1ob, k1ob  # en bas a droite
    k_oblique[2 * (n - 1) - 1], k_oblique[2 * (n - 1) - 2] = k1ob, k1ob  # en haut a droite
    k_oblique[Nb_ressorts_croix - 1], k_oblique[Nb_ressorts_croix - 2] = k1ob, k1ob  # en haut a gauche
    k_oblique[2 * (n - 1) * (m - 2)], k_oblique[2 * (n - 1) * (m - 2) + 1] = k1ob, k1ob  # en bas a gauche

    # côtés verticaux :
    k_oblique[2: 2 * (n - 1) - 2] = k2ob  # côté droit
    k_oblique[2 * (n - 1) * (m - 2) + 2: Nb_ressorts_croix - 2] = k2ob  # côté gauche

    # côtés horizontaux :
    k_oblique[28:169:28], k_oblique[29:170:28] = k3ob, k3ob  # en bas
    k_oblique[55:196:28], k_oblique[54:195:28] = k3ob, k3ob  # en haut

    # milieu :
    for j in range(1, 7):
        k_oblique[2 + 2 * j * (n - 1): 26 + 2 * j * (n - 1)] = k4ob


    ##################################################################################################################
    #COEFFICIENTS D'AMORTISSEMENT a changer
    C = 2*np.ones(n*m)

    ##################################################################################################################
    # MASSES (pris en compte la masse ajoutee par lathlete) :
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

    M = mmilieu * np.ones((n * m))  # on initialise toutes les masses a celle du centre
    M[0], M[n - 1], M[n * (m - 1)], M[n * m - 1] = mcoin, mcoin, mcoin, mcoin
    M[n:n * (m - 1):n] = mpetitbord  # masses du cote bas
    M[2 * n - 1:n * m - 1:n] = mpetitbord  # masses du cote haut
    M[1:n - 1] = mgrandbord  # masse du cote droit
    M[n * (m - 1) + 1:n * m - 1] = mgrandbord  # masse du cote gauche

    return k, k_oblique, M, C


def Points_ancrage_fix(dict_fixed_params):

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

    Pt_ancrage, Pos_repos_new = rotation_points2(Pt_ancrage,Pos_repos_new)

    return Pt_ancrage,Pos_repos_new


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
            Pos_repos[i + j * n, :] = np.array([-np.sum(dl[:j + 1]), np.sum(dL[:i + 1]), 0])

    Pos_repos_new = np.zeros((n * m, 3))
    for j in range(m):
        for i in range(n):
            Pos_repos_new[i + j * n, :] = Pos_repos[i + j * n, :] - Pos_repos[67, :]
    # Pos_repos_new = np.copy(Pos_repos)

    # ancrage :
    Pt_ancrage = np.zeros((2 * (n + m), 3))
    # cote droit :
    for i in range(n):
        Pt_ancrage[i, 1:2] = Pos_repos_new[i, 1:2]
        Pt_ancrage[i, 0] = l_droite
    # cote haut : on fait un truc complique pour center autour de l'axe vertical
    Pt_ancrage[n + 4, :] = np.array([0, L_haut, 0])
    for j in range(n, n + 4):
        Pt_ancrage[j, :] = np.array([0, L_haut, 0]) + np.array([np.sum(dl[1 + j - n:5]), 0, 0])
    for j in range(n + 5, n + m):
        Pt_ancrage[j, :] = np.array([0, L_haut, 0]) - np.array([np.sum(dl[5: j - n + 1]), 0, 0])
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

    Pt_ancrage = rotation_points(Pt_ancrage)

    return Pt_ancrage, Pos_repos

def Spring_bouts_repos(Pos_repos,Pt_ancrage,time,Nb_increments):
    # Definition des ressorts (position, taille)
    Spring_bout_1=np.zeros((Nb_increments,Nb_ressorts,3))

    # RESSORTS ENTRE LE CADRE ET LA TOILE
    for i in range(0, Nb_ressorts_cadre):
        Spring_bout_1[time,i,:] = Pt_ancrage[time,i, :]

    # RESSORTS HORIZONTAUX : il y en a n*(m-1)
    for i in range(Nb_ressorts_horz):
        Spring_bout_1[time,Nb_ressorts_cadre + i,:] = Pos_repos[time,i,:]

    # RESSORTS VERTICAUX : il y en a m*(n-1)
    k=0
    for i in range(n - 1):
        for j in range(m):
            Spring_bout_1[time,Nb_ressorts_cadre+Nb_ressorts_horz+k, :] = Pos_repos[time,i + n * j,:]
            k+=1

    Spring_bout_2=np.zeros((Nb_increments,Nb_ressorts,3))

    # RESSORTS ENTRE LE CADRE ET LA TOILE
    for i in range(0, n): # points droite du bord de la toile
        Spring_bout_2[time,i,:] = Pos_repos[time,i,:]

    k=0
    for i in range(n - 1, m * n, n): # points hauts du bord de la toile
        Spring_bout_2[time, n+k, :] = Pos_repos[time, i, :]
        k+=1

    k=0
    for i in range(m*n-1,n * (m - 1)-1, -1): # points gauche du bord de la toile
        Spring_bout_2[time, n + m + k, :] = Pos_repos[time, i, :]
        k+=1

    k=0
    for i in range(n * (m - 1), -1, -n): # points bas du bord de la toile
        Spring_bout_2[time, 2*n + m + k, :] = Pos_repos[time, i, :]
        k+=1

    # RESSORTS HORIZONTAUX : il y en a n*(m-1)
    k=0
    for i in range(n, n * m):
        Spring_bout_2[time,Nb_ressorts_cadre + k,:] = Pos_repos[time,i,:]
        k+=1

    # RESSORTS VERTICAUX : il y en a m*(n-1)
    k=0
    for i in range(1, n):
        for j in range(m):
            Spring_bout_2[time,Nb_ressorts_cadre + Nb_ressorts_horz + k,:] = Pos_repos[time,i + n * j,:]
            k+=1

    return (Spring_bout_1,Spring_bout_2)

def Spring_bouts_croix_repos(Pos_repos,time,Nb_increments):
    #RESSORTS OBLIQUES : il n'y en a pas entre le cadre et la toile
    Spring_bout_croix_1=np.zeros((Nb_increments,Nb_ressorts_croix,3))

    #Pour spring_bout_1 on prend uniquement les points de droite des ressorts obliques
    k=0
    for i in range ((m-1)*n):
        Spring_bout_croix_1[time,k,:]=Pos_repos[time,i,:]
        k += 1
        #a part le premier et le dernier de chaque colonne, chaque point est relie a deux ressorts obliques
        if (i+1)%n!=0 and i%n!=0 :
            Spring_bout_croix_1[time, k, :] = Pos_repos[time, i, :]
            k+=1

    Spring_bout_croix_2=np.zeros((Nb_increments,Nb_ressorts_croix,3))
    #Pour spring_bout_2 on prend uniquement les points de gauche des ressorts obliques
    #pour chaue carre on commence par le point en haut a gauche, puis en bas a gauche
    #cetait un peu complique mais ca marche, faut pas le changer
    j=1
    k = 0
    while j<m:
        for i in range (j*n,(j+1)*n-2,2):
            Spring_bout_croix_2[time,k,:] = Pos_repos[time,i + 1,:]
            Spring_bout_croix_2[time,k+1,:] = Pos_repos[time,i,:]
            Spring_bout_croix_2[time,k+2,:] = Pos_repos[time,i+ 2,:]
            Spring_bout_croix_2[time,k+3,:] = Pos_repos[time,i + 1,:]
            k += 4
        j+=1

    return Spring_bout_croix_1,Spring_bout_croix_2

def Spring_bouts(Pt,Pt_ancrage,time,Nb_increments):
    # Definition des ressorts (position, taille)
    Spring_bout_1=np.zeros((Nb_increments,Nb_ressorts,3))

    # RESSORTS ENTRE LE CADRE ET LA TOILE
    for i in range(0, Nb_ressorts_cadre):
        Spring_bout_1[time,i,:] = Pt_ancrage[i, :]

    # RESSORTS HORIZONTAUX : il y en a n*(m-1)
    for i in range(Nb_ressorts_horz):
        Spring_bout_1[time,Nb_ressorts_cadre + i,:] = Pt[i,:]

    # RESSORTS VERTICAUX : il y en a m*(n-1)
    k=0
    for i in range(n - 1):
        for j in range(m):
            Spring_bout_1[time,Nb_ressorts_cadre+Nb_ressorts_horz+k, :] = Pt[i + n * j,:]
            k+=1
####################################################################################################################
    Spring_bout_2=np.zeros((Nb_increments,Nb_ressorts,3))

    # RESSORTS ENTRE LE CADRE ET LA TOILE
    for i in range(0, n): # points droite du bord de la toile
        Spring_bout_2[time,i,:] = Pt[i,:]

    k=0
    for i in range(n - 1, m * n, n): # points hauts du bord de la toile
        Spring_bout_2[time, n+k, :] = Pt[ i, :]
        k+=1

    k=0
    for i in range(m*n-1,n * (m - 1)-1, -1): # points gauche du bord de la toile
        Spring_bout_2[time, n + m + k, :] = Pt[ i, :]
        k+=1

    k=0
    for i in range(n * (m - 1), -1, -n): # points bas du bord de la toile
        Spring_bout_2[time, 2*n + m + k, :] = Pt[ i, :]
        k+=1

    # RESSORTS HORIZONTAUX : il y en a n*(m-1)
    k=0
    for i in range(n, n * m):
        Spring_bout_2[time,Nb_ressorts_cadre + k,:] = Pt[i,:]
        k+=1

    # RESSORTS VERTICAUX : il y en a m*(n-1)
    k=0
    for i in range(1, n):
        for j in range(m):
            Spring_bout_2[time,Nb_ressorts_cadre + Nb_ressorts_horz + k,:] = Pt[i + n * j,:]
            k+=1

    return Spring_bout_1, Spring_bout_2

def Spring_bouts_croix(Pt,time,Nb_increments):
    #RESSORTS OBLIQUES : il n'y en a pas entre le cadre et la toile
    Spring_bout_croix_1=np.zeros((Nb_increments,Nb_ressorts_croix,3))

    #Pour spring_bout_1 on prend uniquement les points de droite des ressorts obliques
    k=0
    for i in range ((m-1)*n):
        Spring_bout_croix_1[time,k,:]=Pt[i,:]
        k += 1
        #a part le premier et le dernier de chaque colonne, chaque point est relie a deux ressorts obliques
        if (i+1)%n!=0 and i%n!=0 :
            Spring_bout_croix_1[time, k, :] = Pt[i, :]
            k+=1

    Spring_bout_croix_2=np.zeros((Nb_increments,Nb_ressorts_croix,3))
    #Pour spring_bout_2 on prend uniquement les points de gauche des ressorts obliques
    #pour chaue carre on commence par le point en haut a gauche, puis en bas a gauche
    #cetait un peu complique mais ca marche, faut pas le changer
    j=1
    k = 0
    while j<m:
        for i in range (j*n,(j+1)*n-2,2):
            Spring_bout_croix_2[time,k,:] = Pt[i + 1,:]
            Spring_bout_croix_2[time,k+1,:] = Pt[i,:]
            Spring_bout_croix_2[time,k+2,:] = Pt[i+ 2,:]
            Spring_bout_croix_2[time,k+3,:] = Pt[i + 1,:]
            k += 4
        j+=1

    return Spring_bout_croix_1,Spring_bout_croix_2

def rotation_points (Pt_ancrage) : #(Pt_ancrage, Pos_repos)
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

    # Pos_repos_new = np.zeros((n * m, 3))
    # for index in range(n * m):
    #     Pos_repos_new[index, :] = np.matmul(Pos_repos[index, :], mat_base_inv_np)

    return Pt_ancrage_new#, Pos_repos_new

def rotation_points2(Pt_ancrage, Pos_repos) :
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

    Pos_repos_new = np.zeros((n * m, 3))
    for index in range(n * m):
        Pos_repos_new[index, :] = np.matmul(Pos_repos[index, :], mat_base_inv_np)

    return Pt_ancrage_new, Pos_repos_new

def Force_calc(Spring_bout_1,Spring_bout_2,Spring_bout_croix_1,Spring_bout_croix_2, Masse_centre, time , Nb_increments): #force en chaque ressort
    k, k_oblique, M, C = Param()
    l_repos = dict_fixed_params['l_repos']
    l_repos_croix = dict_fixed_params['l_repos_croix']

    F_spring = np.zeros((Nb_ressorts, 3))
    Vect_unit_dir_F = np.zeros((Nb_ressorts, 3))
    for i in range(Nb_ressorts):
        Vect_unit_dir_F[i, :] = (Spring_bout_2[i, :] - Spring_bout_1[i, :]) / np.linalg.norm(
            Spring_bout_2[i, :] - Spring_bout_1[i, :])
    # Vect_unit_dir_F = (Spring_bout_2 - Spring_bout_1) / cas.norm_fro(Spring_bout_2 - Spring_bout_1)
    for ispring in range(Nb_ressorts):
        F_spring[ispring, :] = Vect_unit_dir_F[ispring, :] * k[ispring] * (
                np.linalg.norm(Spring_bout_2[ispring, :] - Spring_bout_1[ispring, :]) - l_repos[ispring])

    # F_spring_croix = np.zeros((Nb_ressorts_croix, 3))
    F_spring_croix = np.zeros((Nb_ressorts_croix, 3))
    Vect_unit_dir_F_croix = np.zeros((Nb_ressorts, 3))
    for i in range(Nb_ressorts_croix):
        Vect_unit_dir_F_croix[i, :] = (Spring_bout_croix_2[i, :] - Spring_bout_croix_1[i, :]) / np.linalg.norm(
            Spring_bout_croix_2[i, :] - Spring_bout_croix_1[i, :])
    for ispring in range(Nb_ressorts_croix):
        F_spring_croix[ispring, :] = Vect_unit_dir_F_croix[ispring, :] * k_oblique[ispring] * (
                np.linalg.norm(Spring_bout_croix_2[ispring, :] - Spring_bout_croix_1[ispring, :]) - l_repos_croix[
            ispring])
    F_masses = np.zeros((n * m, 3))
    F_masses[:, 2] = - M * 9.81

    return M, F_spring, F_spring_croix, F_masses

def Force_point(F_spring,F_spring_croix,F_masses,time,Nb_increments) : #--> resultante des forces en chaque point a un instant donne

    #forces elastiques
    F_spring_points = np.zeros((n*m,3))

    # - points des coin de la toile : VERIFIE CEST OK
    F_spring_points[0,:]=F_spring[0,:]+\
                         F_spring[Nb_ressorts_cadre-1,:]-\
                         F_spring[Nb_ressorts_cadre,:]- \
                         F_spring[Nb_ressorts_cadre+Nb_ressorts_horz,:] -\
                         F_spring_croix[0,:]# en bas a droite : premier ressort du cadre + dernier ressort du cadre + premiers ressorts horz, vert et croix
    F_spring_points[n-1,:] = F_spring[n-1,:] +\
                              F_spring[n,:] - \
                              F_spring[ Nb_ressorts_cadre + n - 1,:] + \
                              F_spring[ Nb_ressorts_cadre + Nb_ressorts_horz + Nb_ressorts_vert-m,:] - \
                              F_spring_croix[2*(n-1)-1,:]  # en haut a droite
    F_spring_points[ (m-1)*n,:] = F_spring[ 2*n+m-1,:] +\
                                  F_spring[ 2*n+m,:] + \
                                  F_spring[ Nb_ressorts_cadre + (m-2)*n,:] - \
                                  F_spring[ Nb_ressorts_cadre + Nb_ressorts_horz + m-1,:] + \
                                  F_spring_croix[ Nb_ressorts_croix - 2*(n-1) +1,:]  # en bas a gauche
    F_spring_points[ m* n-1,:] = F_spring[ n + m - 1,:] + \
                                 F_spring[ n + m,: ] + \
                                 F_spring[ Nb_ressorts_cadre + Nb_ressorts_horz-1,:] + \
                                 F_spring[ Nb_ressorts-1,:] + \
                                 F_spring_croix[ Nb_ressorts_croix-2,:]  # en haut a gauche

    # - points du bord de la toile> Pour lordre des termes de la somme, on part du ressort cadre puis sens trigo
            # - cote droit VERIFIE CEST OK
    for i in range (1,n-1):
        F_spring_points[ i,:] = F_spring[ i,:] - \
                                F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + m * i,:] - \
                                F_spring_croix[ 2 * (i - 1) + 1,:] - \
                                F_spring[ Nb_ressorts_cadre + i,:] - \
                                F_spring_croix[ 2 * (i - 1)+2,:] + \
                                F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + m * (i - 1),:]
            # - cote gauche VERIFIE CEST OK
    j=0
    for i in range((m-1)*n+1, m*n-1):
        F_spring_points[i,:]=F_spring[Nb_ressorts_cadre - m - (2+j),:] + \
                             F_spring[Nb_ressorts_cadre+Nb_ressorts_horz+(j+1)*m-1,:]+ \
                             F_spring_croix[Nb_ressorts_croix-2*n+1+2*(j+2),:]+\
                             F_spring[Nb_ressorts_cadre+Nb_ressorts_horz-n+j+1,:]+\
                             F_spring_croix[Nb_ressorts_croix-2*n+2*(j+1),:]-\
                             F_spring[Nb_ressorts_cadre+Nb_ressorts_horz+(j+2)*m-1,:]
        j+=1

            # - cote haut VERIFIE CEST OK
    j=0
    for i in range (2*n-1,(m-1)*n,n) :
        F_spring_points[ i,:]= F_spring[ n+1+j,:] - \
                               F_spring[ Nb_ressorts_cadre + i,:] - \
                               F_spring_croix[(j+2)*(n-1)*2-1,:]+\
                               F_spring[Nb_ressorts_cadre + Nb_ressorts_horz + (Nb_ressorts_vert+1) - (m-j),:] +\
                               F_spring_croix[(j+1)*(n-1)*2-2,:]+\
                               F_spring[ Nb_ressorts_cadre + i-n,:]
        j+=1
            # - cote bas VERIFIE CEST OK
    j=0
    for i in range (n,(m-2)*n+1,n) :
        F_spring_points[i,:] = F_spring[ Nb_ressorts_cadre-(2+j),:] + \
                                F_spring[ Nb_ressorts_cadre + n*j,:]+\
                                F_spring_croix[1+2*(n-1)*j,:]-\
                                F_spring[Nb_ressorts_cadre+Nb_ressorts_horz+j+1,:]-\
                                F_spring_croix[2*(n-1)*(j+1),:]-\
                                F_spring[ Nb_ressorts_cadre + n*(j+1),:]
        j+=1

    #Points du centre de la toile (tous les points qui ne sont pas en contact avec le cadre)
    #on fait une colonne puis on passe a la colonne de gauche etc
    #dans lordre de la somme : ressort horizontal de droite puis sens trigo
    for j in range (1,m-1):
        for i in range (1,n-1) :
            F_spring_points[j*n+i,:]=F_spring[Nb_ressorts_cadre+(j-1)*n+i,:] + \
                                     F_spring_croix[2*j*(n-1) - 2*n + 3 + 2*i,:]-\
                                     F_spring[Nb_ressorts_cadre+Nb_ressorts_horz + m*i + j,:]-\
                                     F_spring_croix[j*2*(n-1) + i*2,:]-\
                                     F_spring[ Nb_ressorts_cadre + j * n + i,:]-\
                                     F_spring_croix[j*2*(n-1) + i*2 -1,:]+\
                                     F_spring[Nb_ressorts_cadre+Nb_ressorts_horz + m*(i-1) + j,:]+\
                                     F_spring_croix[j*2*(n-1) -2*n + 2*i,:]

    F_point = F_masses - F_spring_points

    return F_point

def Resultat_PF_collecte(participant,statique_name, vide_name, trial_name, intervalle_dyna) :
    def open_c3d(participant, trial_name):
        dossiers = ['c3d/statique', 'c3d/participant_01', 'c3d/participant_02', 'c3d/', 'c3d/test_plateformes']
        # file_path = '/home/lim/Documents/Thea/UDEM_S2M_Thea_WIP/collecte/' + dossiers[participant]
        # c3d_file = c3d(file_path + '/' + trial_name + '.c3d')


        file_path = '/home/lim/Documents/Thea/UDEM_S2M_Thea_WIP/collecte/c3d/participant_02/labeled_p2_troisquartback_01.c3d'
        c3d_file = c3d(file_path)


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

        if len(indices_supp) == 0:
            ind_stop = int(len(labels))
        if len(indices_supp) != 0:
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

    def point_le_plus_bas(points, labels, intervalle_dyna):
        """

        :param points: liste des 3 coordonnées des points labelisés pour sur l'intervalle de frame
        :param labels: labels des points des essais a chaque frame
        :param intervalle_dyna: intervalle de frame choisi
        :return:
        le min ???? a voir
        """
        idx_min = []
        position_min = []
        labels_min = []
        for frame in range(0, intervalle_dyna[1] - intervalle_dyna[0]):
            minimum_cal = np.nanargmin(points[2, :, frame])
            position_min.append(np.nanmin(points[2, :, frame]))
            idx_min.append(minimum_cal)
            labels_min.append(labels[minimum_cal])
        position_min_dynamique = np.min(position_min)
        idx_min_dynamique = np.argmin(position_min)
        label_min_dynamique = labels_min[idx_min_dynamique]


        # # on cherche le z min obtenu sur notre intervalle
        # #compter le nombre de nan par marqueur au cours de l'intervalle :
        # isnan_marqueur = np.zeros(len(labels))
        # for i in range (len(labels)) :
        #     # for time in range (intervalle_dyna[0], intervalle_dyna[1]) :
        #     for time in range(intervalle_dyna[1] - intervalle_dyna[0]):
        #         if np.isnan(points[2, i,time])==True :
        #             isnan_marqueur[i] += 1
        #
        # # trier les marqueurs en t qui sont des nan sur tout l'intervalle
        # labels_notnan = []
        # for i in range (len(labels)):
        #     #s'il y a autant de nan que de points dans l'intervalle, alors on n'en veut pas  = on enleve les marqueurs qui n'ont pas ete detecté
        #     if isnan_marqueur[i] != intervalle_dyna[1] - intervalle_dyna[0] :
        #         labels_notnan += [labels[i]]
        #
        # indice_notnan= []
        # for i in range (len (labels_notnan)) :
        #     indice_notnan += [labels.index(labels_notnan[i])]
        #
        # labels_modified=[labels[i] for i in indice_notnan]
        # points_modified = points[:,indice_notnan,:] #ensemble des points de toutes les frames MAIS sans les NaN
        #
        # #on peut enfin calculer le minimum :
        # # on cherche le z min de chaque marqueur (on en profite pour supprimer les nan)
        # # minimum_marqueur = [np.nanmin(points_modified[2, i,intervalle_dyna[0] : intervalle_dyna[1]]) for i in range(len(indice_notnan))]
        # minimum_marqueur = [np.nanmin(points_modified[2, i, :]) for i in range(len(indice_notnan))]
        #
        # #indice du marqueur ayant le plus petit z sur tout l'intervalle
        # argmin_marqueur = np.argmin(minimum_marqueur)
        # label_min = labels_modified[argmin_marqueur]
        #
        # return argmin_marqueur, label_min
        return idx_min_dynamique, label_min_dynamique

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
    F_collecte_cas = np.zeros(np.shape(F_totale_collecte))
    F_collecte_cas[:,:] = F_totale_collecte[:,:]

    Pt_collecte_tab=[0 for i in range (np.shape(Pt_collecte)[2])]
    for time in range (np.shape(Pt_collecte)[2]) :
        #passage en casadi
        Pt_time = np.zeros((3,np.shape(Pt_collecte)[1]))
        Pt_time[:,:] = Pt_collecte[:,:,time] #attention pas dans le même ordre que Pt_simu
        #séparation en Nb_increments tableaux
        Pt_collecte_tab[time] = Pt_time

    return F_collecte_cas,Pt_collecte_tab,labels,ind_marqueur_min

def Point_ancrage(Point_collecte, labels):
    """
    :param Point_collecte: ensemble des points de collecte de l'intervalle dynamique
    :param labels: labels des points
    :return:
    ensemble des coordonnées des points d'ancrage a chaque frame (point du cadre avec label C) sous forme de tableaux
    """
    point_ancrage = []
    label_ancrage = []
    for frame in range (len(Point_collecte)):
        pt_ancrage_frame = []
        for idx, lab in enumerate(labels):
            if 'C' in lab:
                pt_ancrage_frame.append(Point_collecte[frame][:,idx])

                if frame == 0:
                    label_ancrage.append(lab)
        point_ancrage.append(pt_ancrage_frame)

    #transformation des elements en array
    liste_point_ancrage = []
    for list_point in point_ancrage:
        tab_point = np.array((list_point[0]))
        for ligne in range(1, len(list_point)):
            tab_point = np.vstack((tab_point, list_point[ligne]))
        liste_point_ancrage.append(tab_point)

    # for point_anc in liste_point_ancrage:
    #     point_anc = rotation_points(point_anc)

    return liste_point_ancrage, label_ancrage


def Point_toile_init(Point_collecte, labels):
    """
    :param Point_collecte: ensemble des points de collecte de l'intervalle dynamique
    :param labels: labels des points
    :return:
    ensemble des coordonnées des points a chaque frame
    """
    point_toile = []
    for frame in range (len(Point_collecte)):
        pt_toile_frame = []
        for idx, lab in enumerate(labels):
            if 'M' in lab or 't' in lab:
                pt_toile_frame.append(Point_collecte[frame][:,idx])
        point_toile.append(pt_toile_frame)

    # transformation des elements en array
    liste_point_toile = []
    for list_point in point_toile:
        tab_point = np.array((list_point[0]))
        for ligne in range(1, len(list_point)):
            tab_point = np.vstack((tab_point, list_point[ligne]))
        liste_point_toile.append(tab_point)

    return liste_point_toile

def interpolation_collecte(Pt_collecte, labels) :
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

def Bouts_ressort_collecte(Pt_interpolés, nb_frame):
    """
    :param Pt_interpolés: point collecte
    :param nb_frame: nombre de frame dans l'intervalle dynamique

    :return: bouts des ressorts, coordonnées
    """
    frame = 0
    Spring_bouts1, Spring_bouts2 = Spring_bouts(Pt_interpolés.T, Pt_ancrage_repos, frame, nb_frame)
    Spring_bouts1, Spring_bouts2 = Spring_bouts1[0], Spring_bouts2[0]
    Spring_bouts_croix1, Spring_bouts_croix2 = Spring_bouts_croix(Pt_interpolés.T, frame, nb_frame)
    Spring_bouts_croix1, Spring_bouts_croix2 = Spring_bouts_croix1[0], Spring_bouts_croix2[0]

    return Spring_bouts1, Spring_bouts2, Spring_bouts_croix1, Spring_bouts_croix2


def Affichage_points_collecte_t(Pt_toile, Pt_ancrage, Ressort , nb_frame, ind_masse):
    """

    :param Pt_toile: points de la toile collectés, avec interpolation ou non des points inexistants en Nan
    :param Pt_ancrage: points du cadre collectés, avec interpolation ou non des points inexistants en Nan
    :param ressort: Booleen, if true on affiche les ressorts

    """
    bout1, bout2, boutc1, boutc2 = Bouts_ressort_collecte(Pt_toile, nb_frame)

    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1.1, 1.8, 1])
    ax.plot(0, 0, -1.2, 'ow')  # mettre a une echelle lisible et reelle

    #afficher les points d'ancrage et les points de la toile avec des couleurs differentes
    ax.plot(Pt_ancrage[:, 0], Pt_ancrage[:, 1], Pt_ancrage[:, 2], 'ok', label = 'Point du cadre ')
    ax.plot(Pt_toile[0, :], Pt_toile[1, :], Pt_toile[2, :], 'ob', label='Point de la toile')
    ax.plot(Pt_toile[0, ind_masse], Pt_toile[1, ind_masse], Pt_toile[2, ind_masse], 'og', label='Point le plus bas sur l\'intervalle dynamique')


    if Ressort == True:
        #affichage des ressort
        for j in range(Nb_ressorts):
            a = []
            a = np.append(a, bout1[j, 0])
            a = np.append(a, bout2[j, 0])

            b = []
            b = np.append(b, bout1[j, 1])
            b = np.append(b, bout2[j, 1])

            c = []
            c = np.append(c, bout1[j, 2])
            c = np.append(c, bout2[j, 2])

            ax.plot3D(a, b, c, '-r', linewidth=1)

        for j in range(Nb_ressorts_croix):
            # pas tres elegant mais cest le seul moyen pour que ca fonctionne
            a = []
            a = np.append(a, boutc1[j, 0])
            a = np.append(a, boutc2[j, 0])

            b = []
            b = np.append(b, boutc1[j, 1])
            b = np.append(b, boutc2[j, 1])

            c = []
            c = np.append(c, boutc1[j, 2])
            c = np.append(c, boutc2[j, 2])

            ax.plot3D(a, b, c, '-g', linewidth=1)

    return ax

def Etat_initial(Ptavant, Ptapres, labels):
    """
    :param Pt intervalle dynamique
    :param labels
    :return:
    vitesse initiale a l'instant -1, de la frame 0 de l'intervalle dynamique
    """

    position_imoins1 = interpolation_collecte(Ptavant, labels)
    position_iplus1 = interpolation_collecte(Ptapres, labels)
    distance_xyz = np.abs(position_imoins1 - position_iplus1)
    vitesse_initiale = distance_xyz / (2 * 0.002)

    return vitesse_initiale

def erreurs(Pt_intergres, Pt_frame2):
    """
    :param Pt_intergres: calcule par intergration
    :param Pt_frame2: points a l'instant i+1
    :return:
    erreur absolue et erreur relative de la différence des positions des points
    """

    position_iplus1 = interpolation_collecte(Pt_frame2, labels)
    point_theorique = position_iplus1.T

    err_abs = (np.abs(point_theorique - Pt_intergres) / point_theorique) * 100
    err_rel = np.abs(point_theorique - Pt_intergres)

    # print('erreur relative : '+ str(err_rel))
    # print('erreur absolue : ' + str(err_abs))

    return err_rel, err_abs

def update(time, Pt, markers_point):
    for i_point in range(len(markers_point)):
        markers_point[i_point][0].set_data(np.array([Pt[time,i_point,0]]),np.array([Pt[time,i_point,1]]))
        markers_point[i_point][0].set_3d_properties(np.array([Pt[time, i_point, 2]]))
    return

def updatemulti(time, Pt,  markers_point):
    for i_point in range(134):
        markers_point[0][i_point][0].set_data(np.array([Pt[0][time,i_point,0]]),np.array([Pt[0][time,i_point,1]]))
        markers_point[0][i_point][0].set_3d_properties(np.array([Pt[0][time, i_point, 2]]))
        markers_point[1][i_point][0].set_data(np.array([Pt[1][time, i_point, 0]]), np.array([Pt[1][time, i_point, 1]]))
        markers_point[1][i_point][0].set_3d_properties(np.array([Pt[1][time, i_point, 2]]))
    return

def Integration(nb_frame, Pt_collecte_tab, labels, Masse_centre):
    """

    :param nb_frame:
    :param Pt_collecte_tab:
    :param labels:
    :param Masse_centre:
    :return:
    -------
    -position des points integres
    -forces calculées a chaque frame en chaque point
    -erreurs de position relative entre points integres et points collectes
    -erreurs de position absolues entre points integres et points collectes
    """

    # ---initialisation---#
    Pt_integ = np.zeros((n * m, 3))
    vitesse_calc = np.zeros((n * m, 3))
    accel_calc = np.zeros((n * m, 3))

    Pt_tot = np.zeros((nb_frame, n * m, 3))
    F_all_point = np.zeros((nb_frame, n * m, 3))
    v_all = np.zeros((nb_frame, n * m, 3))

    erreur_relative, erreur_absolue = [], []

    for frame in range(1, nb_frame - 1):
        # initialisation
        Pt_integ = np.zeros((n * m, 3))
        vitesse_calc = np.zeros((n * m, 3))
        accel_calc = np.zeros((n * m, 3))

        # ---A l'etat initial t---#
        Pt_interpolés = interpolation_collecte(Pt_collecte_tab[frame], labels)
        bt1, bt2, btc1, btc2 = Bouts_ressort_collecte(Pt_interpolés, nb_frame)
        # affichage
        # ax = Affichage_points_collecte_t(Pt_interpolés, Pt_ancrage, False , bt1, bt2, btc1, btc2, ind_masse)

        vitesse_avant = Etat_initial(Pt_collecte_tab[frame - 1], Pt_collecte_tab[frame + 1], labels)
        v_all[frame, :, :] = vitesse_avant.T

        # maintenant qu'on a tous les parametres, on peut passer a l'estimation des positions a l'instant 1 en integrant les donnees de l'etat initial 0
        # Integration de l'etat initial
        M, F_spring, F_spring_croix, F_masses = Force_calc(bt1, bt2, btc1, btc2, Masse_centre, 0, nb_frame)
        F_point = Force_point(F_spring, F_spring_croix, F_masses, 0, nb_frame)  # comprend la force des ressorts et le poids

        F_all_point[frame, :, :] = F_point

        for i in range(0, n * m):
            # acceleration
            accel_calc[i, :] = F_point[i, :] / M[i]
            # vitesse
            vitesse_calc[i, :] = dt * accel_calc[i, :] + vitesse_avant.T[i, :]
            # position
            Pt_integ[i, :] = dt * vitesse_calc[i, :] + Pt_interpolés.T[i, :]


        Pt_tot[frame - 1, :, :] = Pt_integ

        # ---erreurs---#

        erreur_relative.append(erreurs(Pt_integ, Pt_collecte_tab[frame + 1])[0])
        erreur_absolue.append(erreurs(Pt_integ, Pt_collecte_tab[frame + 1])[1])

    return Pt_tot, erreur_relative, erreur_absolue, F_all_point, v_all


def Animation(Pt_tot, intervalle_dyna):

    fig = plt.figure()
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.axes.set_xlim3d(left=-2, right=2)
    ax.axes.set_ylim3d(bottom=-2.2, top=2.2)
    ax.axes.set_zlim3d(bottom=-2.5, top=0.5)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    colors_colormap = sns.color_palette(palette="viridis", n_colors=n*m)
    colors = [[] for i in range(n*m)]
    for i in range(n*m):
        col_0 = colors_colormap[i][0]
        col_1 = colors_colormap[i][1]
        col_2 = colors_colormap[i][2]
        colors[i] = (col_0, col_1, col_2)

    ax.set_box_aspect([1.1, 1.8, 1])
    frame_range = intervalle_dyna
    markers_point = [ax.plot(0, 0, 0, '.',color=colors[i]) for i in range(n*m)]

    animate = animation.FuncAnimation(fig, update, frames=frame_range[1] - frame_range[0], fargs=(Pt_tot[:159], markers_point), blit=False)
    output_file_name = 'simulation.mp4'

    animate.save(output_file_name, fps=20, extra_args=['-vcodec', 'libx264'])
    plt.show()

###############################################################################
###############################################################################
###############################################################################

# RÉSULTATS COLLECTE :
participant = 1
statique_name = 'labeled_statique_leftfront_D7'
trial_name  = 'labeled_p1_sauthaut_01'
vide_name = 'labeled_statique_centrefront_vide'

intervalle_dyna = [7000, 7170] #dépend de l'essai (utiliser plateforme_verification_toutesversions pour toruver l'intervalle)
nb_frame = intervalle_dyna[1]-intervalle_dyna[0] - 1 #on exclut la premiere frame
dt = 0.002
dict_fixed_params = longueurs()

#Récupération de tous les points des frames de l'intervalle dynamique
F_totale_collecte, Pt_collecte_tab, labels, ind_masse = Resultat_PF_collecte(participant, statique_name,vide_name,trial_name,intervalle_dyna)

#Récupération des parametres du problemes
k, k_oblique, M, C = Param()
# Pt_ancrage_repos, pos_repos = Points_ancrage_repos(dict_fixed_params)

# Pt_interpoles = interpolation_collecte(Pt_collecte_tab[0], labels)
#
# Pt_integres ,erreur_relative ,erreur_absolue ,force_points, v_all = Integration(nb_frame, Pt_collecte_tab, labels, Masse_centre)
#
# Pt_ancrage_collecte, labels_ancrage = Point_ancrage(Pt_collecte_tab, labels)
# Pt_toile_collecte = Point_toile_init(Pt_collecte_tab, labels)
#
# Pt_ancrage_collecte= Pt_ancrage_collecte[0]
# Pt_toile_collecte = Pt_toile_collecte[0].T


# fig = plt.figure(0)
# ax = fig.add_subplot(111, projection='3d')
# ax.set_box_aspect([1.1, 1.8, 1])
# ax.plot(0, 0, -1.2, 'ow')  # mettre a une echelle lisible et reelle
#
# ax.plot(Pt_ancrage_repos[:, 0], Pt_ancrage_repos[:, 1], Pt_ancrage_repos[:, 2], 'ok', label = 'Point du cadre ')
# ax.plot(pos_repos.T[0, :], pos_repos.T[1, :], pos_repos.T[2, :], 'ob', label='Point de la toile')

#afficher les points d'ancrage et les points de la toile avec des couleurs differentes
# ax.plot(Pt_ancrage_collecte[:, 0], Pt_ancrage_collecte[:, 1], Pt_ancrage_collecte[:, 2], 'ok', label = 'Point du cadre ')
# ax.plot(Pt_toile_collecte[0, :], Pt_toile_collecte[1, :], Pt_toile_collecte[2, :], 'ob', label='Point de la toile')
# ax.plot(Pt_toile_collecte[0, ind_masse], Pt_toile_collecte[1, ind_masse], Pt_toile_collecte[2, ind_masse], 'og', label='Point le plus bas sur l\'intervalle dynamique')
# ax.plot(Pt_ancrage[:, 0], Pt_ancrage[:, 1], Pt_ancrage[:, 2], '+r', label = 'Point du cadre ')
# plt.show()
# Animation(Pt_integres, intervalle_dyna)
Pt_ancrage_repos, pos_repos = Points_ancrage_fix(dict_fixed_params)


Affichage_points_collecte_t(pos_repos.T, Pt_ancrage_repos, True, nb_frame, ind_masse)
plt.show()




all_F_totale_collecte, all_Pt_collecte_tab, all_labels, all_ind_masse  = Resultat_PF_collecte(participant, statique_name,vide_name,trial_name,[0, 7763])
all_Pt_ancrage_collecte, label_ancrage= Point_ancrage(all_Pt_collecte_tab, all_labels)



#calcul acceleration par double diff finie :
#a = (zi+1 + z1-1 - 2zi)/dt**2


axe = 11
# time = np.linspace(0,10,6850)
# time2 = np.linspace(0,10,6848)
#
# z = []
# az = []
# for i in all_Pt_ancrage_collecte:
#     z.append(i[:,2])
#
#
#
# a,b =signal.butter(4, 0.015)
# zfil = signal.filtfilt(a,b,z[:6850], method="gust")
#
# for pos in range(1,len(zfil)-1):
#     az.append(((zfil[pos+1]+zfil[pos-1]-2*zfil[pos])/(dt*dt))*270)
#
#
#
# fig , ax = plt.subplots(2,1)
# fig.suptitle('Position sur Z du point d\'ancrage')
# ax[0].plot(time, z[:6850])
# ax[0].plot(time, zfil, '-r')
# ax[0].set_xlabel('Temps (s)')
# ax[0].set_ylabel('Z (m)')
# ax[1].plot(time2, az)
# ax[1].set_xlabel('Temps (s)')
# ax[1].set_ylabel('accel Z (m.s-2)')
#
#



time = np.linspace(0,10,7763)
x = []
y = []
z = []

for i in all_Pt_ancrage_collecte:
    x.append(i[:, 0])
    y.append(i[:,1])
    z.append(i[:, 2])

fig , axes = plt.subplots(1,3)
fig.suptitle('Position des points d\'ancrage captés')
axes[0].plot(time, x)
axes[0].set_xlabel('Temps (s)')
axes[0].set_ylabel('X (m)')

axes[1].plot(time, y)
axes[1].set_xlabel('Temps (s)')
axes[1].set_ylabel('Y (m)')

axes[2].plot(time, z)
axes[2].set_xlabel('Temps (s)')
axes[2].set_ylabel('Z (m)')


# fig2 = plt.figure(1)
# fig2.suptitle('Position sur Y du point d\'ancrage')
# ax2 = fig2.add_subplot(111)
# ax2.plot(time, y)
# ax2.set_xlabel('Temps (s)')
# ax2.set_ylabel('Y (m)')
#
#
# fig3 = plt.figure(2)
# fig3.suptitle('Position sur X du point d\'ancrage')
# ax3 = fig3.add_subplot(111)
# ax3.plot(time, x)
# ax3.set_xlabel('Temps (s)')
# ax3.set_ylabel('X (m)')

plt.show()

ok = 2


#---2 ANIMATIONS simultatne---#
# fig=plt.figure()
# ax = p3.Axes3D(fig, auto_add_to_figure=False)
# fig.add_axes(ax)
# ax.axes.set_xlim3d(left=-2, right=2)
# ax.axes.set_ylim3d(bottom=-3, top=3)
# ax.axes.set_zlim3d(bottom=-2.5, top=0.5)
# ax.set_xlabel('x (m)')
# ax.set_ylabel('y (m)')
# ax.set_zlabel('z (m)')
#
# colors_colormap = sns.color_palette(palette="viridis", n_colors=n*m)
# colors = [[] for i in range(n*m)]
# for i in range(n*m):
#     col_0 = colors_colormap[i][0]
#     col_1 = colors_colormap[i][1]
#     col_2 = colors_colormap[i][2]
#     colors[i] = (col_0, col_1, col_2)
#
# ax.set_box_aspect([1.1, 1.8, 1])
# frame_range = [7010, 7170]
# # markers_point = [ax.plot(0, 0, 0, '.',color=colors[i]) for i in range(n*m)]
#
# markers_pointsv = [ax.plot(0, 0, 0, '.c') for i in range(n*m)]
# markers_point = [ax.plot(0, 0, 0, '.m') for i in range(n*m)]
#
# marker_tot = [markers_point, markers_pointsv]
#
# animate2=animation.FuncAnimation(fig, updatemulti, frames=frame_range[1] - frame_range[0], fargs=([Pt_tot[:159], Pt_totsv[:159]], marker_tot))
# output_file_name = 'simulation.mp4'
# animate2.save(output_file_name, fps=20, extra_args=['-vcodec', 'libx264'])
# plt.show()