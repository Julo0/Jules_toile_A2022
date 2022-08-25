"""
Modele dynamique avec deux possibilités d'affichage : subplot en plusieurs instants ou enregistrement de video
Les ecarts entre les points du maillage (positions au repos) sont ceux mesurés sur le trampo réel
k issus de la these de Jacques (2008)
On affiche aussi la force, la position, la vitesse et l'accélération du point auquel on met la masse
"""

import numpy as np
# import numpy.matlib
# import sys
#import ipopt
# sys.path.append('/home/user/anaconda3/envs/qpoases/lib')
# sys.path.append('/home/user/anaconda3/envs/qpoases/include/casadi/build/lib')
# import casadi as cas
# from casadi import MX, SX, sqrt
# import biorbd
import matplotlib.pyplot as plt
# from os.path import dirname, join as pjoin
# import scipy.io as sio
# from IPython import embed
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import seaborn as sns

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
Masse_centre=140
ind_masse = 67
#PARAMETRES POUR LA DYNAMIQUE :
dt = 0.002 #fonctionne pour dt<0.004
Nb_increments=10000
T_total=Nb_increments*dt

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
    return (dL, dLmilieu, dl, dlmilieu, l_droite, l_gauche, L_haut, L_bas)

def Param(ind_masse):
    #LONGUEURS AU REPOS
    l_repos = np.zeros(Nb_ressorts_cadre) #on fera des append plus tard, l_repos prend bien en compte tous les ressorts non obliques

    #entre la toile et le cadre :
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

    #dans la toile : on dit que les longueurs au repos sont les memes que en pretension
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


    ##########################################################################
    #RAIDEURS A CHANGER

    # k trouves a partir du programme 5x3:
    k1 = (5 / n) * 3266.68
    k2 = k1 * 2
    k3 = (3 / m) * 3178.4
    k4 = k3 * 2
    k5 = 4 / (n - 1) * 22866.79
    k6 = 2 * k5
    k7 = 2 / (m - 1) * 23308.23
    k8 = 2 * k7
    # k_croix=(k6**2+k8**2)**(1/2)
    k_croix = 3000  # je sais pas

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

    # ressorts obliques dans la toile
    k_croix_tab = k_croix * np.ones(Nb_ressorts_croix)
    k = np.append(k_horizontaux, k_verticaux)
    k = np.append(k_bord, k)


    ##################################################################################################################
    #COEFFICIENTS D'AMORTISSEMENT a changer
    C = 2*np.ones(n*m)

    ##################################################################################################################
    # MASSES (pris en compte la masse ajoutee par lathlete) :
    Mtoile = 7.15
    Mressort = 0.324
    mcoin = Mtoile / (Nb_ressorts_vert + Nb_ressorts_horz) + (
            37 / (n - 1) + 18 / (m - 1)) * Mressort / 4  # masse d'un point se trouvant sur un coin de la toile
    mgrand = 1.5 * Mtoile / (Nb_ressorts_vert + Nb_ressorts_horz) + 37 * Mressort / (
            2 * (n - 1))  # masse d'un point se trouvant sur le grand cote de la toile
    mpetit = 1.5 * Mtoile / (Nb_ressorts_vert + Nb_ressorts_horz) + 18 * Mressort / (
            2 * (m - 1))  # masse d'un point se trouvant sur le petit cote de la toile
    mmilieu = 2 * Mtoile / (
            Nb_ressorts_vert + Nb_ressorts_horz)  # masse d'un point se trouvant au milieu de la toile

    M = mmilieu * np.ones(n * m)  # on initialise toutes les masses a celle du centre
    M[0], M[n - 1], M[n * (m - 1)], M[n * m - 1] = mcoin, mcoin, mcoin, mcoin
    M[n:n * (m - 1):n] = mpetit  # masses du cote bas
    M[2 * n - 1:n * m - 1:n] = mpetit  # masses du cote haut
    M[1:n - 1] = mgrand  # masse du cote droit
    M[n * (m - 1) + 1:n * m - 1] = mgrand  # masse du cote gauche

    if masse_type == 'ponctuelle' :
        M[ind_masse] += Masse_centre
    if masse_type == 'repartie':
        M[ind_masse] += Masse_centre/5
        M[ind_masse - 1] += Masse_centre / 5
        M[ind_masse + 1] += Masse_centre / 5
        M[ind_masse - n] += Masse_centre / 5
        M[ind_masse + n] += Masse_centre / 5


    return k, l_repos, M, k_croix_tab, l_repos_croix,C

def Points_ancrage_repos(Nb_increments):
    # repos :
    Pos_repos = np.zeros((Nb_increments, n * m , 3))

    # on dit que le point numero 0 est a l'origine
    for j in range(m):
        for i in range(n):
            Pos_repos[:, i + j * n] = np.array([-np.sum(dl[:j + 1]), np.sum(dL[:i + 1]), 0])

    # on soustrait la position du point du milieu pour qu'il soit centre sur 0
    Pos_repos_new = np.copy(Pos_repos)
    for j in range(m):
        for i in range(n):
            Pos_repos_new[:, i + j * n] = Pos_repos[:, i + j * n] - Pos_repos[:, 67]

    # ancrage :
    Pt_ancrage = np.zeros((Nb_increments, 2 * (n + m), 3))
    # cote droit :
    for i in range(n):
        Pt_ancrage[:, i, 1:2] = Pos_repos_new[:, i, 1:2]
        Pt_ancrage[:, i, 0] = l_droite
    # cote haut : on fait un truc complique pour center autour de l'axe vertical
    Pt_ancrage[:, n + 4, :] = np.array([0, L_haut, 0])
    for j in range(n, n + 4):
        Pt_ancrage[:, j, :] = Pt_ancrage[:, n + 4, :] + np.array([np.sum(dl[1 + j - n:5]), 0, 0])
    for j in range(n + 5, n + m):
        Pt_ancrage[:, j, :] = Pt_ancrage[:, n + 4, :] - np.array([np.sum(dl[5: j - n + 1]), 0, 0])
    # cote gauche :
    for k in range(n + m, 2 * n + m):
        Pt_ancrage[:, k, 1:2] = - Pos_repos_new[:, k - n - m, 1:2]
        Pt_ancrage[:, k, 0] = -l_gauche
    # cote bas :
    Pt_ancrage[:, 2 * n + m + 4, :] = np.array([0, -L_bas, 0])

    Pt_ancrage[:, 2 * n + m, :] = Pt_ancrage[:, 2 * n + m + 4, :] - np.array([np.sum(dl[5:9]), 0, 0])
    Pt_ancrage[:, 2 * n + m + 1, :] = Pt_ancrage[:, 2 * n + m + 4, :] - np.array([np.sum(dl[5:8]), 0, 0])
    Pt_ancrage[:, 2 * n + m + 2, :] = Pt_ancrage[:, 2 * n + m + 4, :] - np.array([np.sum(dl[5:7]), 0, 0])
    Pt_ancrage[:, 2 * n + m + 3, :] = Pt_ancrage[:, 2 * n + m + 4, :] - np.array([np.sum(dl[5:6]), 0, 0])

    Pt_ancrage[:, 2 * n + m + 5, :] = Pt_ancrage[:, 2 * n + m + 4, :] + np.array([np.sum(dl[4:5]), 0, 0])
    Pt_ancrage[:, 2 * n + m + 6, :] = Pt_ancrage[:, 2 * n + m + 4, :] + np.array([np.sum(dl[3:5]), 0, 0])
    Pt_ancrage[:, 2 * n + m + 7, :] = Pt_ancrage[:, 2 * n + m + 4, :] + np.array([np.sum(dl[2:5]), 0, 0])
    Pt_ancrage[:, 2 * n + m + 8, :] = Pt_ancrage[:, 2 * n + m + 4, :] + np.array([np.sum(dl[1:5]), 0, 0])

    return Pt_ancrage, Pos_repos_new

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
####################################################################################################################
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
        Spring_bout_1[time,i,:] = Pt_ancrage[time,i, :]

    # RESSORTS HORIZONTAUX : il y en a n*(m-1)
    for i in range(Nb_ressorts_horz):
        Spring_bout_1[time,Nb_ressorts_cadre + i,:] = Pt[time,i,:]

    # RESSORTS VERTICAUX : il y en a m*(n-1)
    k=0
    for i in range(n - 1):
        for j in range(m):
            Spring_bout_1[time,Nb_ressorts_cadre+Nb_ressorts_horz+k, :] = Pt[time,i + n * j,:]
            k+=1
####################################################################################################################
    Spring_bout_2=np.zeros((Nb_increments,Nb_ressorts,3))

    # RESSORTS ENTRE LE CADRE ET LA TOILE
    for i in range(0, n): # points droite du bord de la toile
        Spring_bout_2[time,i,:] = Pt[time,i,:]

    k=0
    for i in range(n - 1, m * n, n): # points hauts du bord de la toile
        Spring_bout_2[time, n+k, :] = Pt[time, i, :]
        k+=1

    k=0
    for i in range(m*n-1,n * (m - 1)-1, -1): # points gauche du bord de la toile
        Spring_bout_2[time, n + m + k, :] = Pt[time, i, :]
        k+=1

    k=0
    for i in range(n * (m - 1), -1, -n): # points bas du bord de la toile
        Spring_bout_2[time, 2*n + m + k, :] = Pt[time, i, :]
        k+=1

    # RESSORTS HORIZONTAUX : il y en a n*(m-1)
    k=0
    for i in range(n, n * m):
        Spring_bout_2[time,Nb_ressorts_cadre + k,:] = Pt[time,i,:]
        k+=1

    # RESSORTS VERTICAUX : il y en a m*(n-1)
    k=0
    for i in range(1, n):
        for j in range(m):
            Spring_bout_2[time,Nb_ressorts_cadre + Nb_ressorts_horz + k,:] = Pt[time,i + n * j,:]
            k+=1

    return (Spring_bout_1,Spring_bout_2)

def Spring_bouts_croix(Pt,time,Nb_increments):
    #RESSORTS OBLIQUES : il n'y en a pas entre le cadre et la toile
    Spring_bout_croix_1=np.zeros((Nb_increments,Nb_ressorts_croix,3))

    #Pour spring_bout_1 on prend uniquement les points de droite des ressorts obliques
    k=0
    for i in range ((m-1)*n):
        Spring_bout_croix_1[time,k,:]=Pt[time,i,:]
        k += 1
        #a part le premier et le dernier de chaque colonne, chaque point est relie a deux ressorts obliques
        if (i+1)%n!=0 and i%n!=0 :
            Spring_bout_croix_1[time, k, :] = Pt[time, i, :]
            k+=1

    Spring_bout_croix_2=np.zeros((Nb_increments,Nb_ressorts_croix,3))
    #Pour spring_bout_2 on prend uniquement les points de gauche des ressorts obliques
    #pour chaue carre on commence par le point en haut a gauche, puis en bas a gauche
    #cetait un peu complique mais ca marche, faut pas le changer
    j=1
    k = 0
    while j<m:
        for i in range (j*n,(j+1)*n-2,2):
            Spring_bout_croix_2[time,k,:] = Pt[time,i + 1,:]
            Spring_bout_croix_2[time,k+1,:] = Pt[time,i,:]
            Spring_bout_croix_2[time,k+2,:] = Pt[time,i+ 2,:]
            Spring_bout_croix_2[time,k+3,:] = Pt[time,i + 1,:]
            k += 4
        j+=1

    return Spring_bout_croix_1,Spring_bout_croix_2

def Force_calc(Spring_bout_1,Spring_bout_2,Spring_bout_croix_1,Spring_bout_croix_2,Masse_centre,time,Nb_increments): #force dans chaque ressort
    k, l_repos, M, k_croix_tab, l_repos_croix,C = Param(ind_masse)

    F_spring = np.zeros((Nb_ressorts,3))
    Vect_unit_dir_F = (Spring_bout_2 - Spring_bout_1) / np.linalg.norm(Spring_bout_2 - Spring_bout_1)
    for ispring in range(Nb_ressorts):
        F_spring[ispring,:] = Vect_unit_dir_F[ispring, :] * k[ispring] * (
                np.linalg.norm(Spring_bout_2[ispring, :] - Spring_bout_1[ispring, :]) - l_repos[ispring])

    F_spring_croix = np.zeros((Nb_ressorts_croix,3))
    Vect_unit_dir_F_croix = (Spring_bout_croix_2 - Spring_bout_croix_1) /np.linalg.norm(Spring_bout_croix_2 - Spring_bout_croix_1)
    for ispring in range(Nb_ressorts_croix):
        F_spring_croix[ispring,:] = Vect_unit_dir_F_croix[ispring, :] * k_croix_tab[ispring] * (
                np.linalg.norm(Spring_bout_croix_2[ispring, :] - Spring_bout_croix_1[ispring, :]) - l_repos_croix[ispring])

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
    F_point=F_masses-F_spring_points
    return F_point

def Etat_initial(Pt_ancrage,Pos_repos,Nb_increments,fig) :
    Pt[0, :, :] = Pos_repos[0, :, :]
    # Spring_bout_1, Spring_bout_2 = Spring_bouts(Pt, Pt_ancrage, 1, Nb_increments)
    Spring_bout_1, Spring_bout_2 = Spring_bouts_repos(Pos_repos, Pt_ancrage, 0, Nb_increments)

    # Spring_bout_croix_1, Spring_bout_croix_2 = Spring_bouts_croix(Pt, 1, Nb_increments)
    Spring_bout_croix_1, Spring_bout_croix_2 = Spring_bouts_croix_repos(Pos_repos, 0, Nb_increments)

    Spb1, Spb2 = Spring_bout_1[0, :, :], Spring_bout_2[0, :, :]
    Spbc1, Spbc2 = Spring_bout_croix_1[0, :, :], Spring_bout_croix_2[0, :, :]

    # ax = fig.add_subplot(2, 5, 1, projection='3d')
    ax = fig.add_subplot(2, 5, 1, projection='3d')
    # ax = fig.add_subplot(1,1, 1, projection='3d')
    ax.set_box_aspect([1.1, 1.8, 1])
    ax.plot(Pos_repos[0, :, 0], Pos_repos[0, :, 1], Pos_repos[0, :, 2], '.b')
    ax.plot(Pt_ancrage[0, :, 0], Pt_ancrage[0, :, 1], Pt_ancrage[0, :, 2], '.k')

    for j in range(Nb_ressorts):
        # pqs tres elegant mais cest le seul moyen pour que ca fonctionne
        a = []
        a = np.append(a, Spb1[j, 0])
        a = np.append(a, Spb2[j, 0])

        b = []
        b = np.append(b, Spb1[j, 1])
        b = np.append(b, Spb2[j, 1])

        c = []
        c = np.append(c, Spb1[j, 2])
        c = np.append(c, Spb2[j, 2])

        ax.plot3D(a, b, c, '-r', linewidth=1)

    for j in range(Nb_ressorts_croix):
        # pqs tres elegant mais cest le seul moyen pour que ca fonctionne
        a = []
        a = np.append(a, Spbc1[j, 0])
        a = np.append(a, Spbc2[j, 0])

        b = []
        b = np.append(b, Spbc1[j, 1])
        b = np.append(b, Spbc2[j, 1])

        c = []
        c = np.append(c, Spbc1[j, 2])
        c = np.append(c, Spbc2[j, 2])

        ax.plot3D(a, b, c, '-g', linewidth=1)

    plt.title('temps = ' + str(0))
    ax.axes.set_xlim3d(left=-2, right=2)
    ax.axes.set_ylim3d(bottom=-3, top=3)
    ax.axes.set_zlim3d(bottom=-1000, top=1000)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    #plt.show()
    return Spb1,Spb2,Spbc1,Spbc2

def Affichage(Pt,Pt_ancrage,Spb1,Spb2,Spbc1,Spbc2,time,Nb_increment,fig) :
    # ax = fig.add_subplot(int(T_total / (2 * dt)), int(T_total / (2 * dt)), time + 1, projection='3d')
    ax = fig.add_subplot(2, 5, time //(150) + 1, projection='3d')


    ax.set_box_aspect([1.1, 1.8, 1])
    ax.plot(Pt[time, :, 0], Pt[time, :, 1], Pt[time, :, 2], '.b')
    ax.plot(Pt_ancrage[time, :, 0], Pt_ancrage[time, :, 1], Pt_ancrage[time, :, 2], '.k')

    for j in range(Nb_ressorts):
        a = []
        a = np.append(a, Spb1[j, 0])
        a = np.append(a, Spb2[j, 0])

        b = []
        b = np.append(b, Spb1[j, 1])
        b = np.append(b, Spb2[j, 1])

        c = []
        c = np.append(c, Spb1[j, 2])
        c = np.append(c, Spb2[j, 2])

        ax.plot3D(a, b, c, '-r', linewidth=1)
    #
    # for j in range(Nb_ressorts_croix):
    #     a = []
    #     a = np.append(a, Spbc1[j, 0])
    #     a = np.append(a, Spbc2[j, 0])
    #
    #     b = []
    #     b = np.append(b, Spbc1[j, 1])
    #     b = np.append(b, Spbc2[j, 1])
    #
    #     c = []
    #     c = np.append(c, Spbc1[j, 2])
    #     c = np.append(c, Spbc2[j, 2])
    #
    #     ax.plot3D(a, b, c, '-g', linewidth=1)
    #

    plt.title('temps = ' + str(time*dt) + ' s')
    ax.axes.set_xlim3d(left=-2, right=2)
    ax.axes.set_ylim3d(bottom=-3, top=3)
    ax.axes.set_zlim3d(bottom=-2.1, top=1)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    # plt.show()

def Etat_initial_anim(Pt_ancrage, Pos_repos, Nb_increments):
    # Pt_ancrage, Pos_repos = Points_ancrage_repos(Nb_increments)
    Pt[0, :, :] = Pos_repos[0, :, :]

    Spring_bout_1, Spring_bout_2 = Spring_bouts_repos(Pos_repos, Pt_ancrage, 0, Nb_increments)
    Spring_bout_croix_1, Spring_bout_croix_2 = Spring_bouts_croix_repos(Pos_repos, 0, Nb_increments)

    Spb1, Spb2 = Spring_bout_1[0, :, :], Spring_bout_2[0, :, :]
    Spbc1, Spbc2 = Spring_bout_croix_1[0, :, :], Spring_bout_croix_2[0, :, :]

    return Spb1, Spb2, Spbc1, Spbc2

def update(time, Pt, markers_point):
    for i_point in range(len(markers_point)):
        markers_point[i_point][0].set_data(np.array([Pt[time,i_point,0]]),np.array([Pt[time,i_point,1]]))
        markers_point[i_point][0].set_3d_properties(np.array([Pt[time, i_point, 2]]))
    return

def Anim(Pt,Nb_increments) :
    fig_1 = plt.figure()
    ax = p3.Axes3D(fig_1, auto_add_to_figure=False)
    fig_1.add_axes(ax)
    ax.axes.set_xlim3d(left=-2, right=2)
    ax.axes.set_ylim3d(bottom=-3, top=3)
    ax.axes.set_zlim3d(bottom=-2.5, top=0.5)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    ax.set_box_aspect([1.1, 1.8, 1])
    frame_range = [800, Nb_increments - 1]
    markers_point = [ax.plot(0, 0, 0, '.k') for i in range(m * n)]

    animate = animation.FuncAnimation(fig_1, update, frames=frame_range[1] - frame_range[0], fargs=(Pt, markers_point),
                                      blit=False)
    output_file_name = 'simulation.mp4'
    animate.save(output_file_name, fps=20, extra_args=['-vcodec', 'libx264'])

    plt.show()

###############################################################################
#initialisation :
Pt = np.zeros((Nb_increments, n*m,3))
vitesse = np.zeros((Nb_increments, n*m,3))
accel = np.zeros((Nb_increments, n*m,3))

dL,dLmilieu,dl,dlmilieu,l_droite,l_gauche,L_haut,L_bas = longueurs()
Pt_ancrage, Pos_repos = Points_ancrage_repos(Nb_increments)
Pt_tot=np.zeros((Nb_increments,n*m+Nb_ressorts_cadre,3))
F_point = np.zeros((Nb_increments,n*m,3))
F_totale = np.zeros((Nb_increments,3))

#######################################################################################################################
if affichage == 'subplot' :
    #BOUCLE TEMPORELLE
    fig = plt.figure(0)

    Spb1,Spb2,Spbc1,Spbc2 = Etat_initial(Pt_ancrage,Pos_repos,Nb_increments,fig) #--> actualise Pt[0,:,:] et fait l'affichage

    #ajout de la force d'amortissement visqueux :
    C=0.5*np.ones(n*m)
    Force_amortissement = np.zeros((Nb_increments,n*m,3))
    for time in range(1, Nb_increments):
        M,F_spring, F_spring_croix, F_masses = Force_calc(Spb1,Spb2,Spbc1,Spbc2,Masse_centre, time,Nb_increments) #calcule la force de chaque ressort
        # F_point = Force_point(F_spring, F_spring_croix, F_masses, time, Nb_increments) + Force_amortissement[time]  #calcule la force en chaque point
        F_point[time] = Force_point(F_spring, F_spring_croix, F_masses, time, Nb_increments)
        for index in range (n*m) : #schema d'Euler pour l'integration
            accel[time, index, :] = (F_point[time,index,:] + Force_amortissement[time-1,index,:])  / M[index]
            vitesse[time, index, :] = dt * accel[time, index, :] + vitesse[time - 1, index, :]
            Force_amortissement[time,index,:] = -C[index]*vitesse[time, index, :]
            Pt[time, index, :] = dt * vitesse[time, index, :] + Pt[time - 1, index, :]
            F_totale[time, :] += Force_amortissement[time, index, :] + F_point[time, index, :]

        Spring_bout_1, Spring_bout_2 = Spring_bouts(Pt, Pt_ancrage, time, Nb_increments)
        Spring_bout_croix_1, Spring_bout_croix_2 = Spring_bouts_croix(Pt, time, Nb_increments)
        Spb1, Spb2 = Spring_bout_1[time, :, :], Spring_bout_2[time, :, :]
        Spbc1, Spbc2 = Spring_bout_croix_1[time, :, :], Spring_bout_croix_2[time, :, :]

        if (time)%150==0 and time//150 <= 9:
            Affichage(Pt, Pt_ancrage, Spb1, Spb2, Spbc1, Spbc2, time, Nb_increments, fig)
            print(str(time) + ' / ' + str(Nb_increments))

    x = np.linspace(0, Nb_increments*dt, Nb_increments)
    plt.figure(1)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(x, accel[:, ind_masse, i])
        if  i == 0 : plt.title('accélération du point du milieu pour ' + str(Nb_increments) + ' incréments et C = ' + str(C[ind_masse]))

    plt.figure(2)
    for i in range (3) :
        plt.subplot(3,1,i+1)
        plt.plot(x,vitesse[:,ind_masse,i])
        if i == 0: plt.title('vitesse du point du milieu pour ' + str(Nb_increments) + ' incréments et C = ' + str(C[ind_masse]))

    plt.figure(3)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(x, Pt[:, ind_masse, i])
        if i == 0: plt.title('position du point du milieu pour ' + str(Nb_increments) + ' incréments et C = ' + str(C[ind_masse]))

    plt.figure(4)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(x, F_totale[:, i])
        if i == 0: plt.title(
            'somme des forces en chaque point pour C = ' + str(C[ind_masse]))

    plt.figure(5)
    plt.subplot(2, 1, 1)
    plt.plot(x, Pt[:, ind_masse, i])
    plt.ylabel('z(m)')
    plt.title('Position verticale du point t67 avec une masse de 140kg')
    plt.subplot(2, 1, 2)
    plt.plot(x, F_totale[:, i])
    plt.ylabel('Fz (N)')
    plt.title('Force verticale totale de la toile')
    plt.xlabel('temps (s)')

    plt.show()
#############################################################################################################
if affichage== 'animation' :
    #POUR LANIMATION

    Spb1,Spb2,Spbc1,Spbc2 = Etat_initial_anim(Pt_ancrage,Pos_repos,Nb_increments) #--> actualise Pt[0,:,:] et fait l'affichage
    # Pt_reduit = np.zeros((500, n * m, 3))

    # ajout de la force d'amortissement visqueux :
    C = 100000
    Force_amortissement = np.zeros((Nb_increments, n * m, 3))
    for time in range(1, Nb_increments):
        M, F_spring, F_spring_croix, F_masses = Force_calc(Spb1, Spb2, Spbc1, Spbc2, Masse_centre, time,
                                                           Nb_increments)  # calcule la force de chaque ressort
        F_point = Force_point(F_spring, F_spring_croix, F_masses, time, Nb_increments) + Force_amortissement[
            time]  # calcule la force en chaque point
        for index in range(n * m):  # schema d'Euler pour l'integration
            accel[time, index, :] = F_point[index, :] / M[index]
            vitesse[time, index, :] = dt * accel[time, index, :] + vitesse[time - 1, index, :]
            Force_amortissement[time, index, :] = -C * vitesse[time, index, :]
            Pt[time, index, :] = dt * vitesse[time, index, :] + Pt[time - 1, index, :]
        print(str(time) + ' / ' + str(Nb_increments))

    # for time in range(1, Nb_increments):
    #     M,F_spring, F_spring_croix, F_masses = Force_calc(Spb1,Spb2,Spbc1,Spbc2,Masse_centre, time,Nb_increments)
    #     F_point = Force_point(F_spring, F_spring_croix, F_masses, time, Nb_increments)
    #     for index in range (n*m) :
    #         accel[time, index, :] = F_point[index,:] / M[index]
    #         vitesse[time, index, :] = dt * accel[time, index, :] + vitesse[time - 1, index, :]
    #         Pt[time, index, :] = dt * vitesse[time, index, :] + Pt[time - 1, index, :]

        Spring_bout_1, Spring_bout_2 = Spring_bouts(Pt, Pt_ancrage, time, Nb_increments)
        Spring_bout_croix_1, Spring_bout_croix_2 = Spring_bouts_croix(Pt, time, Nb_increments)
        Spb1, Spb2 = Spring_bout_1[time, :, :], Spring_bout_2[time, :, :]
        Spbc1, Spbc2 = Spring_bout_croix_1[time, :, :], Spring_bout_croix_2[time, :, :]
        print(str(time) + ' / '+ str(Nb_increments))
        for j in range (Nb_ressorts_cadre) :
            Pt_tot[time,j]=Pt_ancrage[time,j]
        for h in range (n*m) :
            Pt_tot[time,Nb_ressorts_cadre + h]=Pt[time,h]

        # if (time)%4==0 :
        #     Pt_reduit[int(time/4),:,:]=Pt[time,:,:]

    # Anim(Pt, Nb_increments)
    fig_1=plt.figure()
    ax = p3.Axes3D(fig_1, auto_add_to_figure=False)
    fig_1.add_axes(ax)
    ax.axes.set_xlim3d(left=-2, right=2)
    ax.axes.set_ylim3d(bottom=-3, top=3)
    ax.axes.set_zlim3d(bottom=-2.5, top=0.5)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    colors_colormap = sns.color_palette(palette="viridis", n_colors=n*m + Nb_ressorts_cadre)
    colors = [[] for i in range(n*m+Nb_ressorts_cadre)]
    for i in range(n*m + Nb_ressorts_cadre):
        col_0 = colors_colormap[i][0]
        col_1 = colors_colormap[i][1]
        col_2 = colors_colormap[i][2]
        colors[i] = (col_0, col_1, col_2)

    ax.set_box_aspect([1.1, 1.8, 1])
    frame_range = [0, Nb_increments - 1]
    markers_point = [ax.plot(0, 0, 0, '.',color=colors[i]) for i in range(Nb_ressorts_cadre + n*m)]


    animate=animation.FuncAnimation(fig_1, update, frames=frame_range[1] - frame_range[0], fargs=(Pt_tot, markers_point), blit=False)
    output_file_name = 'simulation.mp4'
    animate.save(output_file_name, fps=20, extra_args=['-vcodec', 'libx264'])

    plt.show()

