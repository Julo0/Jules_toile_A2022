"""
Calcule aussi les matrices e rotation des plateformes
"""



import numpy as np
from ezc3d import c3d
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import seaborn as sns
import scipy as sc
from scipy import signal

#PARAMETRES :
masses=[64.5,87.2]
Nb_disque=['vide','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11']
Poids_disque=[0,197.181,394.362,592.524, 789.705,937.836,1135.017,1333.179,1531.341,1728.522,1926.684, 2174.877]
for i in range (1,len(Poids_disque)) :
    Poids_disque[i]+= 67.689 #oups j'avais oublie le poids de la plaque

#ACTION :
action = 'dynamique'  #'raw' #'calibrage' #'dynamique' #'soustraction' ######'comparaison' #'acceleration' #named_vs_raw
matrice_de_rotation = 1#doit etre nul si on est en calibrage
bonne_orientation=1 #si 1, alors on met les x et les y dans le bon sens (fonction plateforme_calcul)
if action == 'calibrage' :
    matrice_de_rotation = 0
    bonne_orientation = 0

#FICHIERS :
numero_disque = 7
participant=1 #doit être cohérent avec trial_name

vide_name='labeled_statique_centrefront_vide'
trial_name = 'labeled_p1_sauthaut_01'
statique_name='labeled_statique_leftfront_' + Nb_disque[numero_disque]
calibrage_name = 'Test_Plateformes_4empilees05'
raw_name = 'labeled_statique_leftfront_' + Nb_disque[numero_disque]


def open_c3d(participant, trial_name):
    dossiers = ['statique', 'participant_01', 'participant_02']
    if participant == 4 :
        file_path = '/home/lim/Documents/Thea_final/Collecte/c3d_files/'
    else :
        file_path = '/home/lim/Documents/Thea_final/Collecte/c3d_files/' + dossiers[participant]
    c3d_file = c3d(file_path + '/' + trial_name + '.c3d')
    return c3d_file

def Param_platform(c3d_experimental) :
    # parametres des plateformes
    Nb_platform=c3d_experimental['parameters']['FORCE_PLATFORM']['USED']['value'][0]
    platform_type=c3d_experimental['parameters']['FORCE_PLATFORM']['TYPE']['value']
    platform_zero=c3d_experimental['parameters']['FORCE_PLATFORM']['ZERO']['value'] #pas normal ca doit etre des valeurs non nulles
    platform_corners=c3d_experimental['parameters']['FORCE_PLATFORM']['CORNERS']['value'] #position des coins de chaeu plateforme : [xyz,quel coin,quelle plateforme] --> pb plateforme 1
    platform_origin=c3d_experimental['parameters']['FORCE_PLATFORM']['ORIGIN']['value']
    platform_channel=c3d_experimental['parameters']['FORCE_PLATFORM']['CHANNEL']['value'] #dit quel chanel contient quelle donnee
    platform_calmatrix=c3d_experimental['parameters']['FORCE_PLATFORM']['CAL_MATRIX']['value'] #matrice de calibration : il n'y en a pas
    return (Nb_platform,platform_type,platform_zero,platform_corners,platform_origin,platform_channel,platform_calmatrix)

def matrices() :

    #M1,M2,M4 sont les matrices obtenues apres la calibration sur la plateforme 3

    M4_new = [[5.4526,    0.1216  ,  0.0937 ,  -0.0001  , -0.0002  ,  0.0001],
              [0.4785 ,   5.7700  ,  0.0178  ,  0.0001  ,  0.0001  ,  0.0001],
              [-0.1496 ,  -0.1084  , 24.6172  ,  0.0000 ,  -0.0000  ,  0.0002],
              [12.1726 ,-504.1483  ,-24.9599  ,  3.0468  ,  0.0222  ,  0.0042],
              [475.4033,   10.6904 ,  -4.2437  , -0.0008 ,   3.0510 ,   0.0066],
              [-6.1370   , 4.3463  , -4.6699  , -0.0050  ,  0.0038   , 1.4944]]

    M1_new = [[2.4752,    0.1407,    0.0170  , -0.0000 ,  -0.0001  ,  0.0001],
              [0.3011,    2.6737  , -0.0307 ,   0.0000 ,   0.0001  ,  0.0000],
              [0.5321  ,  0.3136 ,  11.5012  , -0.0000 ,  -0.0002   , 0.0011],
              [20.7501 ,-466.7832,   -8.4437 ,   1.2666 ,  -0.0026  ,  0.0359],
              [459.7415,    9.3886 ,  -4.1276 ,  -0.0049 ,   1.2787 ,  -0.0057],
              [265.6717 , 303.7544 , -45.4375  , -0.0505 ,  -0.1338 ,   0.8252]]

    M2_new = [[2.9967,   -0.0382  ,  0.0294 ,  -0.0000  ,  0.0000 ,  -0.0000],
                [-0.1039 ,   3.0104 ,  -0.0324  , -0.0000 ,  -0.0000  ,  0.0000],
                [-0.0847 ,  -0.0177 ,  11.4614  , -0.0000  , -0.0000  , -0.0000],
                [13.6128 , 260.5267 ,  17.9746  ,  1.2413  ,  0.0029  ,  0.0158],
                [-245.7452 ,  -7.5971,   11.5052,    0.0255,    1.2505 ,  -0.0119],
                [-10.3828 ,  -0.9917  , 15.3484 ,  -0.0063  , -0.0030  ,  0.5928]]

    M3_new = [[2.8992,0,0,0,0,0],
              [0,2.9086,0,0,0,0],
              [0,0,11.4256,0,0,0],
              [0,0,0,1.2571,0,0],
              [0,0,0,0,1.2571,0],
              [0,0,0,0,0,0.5791]]

    # zeros donnes par Nexus
    zeros1 = np.array([1.0751899, 2.4828501, -0.1168980, 6.8177500, -3.0313399, -0.9456340])
    zeros2 = np.array([0., -2., -2., 0., 0., 0.])
    zeros3 = np.array([0.0307411, -5., -4., -0.0093422, -0.0079338, 0.0058189])
    zeros4 = np.array([-0.1032560, -3., -3., 0.2141770, 0.5169040, -0.3714130])

    return M1_new, M2_new, M3_new, M4_new, zeros1, zeros2, zeros3, zeros4

def matrices_rotation():
    #nouvelle position des PF
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

def plateformes_separees_rawpins(c3d) :
    #on garde seulement les Raw pins
    force_labels=c3d['parameters']['ANALOG']['LABELS']['value']
    ind=[]
    for i in range (len(force_labels)) :
        if 'Raw' in force_labels[i] :
            ind=np.append(ind,i)
    # ind_stop=int(ind[0])
    indices = np.array([int(ind[i]) for i in range(len(ind))])
    ana = c3d['data']['analogs'][0, indices, :]
    platform1 = ana[0:6, :]  # pins 10 a 15
    platform2 = ana[6:12, :]  # pins 19 a 24
    platform3 = ana[12:18, :]  # pins 28 a 33
    platform4 = ana[18:24, :]  # pins 1 a 6

    platform = np.array([platform1, platform2, platform3, platform4])
    return platform

def plateformes_separees_namedpins(c3d) :
    force_labels = c3d['parameters']['ANALOG']['LABELS']['value']
    ind = []
    for i in range(len(force_labels)):
        if 'Force' in force_labels[i]:
            # ind_stop = int(i)
            ind = np.append(ind, i)
        if 'Moment' in force_labels[i]:
            ind = np.append(ind, i)
    indices = np.array([int(ind[i]) for i in range(len(ind))])
    ana = c3d['data']['analogs'][0, indices, :]

    platform1 = ana[0:6, :]  # pins 10 a 15
    platform2 = ana[6:12, :]  # pins 19 a 24
    platform3 = ana[12:18, :]  # pins 28 a 33
    platform4 = ana[18:24, :]  # pins 1 a 6

    platform = np.array([platform1, platform2, platform3, platform4])
    return platform

def soustraction_zero(platform) : #soustrait juste la valeur du debut aux raw values
    longueur = np.size(platform[0, 0])
    zero_variable = np.zeros((4, 6))
    for i in range(6):
        for j in range(4):
            zero_variable[j, i] = np.mean(platform[j, i, 0:100])
            platform[j, i, :] = platform[j, i, :] - zero_variable[j, i] * np.ones(longueur)
    return platform

def plateforme_calcul (platform) : #prend les plateformes separees, passe les N en Nmm, calibre, multiplie par mat rotation, met dans la bonne orientation
    # ind_stop = Named_pins(c3d)
    # ana = c3d['data']['analogs'][0, ind_stop:, :]
    M1, M2, M3, M4, zeros1, zeros2, zeros3, zeros4 = matrices()
    # rot43,rot42,rot41 = matrices_rotation()
    rot31, rot34, rot32 = matrices_rotation()

# N--> Nmm
    platform[:, 3:6] = platform[:, 3:6] * 1000

#calibration
    platform[0] = np.matmul(M1, platform[0]) * 100
    platform[1] = np.matmul(M2, platform[1]) * 200
    platform[2] = np.matmul(M3, platform[2]) * 100
    platform[3] = np.matmul(M4, platform[3]) * 25

#matrices de rotation ; ancienne position des PF
    # platform[0,0:2] = np.matmul(rot41, platform[0,0:2])
    # platform[1,0:2] = np.matmul(rot42,platform[1,0:2])
    # platform[2,0:2] = np.matmul(rot43, platform[2,0:2])
#matrices de rotation ; nouvelle position des PF
    platform[0,0:2] = np.matmul(rot31, platform[0,0:2])
    platform[1,0:2] = np.matmul(rot32,platform[1,0:2])
    platform[3,0:2] = np.matmul(rot34, platform[3,0:2])

    # bonne orientation ; nouvelle position des PF (pas sure si avant ou apres)
    platform[0, 1] = -platform[0, 1]
    platform[1, 0] = -platform[1, 0]
    platform[2, 0] = -platform[2, 0]
    platform[3, 1] = -platform[3, 1]

#bonne orientation - ancienne position des PF
    # platform[0,0] = -platform[0,0]
    # platform[1,1] = -platform[1,1]
    # platform[2,1] = -platform[2,1]
    # platform[3,0] = -platform[3,0]
# # bonne orientation - nouvelle position des PF
#     platform[0, 1] = -platform[0, 1]
#     platform[1, 0] = -platform[1, 0]
#     platform[2, 0] = -platform[2, 0]
#     platform[3, 1] = -platform[3, 1]

    # # prendre un point sur 4 pour avoir la même fréquence que les caméras
    platform_new = np.zeros((4, 6, int(np.shape(platform)[2] / 4)))
    for i in range(np.shape(platform)[2]):
        if i % 4 == 0:
            platform_new[:, :, i // 4] = platform[:, :, i]


    return platform_new

def soustraction_rawpin (c3d_statique,c3d_vide) : #pour les forces calculees par Vicon
    platform_statique = plateformes_separees_rawpins(c3d_statique)
    platform_vide = plateformes_separees_rawpins(c3d_vide)
    platform = np.copy(platform_statique)

#on soustrait les valeurs du fichier a vide
    for j in range (6) :
        for i in range (4) :
            platform[i, j, :] = platform_statique[i, j, :] - np.mean(platform_vide[i, j,:])
    platform = plateforme_calcul(platform)
    return platform

def soustraction_namedpin (c3d_statique,c3d_vide) : #pour les forces calculees avec matrice de calib etc
    # on garde seulement les pins de Vicon
    platform_statique=plateformes_separees_namedpins(c3d_statique)
    platform_vide = plateformes_separees_namedpins(c3d_vide)
    platform = np.copy(platform_statique)

    for j in range (6) :
        for i in range (4) :
            # platform_statique[i,j,:]=platform_statique[i,j,:] - np.mean(platform_vide[i,j])
            platform[i, j, :] = platform_statique[i, j, :] - np.mean(platform_vide[i, j,:])

    return platform

def comparaison (c3d_statique,c3d_vide) : #on ne soustrait pas les valeur a vide
    platform_statique = plateformes_separees_rawpins(c3d_statique)
    platform_statique = plateforme_calcul(platform_statique)

    platform_vide = plateformes_separees_rawpins(c3d_statique)
    platform_vide = plateforme_calcul(platform_vide)
    return platform_statique, platform_vide

def dynamique(c3d_experimental) :
    platform = plateformes_separees_rawpins(c3d_experimental)
    platform = soustraction_zero(platform)
    platform = plateforme_calcul(platform)
    return platform

def calibrage_rawpin(c3d_calibrage) :
    platform = plateformes_separees_rawpins(c3d_calibrage)
    platform = soustraction_zero(platform)

    #multiplication des moments par 1000 pour passer en Nmm
    platform[:, 3:6] = platform[:, 3:6] * 1000

    # multiplication par les matrices de calibration et le gain
    M1, M2, M3, M4, zeros1, zeros2, zeros3, zeros4 = matrices()
    platform[0] = -np.matmul(M1, platform[0]) * 100
    platform[1] = -np.matmul(M2, platform[1])*200
    platform[2]= -np.matmul(M3, platform[2])*100
    platform[3] = -np.matmul(M4, platform[3])*25

    return platform

def calibrage_namedpin(c3d_calibrage) :
    platform = plateformes_separees_namedpins(c3d_calibrage)
    platform = soustraction_zero(platform)

    # multiplication par les matrices de calibration et le gain
    M1, M2, M3, M4, zeros1, zeros2, zeros3, zeros4 = matrices()
    platform[0] = -np.matmul(M1, platform[0]) * 100 * 10
    platform[1] = -np.matmul(M2, platform[1]) * 200 * 10
    platform[2] = -np.matmul(M3, platform[2]) * 100 * 10
    platform[3] = -np.matmul(M4, platform[3]) * 25 * 10

    return platform

def unlabeled(c3d_experimental) :
    #boucle pour supprimer les points non labeled
    labels=c3d_experimental['parameters']['POINT']['LABELS']['value']
    indices_supp=[]
    for i in range (len(labels)) :
        if '*' in labels[i] :
            indices_supp=np.append(indices_supp,i)
    ind_stop=int(indices_supp[0])
    #labels et points avec les points non labelles supprimes
    labels=c3d_experimental['parameters']['POINT']['LABELS']['value'][0:ind_stop]
    points=c3d_experimental['data']['points'][:3,:ind_stop,:]
    return ind_stop,labels,points

def acceleration_point_bas(ind_stop,labels,points) : #trouver le point qui descend le plus bas :

    #on cherche le z min de chaque marqueur (on en profite pour supprimer les nan)
    minimum_marqueur=[np.nanmin(points[2,i]) for i in range (ind_stop)]
    argmin_temps=[np.where((points[2,i])==minimum_marqueur[i]) for i in range (ind_stop)]

    #on cherche le z min total
    argmin_marqueur=np.argmin(minimum_marqueur)
    argmin=[argmin_marqueur,int(argmin_temps[argmin_marqueur][0])]
    label_min=labels[argmin[0]]

    print('Altitude minimale obtenue au marqueur ' + str(label_min) + ' au temps t=' + str(argmin[1]))

    point_bas = points[:, argmin[0]]*0.001 #on veut les coordonnees en metres !!
    T = np.size(points[0, 0])
    dt=0.002
    vitesse_point_bas=np.array([[(point_bas[j, i + 1] - point_bas[j, i])/dt for i in range(0, T - 1)] for j in range(3)])
    accel_point_bas=np.array([[(vitesse_point_bas[j, i + 1] - vitesse_point_bas[j, i])/dt for i in range(0, T - 2)] for j in range(3)])
    # accel_point_bas = np.array([[(point_bas[j, i + 2] - 2 * point_bas[j, i + 1] + point_bas[j, i])/dt**2 for i in range(0, T - 2)] for j in range(3)])

    return label_min, point_bas, vitesse_point_bas, accel_point_bas, T


fig = plt.figure()
fig.add_subplot(313)

if action == 'raw' :
    c3d = open_c3d(participant,raw_name)
    platform= plateformes_separees_rawpins(c3d)
    longueur = np.size(platform[0, 0])
    x = np.linspace(0, longueur, longueur)
    labels = ['output de Fx (V)', 'output de Fy (V)', 'output de Fz (V)']
    for i in range (3):
        plt.subplot(3,1,i+1)
        plt.plot(x, platform[0, i, :], '-k', linewidth=0.99, label='PF1')
        plt.plot(x, platform[1, i, :], '-b', linewidth=0.99, label='PF2')
        plt.plot(x, platform[2, i, :], '-r', linewidth=0.99, label='PF3')
        plt.plot(x, platform[3, i, :], '-m', linewidth=0.99, label='PF4')
        plt.ylabel(labels[i])
        if i ==2 :
            plt.xlabel('instant')
        if i==0 :
            plt.title('Forces des plateformes avant modifications pour l\'essai statique ' +str(raw_name))
        plt.legend()


if action == 'named_vs_raw' :
    c3d = open_c3d(participant,trial_name)
    platform_named = plateformes_separees_namedpins(c3d) #calcul fait par Vicon
    platform_named = soustraction_zero(platform_named)

    platform_raw = plateformes_separees_rawpins(c3d)  # calcul a la main avec la matrice
    platform_raw=soustraction_zero(platform_raw)
    platform_raw=plateforme_calcul(platform_raw)

    longueur = np.size(platform_raw[0,0])
    x = np.linspace(0, longueur, longueur)
    legende = ['Fx (N)','Fy (N)','Fz (N)','Mx (Nmm)','My (Nmm)','Mz (Nmm)']
    for j in range (4) :
        plt.figure(j+1)
        for i in range (6) :
            plt.subplot(6,1,i+1)
            plt.plot(x, platform_raw[j, i], '-k', label='PF' + str(j+1) + ' a la main')
            plt.plot(x, -platform_named[j, i], '0.5', label='PF' + str(j+1) + ' vicon')
            if i == 0 : plt.title('comparaison Vicon et Calcul perso pour plateforme ' + str(j+1) + ' pour lessai ' + str(trial_name))
            plt.legend()
            plt.ylabel(legende[i])
            # plt.title('Essai statique ' + str(statique_name) + ' compare avec l\'essai a vide ' + str(vide_name))

if action == 'soustraction' :
    c3d_vide = open_c3d(0, vide_name)
    c3d_statique = open_c3d(0, statique_name)
    platform = soustraction_rawpin (c3d_statique,c3d_vide) #plateforme statique a laquelle on a soustrait la valeur de la plateforme a vide
    longueur = np.size(platform[0,0])
    x = np.linspace(0, longueur, longueur)

    # plt.figure(0)
    plt.subplot(311)
    plt.plot(x, platform[0,0, :], '-k', label='Fx PF1')
    plt.plot(x, platform[1,0, :], '-b', label='Fx PF2')
    plt.plot(x, platform[2,0, :], '-r', label='Fx PF3')
    plt.plot(x, platform[3,0, :], '-m', label='Fx PF4')
    plt.legend()
    plt.ylabel('Fx (N)')
    plt.title('Forces calibrées pour l\'essai statique ' + str(statique_name))

    plt.subplot(312)
    plt.plot(x, platform[0,1, :], '-k', label='Fy PF1')
    plt.plot(x, platform[1,1, :], '-b', label='Fy PF2')
    plt.plot(x, platform[2,1, :], '-r', label='Fy PF3')
    plt.plot(x, platform[3,1, :], '-m', label='Fy PF4')
    plt.ylabel('Fy (N)')
    plt.legend()

    plt.subplot(313)
    plt.plot(x, platform[0,2, :], '-k', label='Fz PF1')
    plt.plot(x, platform[1,2, :], '-b', label='Fz PF2')
    plt.plot(x, platform[2,2, :], '-r', label='Fz PF3')
    plt.plot(x, platform[3,2, :], '-m', label='Fz PF4')
    plt.ylabel('Fz (N)')
    plt.xlabel('instant')
    plt.plot(x,(Poids_disque[numero_disque] )*np.ones(longueur), '-g',label='Poids applique a la toile')
    plt.plot(platform[0,2, :]+platform[1,2, :]+platform[2,2, :]+platform[3,2, :],'-y',label = 'Somme des 4 PF')
    plt.legend()

    print(np.mean(platform[0,2, :]+platform[1,2, :]+platform[2,2, :]+platform[3,2, :]))
    print(Poids_disque[numero_disque])
    print(Poids_disque[numero_disque]/(np.mean(platform[0,2, :]+platform[1,2, :]+platform[2,2, :]+platform[3,2, :])))
    print('pourcentage d erreur : ' + str((Poids_disque[numero_disque] - (np.mean(platform[0,2, :]+platform[1,2, :]+platform[2,2, :]+platform[3,2, :])))*100/Poids_disque[numero_disque]))

if action == 'comparaison' :
    c3d_vide = open_c3d(0, vide_name)
    c3d_statique = open_c3d(0, statique_name)
    platform_statique,platform_vide = comparaison(c3d_statique,c3d_vide)
    longueur = np.minimum(np.size(platform_statique[0, 0]),np.size(platform_vide[0, 0]))
    x = np.linspace(0, longueur, longueur)

    plt.subplot(311)
    plt.plot(x, platform_statique[0,0, :longueur], '-k', label='Fx PF1')
    plt.plot(x, platform_statique[1,0, :longueur], '-b', label='Fx PF2')
    plt.plot(x, platform_statique[2,0, :longueur], '-r', label='Fx PF3')
    plt.plot(x, platform_statique[3,0, :longueur], '-m', label='Fx PF4')
    plt.legend()
    plt.title('Essai statique ' + str(statique_name) + ' compare avec l\'essai a vide ' + str(vide_name))

    plt.subplot(312)
    plt.plot(x, platform_statique[0,1, :longueur], '-k', label='Fy PF1')
    plt.plot(x, platform_statique[1,1, :longueur], '-b', label='Fy PF2')
    plt.plot(x, platform_statique[2,1, :longueur], '-r', label='Fy PF3')
    plt.plot(x, platform_statique[3,1, :longueur], '-m', label='Fy PF4')
    plt.legend()

    plt.subplot(313)
    plt.plot(x, platform_statique[0,2, :longueur], '-k', label='Fz PF1')
    plt.plot(x, platform_statique[1,2, :longueur], '-b', label='Fz PF2')
    plt.plot(x, platform_statique[2,2, :longueur], '-r', label='Fz PF3')
    plt.plot(x, platform_statique[3,2, :longueur], '-m', label='Fz PF4')
    plt.plot(x, (Poids_disque[numero_disque]/4)*np.ones(longueur), '-g', label='Poids theorique divise par 4 pour ' + str(numero_disque) + ' disques')
    plt.legend()

if action == 'dynamique' :
    c3d_experimental = open_c3d(participant, trial_name)
    platform = dynamique(c3d_experimental)
    longueur = np.size(platform[0, 0])
    x = np.linspace(0, longueur, longueur)

    masse = masses[participant-1]
    poids = masse * 9.81

    plt.subplot(311)
    plt.plot(x, platform[0, 0, :], '-k', linewidth = 1, label='PF1')
    plt.plot(x, platform[1, 0, :], '-b',linewidth = 1, label='PF2')
    plt.plot(x, platform[2, 0, :], '-r',linewidth = 1, label='PF3')
    plt.plot(x, platform[3, 0, :], '-m',linewidth = 1, label='PF4')
    plt.legend()
    plt.ylabel('Fx (N)')
    plt.title('Forces calibrées pour l\'essai dynamique ' + str(trial_name))

    plt.subplot(312)
    plt.plot(x, platform[0, 1, :], '-k',linewidth = 1, label='PF1')
    plt.plot(x, platform[1, 1, :], '-b',linewidth = 1, label='PF2')
    plt.plot(x, platform[2, 1, :], '-r',linewidth = 1, label='PF3')
    plt.plot(x, platform[3, 1, :], '-m',linewidth = 1, label='PF4')
    plt.ylabel('Fy (N)')
    plt.legend()


    plt.subplot(313)
    plt.plot(x, platform[0, 2, :], '-k',linewidth = 1, label='PF1')
    plt.plot(x, platform[1, 2, :], '-b',linewidth = 1, label='PF2')
    plt.plot(x, platform[2, 2, :], '-r',linewidth = 1, label='PF3')
    plt.plot(x, platform[3, 2, :], '-m',linewidth = 1, label='PF4')
    plt.ylabel('Fz (N)')
    plt.xlabel('instant')
    if 'statique' in trial_name or 'pieds' in trial_name :
        plt.plot(platform[0,2] + platform[1,2] + platform[2,2] + platform[3,2],'-g',label = 'somme des forces')
        plt.plot(masse*9.81*np.ones(longueur), '-y', label='Poids du participant')
        print('Difference entre poids theorique et poids obtenu = ' + str(round(masse*9.81 - np.mean(platform[0,2,10000:35000] + platform[1,2,10000:35000] + platform[2,2,10000:35000] + platform[3,2,10000:35000]),3)) + ' N = ' + str(round(masse- np.mean(sum(platform[:, 2,10000:35000]))/9.81,3)) + ' kg')
    plt.legend()

if action == 'calibrage' :
    c3d_calibrage = open_c3d(4, calibrage_name)
    platform=calibrage_rawpin(c3d_calibrage)
    longueur = np.size(platform[0, 0])
    x = np.linspace(0, longueur, longueur)

    plt.figure(1)
    plt.subplot(611)
    plt.plot(x, platform[0, 0, :], '-k', linewidth = 1, label='PF1')
    plt.plot(x, platform[1, 0, :], '-b', linewidth = 1, label='PF2')
    plt.plot(x, platform[2, 0, :], '-r', linewidth = 1, label='PF3')
    plt.plot(x, platform[3, 0, :], '-m', linewidth = 1, label='PF4')
    plt.tick_params(labelbottom=False, bottom=False)
    plt.ylabel('Fx (Nm)')
    # plt.legend()
    plt.title('Forces du fichier de calibrage, calibrées avec les matrices optimisées : ' + str(calibrage_name))

    plt.subplot(612)
    plt.plot(x, platform[0, 1, :], '-k', linewidth = 1, label=' PF1')
    plt.plot(x, platform[1, 1, :], '-b', linewidth = 1, label='PF2')
    plt.plot(x, platform[2, 1, :], '-r', linewidth = 1, label='PF3')
    plt.plot(x, platform[3, 1, :], '-m', linewidth = 1, label='PF4')
    plt.tick_params(labelbottom=False, bottom=False)
    plt.ylabel('Fy (Nm)')
    # plt.legend()

    plt.subplot(613)
    plt.plot(x, platform[0, 2, :], '-k', linewidth = 1, label='PF1')
    plt.plot(x, platform[1, 2, :], '-b', linewidth = 1, label='PF2')
    plt.plot(x, platform[2, 2, :], '-r', linewidth = 1, label='PF3')
    plt.plot(x, platform[3, 2, :], '-m', linewidth = 1, label='PF4')
    plt.tick_params(labelbottom=False, bottom=False)
    plt.ylabel('Fz (Nm)')
    # plt.legend()

    plt.subplot(614)
    plt.plot(x, platform[0, 3, :]/1000, '-k', linewidth = 1, label='PF1')
    plt.plot(x, platform[1, 3, :]/1000, '-b', linewidth = 1, label='PF2')
    plt.plot(x, platform[2, 3, :]/1000, '-r', linewidth = 1, label='PF3')
    plt.plot(x, platform[3, 3, :]/1000, '-m', linewidth = 1, label='PF4')
    plt.tick_params(labelbottom=False, bottom=False)
    plt.ylabel('Mx (Nm)')
    # plt.legend()

    plt.subplot(615)
    plt.plot(x, platform[0, 4, :]/1000, '-k', linewidth = 1, label=' PF1')
    plt.plot(x, platform[1, 4, :]/1000, '-b', linewidth = 1, label='PF2')
    plt.plot(x, platform[2, 4, :]/1000, '-r', linewidth = 1, label='PF3')
    plt.plot(x, platform[3, 4, :]/1000, '-m', linewidth = 1, label='PF4')
    plt.tick_params(labelbottom=False, bottom=False)
    plt.ylabel('My (Nm)')
    # plt.legend()

    plt.subplot(616)
    plt.plot(x, platform[0, 5, :]/1000, '-k', linewidth = 1, label='PF1')
    plt.plot(x, platform[1, 5, :]/1000, '-b', linewidth = 1, label='PF2')
    plt.plot(x, platform[2, 5, :]/1000, '-r', linewidth = 1, label='PF3')
    plt.plot(x, platform[3, 5, :]/1000, '-m', linewidth = 1, label='PF4')
    plt.ylabel('Mz (Nm)')
    plt.xlabel('instant')
    plt.legend()

    #differences

    plt.figure(2)
    plt.subplot(611)
    plt.plot(x, platform[2, 0, :]-platform[0, 0, :], '-k', label='PF3-PF1')
    plt.plot(x, platform[2, 0, :]-platform[1, 0, :], '-b', label='PF3-PF2')
    plt.plot(x, platform[2, 0, :]-platform[3, 0, :], '-r', label='PF3-PF4')
    plt.legend()
    plt.title('Difference entre PF3 et les autres')

    plt.subplot(612)
    plt.plot(x, platform[2, 1, :] - platform[0, 1, :], '-k', label='PF3-PF1')
    plt.plot(x, platform[2, 1, :] - platform[1, 1, :], '-b', label='PF3-PF2')
    plt.plot(x, platform[2, 1, :] - platform[3, 1, :], '-r', label='PF3-PF4')
    plt.legend()

    plt.subplot(613)
    plt.plot(x, platform[2, 2, :] - platform[0, 2, :], '-k', label='PF3-PF1')
    plt.plot(x, platform[2, 2, :] - platform[1, 2, :], '-b', label='PF3-PF2')
    plt.plot(x, platform[2, 2, :] - platform[3, 2, :], '-r', label='PF3-PF4')
    plt.legend()

    plt.subplot(614)
    plt.plot(x, platform[2, 3, :] - platform[0, 3, :], '-k', label='PF3-PF1')
    plt.plot(x, platform[2, 3, :] - platform[1, 3, :], '-b', label='PF3-PF2')
    plt.plot(x, platform[2, 3, :] - platform[3, 3, :], '-r', label='PF3-PF4')
    plt.legend()

    plt.subplot(615)
    plt.plot(x, platform[2, 4, :] - platform[0, 4, :], '-k', label='PF3-PF1')
    plt.plot(x, platform[2, 4, :] - platform[1, 4, :], '-b', label='PF3-PF2')
    plt.plot(x, platform[2, 4, :] - platform[3, 4, :], '-r', label='PF3-PF4')
    plt.legend()

    plt.subplot(616)
    plt.plot(x, platform[2, 5, :] - platform[0, 5, :], '-k', label='PF3-PF1')
    plt.plot(x, platform[2, 5, :] - platform[1, 5, :], '-b', label='PF3-PF2')
    plt.plot(x, platform[2, 5, :] - platform[3, 5, :], '-r', label='PF3-PF4')
    plt.legend()

if action == 'acceleration' :
    c3d_experimental= open_c3d(participant, trial_name)
    rapport = 2.92
    masse=masses[participant-1]

    ind_stop, labels, points = unlabeled(c3d_experimental)
    label_min, point_bas, vitesse_point_bas, accel_point_bas, T = acceleration_point_bas(ind_stop, labels, points)

    platform = dynamique(c3d_experimental)

    x = np.linspace(0, T - 2, T - 2)

    plt.subplot(211)
    plt.plot(x, masse * accel_point_bas[2], '-c')
    plt.title('Acceleration verticale du point ' + str(label_min) + ' au cours de l\'essai ' + str(trial_name))

    plt.subplot(212)
    #on filtre l'accceleration :
    sos = signal.butter(3, [1,10], 'bandpass', fs=500, output='sos')
    accel_point_bas = signal.sosfilt(sos, accel_point_bas)
    plt.plot(x, masse*accel_point_bas[2], '-y', label='masse fois acceleration du point qui descend le plus bas, filtré')
    plt.plot(x,
             [(platform[0, 2, i * 4] + platform[1, 2, i * 4] + platform[2, 2, i * 4] + platform[3, 2, i * 4]) * rapport
              for i in range(T - 2)], '-g', label='somme des Fz des PF multipliee par le rapport de poids')
    plt.legend()

plt.show()



