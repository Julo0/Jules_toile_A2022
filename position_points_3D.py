
"""
Permet de générer la matrice de rotatio permettant de tourner le mode pour fitter les marquers de la collecte au repos
"""


import numpy as np
from ezc3d import c3d
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import seaborn as sns
import scipy as sc
from scipy import signal
from sklearn.linear_model import LinearRegression


def open_c3d(participant, trial_name):
    dossiers = ['statique', 'participant_01', 'participant_02']
    file_path = '/home/lim/Documents/Thea_final/Collecte/c3d_files/' + dossiers[participant]
    c3d_file = c3d(file_path + '/' + trial_name + '.c3d')
    return c3d_file

def Named_markers (c3d_experimental) :
    labels=c3d_experimental['parameters']['POINT']['LABELS']['value']

    indices_supp=[]
    for i in range (len(labels)) :
        if '*' in labels[i] :
            indices_supp=np.append(indices_supp,i)
    ind_stop=int(indices_supp[0])

    labels=c3d_experimental['parameters']['POINT']['LABELS']['value'][0:ind_stop]
    named_positions=c3d_experimental['data']['points'][0:3, 0:ind_stop, :]

    ind_milieu = labels.index('t67')
    moyenne_milieu = np.array([np.mean(named_positions[i, ind_milieu, :100]) for i in range(3)])
    return labels, ind_stop, ind_milieu, moyenne_milieu, named_positions

def normalize_position(named_positions,moyenne) :
    #on soustrait la moyenne de la position du milieu
    for i in range(3):
        named_positions[i, :, :] = named_positions[i, :, :] - moyenne[i]

    # on remet les axes dans le meme sens que dans la modelisation
    named_positions_bonsens = np.copy(named_positions)
    named_positions_bonsens[0, :, :] = - named_positions[1, :, :]
    named_positions_bonsens[1, :, :] = named_positions[0, :, :]

    position_moyenne = np.zeros((3, ind_stop))
    for ind_print in range(ind_stop):
        position_moyenne[:, ind_print] = np.array(
            [np.mean(named_positions_bonsens[i, ind_print, :100]) for i in range(3)])

    return named_positions_bonsens,position_moyenne

def rotation(labels,position_moyenne,named_positions_bonsens):
    ind_bord=labels.index('t73')
    theta=np.arcsin(position_moyenne[0,ind_bord]/position_moyenne[1,ind_bord])
    r = np.array(( (np.cos(theta), -np.sin(theta)),
               (np.sin(theta),  np.cos(theta)) ))
    for i in range(ind_stop):
        named_positions_bonsens[0:2, i] = r.dot(named_positions_bonsens[0:2, i])

    for ind_print in range(ind_stop):
        position_moyenne[:, ind_print] = np.array(
            [np.mean(named_positions_bonsens[i, ind_print, :100]) for i in range(3)])
    return r,ind_bord,named_positions_bonsens,position_moyenne

def affichage_3d(position_moyenne, Pt_ancrage, Pt):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot(Pt_ancrage[:,0],Pt_ancrage[:,1],Pt_ancrage[:,2],'.r',label = 'positions simulées au repos')
    ax.plot(Pt[:,0], Pt[:,1], Pt[:,2], '.r')
    ax.plot(position_moyenne[0, :], position_moyenne[1, :], position_moyenne[2, :], '.', label='positions collectées au repos')

    plt.grid()
    ax.axes.set_zlim3d(bottom=-1, top=1)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    plt.legend()
    plt.title('Positions non modifiées des points labellisés \n de l\'essai labeled_p2_sauthaut_02 à l\'image 100, \n comparé avec les positions au repos prévues')
    plt.show()

def affichage_2d(position_moyenne, Pt_ancrage, Pt) :
    plt.plot(position_moyenne[0,:],position_moyenne[1,:],'.',label='positions au repos')
    plt.axis('equal')
    plt.title('Position moyenne des points de ' + str(trial_name) + ' sur les 100 premiers frames')
    plt.grid()

def subplot_point(labels,ind_print,named_positions_bonsens) :
    ind_print = labels.index(ind_print)
    longueur = np.size(named_positions_bonsens[0, 0])
    x = np.linspace(0, longueur, longueur)
    pos=['x','y','z']
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        if i == 0:
            plt.title('Position du point ' + str(labels[ind_print]) + ' selon x, y, z')
        plt.plot(x, named_positions_bonsens[i, ind_print],'-')
        plt.ylabel(str(pos[i]) + ' (mm)')
    plt.show()

def point_du_cadre(ind_cadre,named_positions,labels) :
    ind = labels.index(ind_cadre)
    longueur = np.size(named_positions[0, 0])
    x = np.linspace(0, longueur, longueur)

    position_point_cadre = named_positions[:,ind]/1000

    plt.subplot(3,1,1)
    plt.plot(x,position_point_cadre[2],label='position non filtree')
    plt.title('Position en z du point ' +str(ind_cadre))
    plt.legend()

    T = int(np.size(position_point_cadre)/3)
    # T=2000
    xbis=np.linspace(0,T,T)
    dt = 0.002
    vitesse_point_cadre = np.array(
        [[(position_point_cadre[j, i + 1] - position_point_cadre[j, i]) / dt for i in range(0, T - 1)] for j in range(3)])
    plt.subplot(3, 1, 2)
    plt.plot(vitesse_point_cadre[2], label='vitesse non filtree')
    plt.legend()
    accel_point_cadre = np.array(
        [[(vitesse_point_cadre[j, i + 1] - vitesse_point_cadre[j, i]) / dt for i in range(0, T - 2)] for j in range(3)])
    plt.subplot(3, 1, 3)
    plt.plot(accel_point_cadre[2], label='accélération non filtree')
    plt.legend()

    # # on filtre le signal de la position :
    # sos = signal.butter(1, [2.5, 200], 'bandpass', fs=1000, output='sos')
    # named_positions = signal.sosfilt(sos, named_positions)
    # plt.subplot(2,1,2)
    # plt.plot( x,named_positions[0,ind],label='position filtree')
    # plt.legend()

    plt.show()

def equation_droite_horizontale(position_moyenne,labels) :
    labels_milieu = [labels.index('t22'), labels.index('t37'), labels.index('t52'), labels.index('M0'),
                               labels.index('t67'), labels.index('M4'), labels.index('t82'), labels.index('t97'),
                               labels.index('t112')]

    x = position_moyenne[0, labels_milieu]
    y = position_moyenne[1, labels_milieu]

    #on fit y sur x
    m, b = np.polyfit(x, y, 1)
    y = m*x + b

    return m,b,x,y

def equation_droite_verticale(position_moyenne,labels) :
    labels_milieu = [labels.index('t62'), labels.index('t63'), labels.index('t64'), labels.index('t65'),
                             labels.index('t66'), labels.index('M6'), labels.index('t67'), labels.index('M2'),
                             labels.index('t68'), labels.index('t69'), labels.index('t70'), labels.index('t71'),
                             labels.index('t72'), labels.index('t73'), labels.index('t74')]

    x = position_moyenne[0, labels_milieu]
    y = position_moyenne[1, labels_milieu]

    # on fit x sur y
    m, b = np.polyfit(y, x, 1)
    x = y*m + b

    return m,b,x,y

def equation_droite_3d_horizontale(ax,position_moyenne,labels,affichage) :
    labels_milieu = [labels.index('t22'), labels.index('t37'), labels.index('t52'), labels.index('M0'),
                               labels.index('t67'), labels.index('M4'), labels.index('t82'), labels.index('t97'),
                               labels.index('t112')]
    if affichage == 1 :
        ax.plot(position_moyenne[0, labels_milieu],position_moyenne[1, labels_milieu],position_moyenne[2, labels_milieu],'.',label = 'points de la ligne horizontale')

    x = position_moyenne[0, labels_milieu]
    y = position_moyenne[1, labels_milieu]
    z = position_moyenne[2, labels_milieu]

    yz=np.array([y,z]).T #shape (N,2)
    p = np.polyfit(x, yz, 1)

    mxy,bxy=p[0,0],p[1,0]
    mxz,bxz=p[0,1],p[1,1]

    y = x*mxy + bxy
    z = x*mxz + bxz

    return p,x,y,z

def equation_droite_3d_verticale(ax,position_moyenne,labels,affichage) :
    labels_milieu = [labels.index('t62'), labels.index('t63'), labels.index('t64'), labels.index('t65'),
                     labels.index('t66'), labels.index('M6'), labels.index('t67'), labels.index('M2'),
                     labels.index('t68'), labels.index('t69'), labels.index('t70'), labels.index('t71'),
                     labels.index('t72')]  # , labels.index('t73'), labels.index('t74')]

    if affichage == 1 :
        ax.plot(position_moyenne[0, labels_milieu],position_moyenne[1, labels_milieu],position_moyenne[2, labels_milieu],'.',label = 'points de la ligne verticale')

    x = position_moyenne[0, labels_milieu]
    y = position_moyenne[1, labels_milieu]
    z = position_moyenne[2, labels_milieu]

    xz=np.array([x,z]).T #shape (N,2)
    #on fit x et z sur y
    p = np.polyfit(y, xz, 1)
    mxy,bxy=p[0,0],p[1,0]
    myz,byz=p[0,1],p[1,1]

    x = y*mxy + bxy
    z = y*myz + byz

    return p,x,y,z

def normer(vecteur) :
    return vecteur/np.linalg.norm(vecteur)

def afficher_regression_3d(ax,position_moyenne,labels,orientation,affichage) :
    if orientation == 'verticale' :
        p, x, y, z = equation_droite_3d_verticale(ax, position_moyenne, labels,affichage)
    if orientation == 'horizontale' :
        p, x, y, z = equation_droite_3d_horizontale(ax, position_moyenne, labels,affichage)
    if affichage == 1 :
        ax.plot(x, y, z, '-', label='premiere regression '+str(orientation))  # trace de la ligne de regression
    return p

def calcul_vecteur_3d (p,orientation,affichage) :
    if orientation == 'verticale' :
        points = np.array([[p[1, 0], 0, p[1, 1]],[700 * p[0, 0] + p[1, 0], 700, 700 * p[0, 1] + p[1, 1]]])
    if orientation == 'horizontale' :
        points = np.array([[0, p[1, 0], p[1, 1]],[300, 300 * p[0, 0] + p[1, 0], 300 * p[0, 1] + p[1, 1]]])
    if affichage == 1 :
        plt.plot(points[:, 0], points[:, 1], points[:, 2],'x',label='position du point sur le vecteur ' +orientation)  # verifier que mes vecteurs sont comme il faut
    vecteur = points[1] - points[0]
    return points,vecteur

def produits_vectoriels_3d (vecteur_vert,vecteur_horz,trial_name,affichage) :
    # calcul du vecteur z perpendiculaire a nos deux vecteurs horz et vert
    vecteur_z = np.cross(vecteur_horz, vecteur_vert)
    points_z = np.array([[0, 0, 0], normer(vecteur_z) * 20])

    # calcul du vecteur horz perpendiculaire aux vecteurs z et vert
    vecteur_horz_normal = np.cross(vecteur_vert, vecteur_z)
    points_horz_normal = np.array([[0, 0, 0], normer(vecteur_horz_normal) * 1000])

    # calcul du vecteur vert qui a pour origine 0
    vecteur_vert_normal = np.cross(vecteur_z, vecteur_horz_normal)
    points_vert_normal = np.array([[0, 0, 0], normer(vecteur_vert_normal) * 1000])

    base = np.array([normer(vecteur_horz_normal), normer(vecteur_vert_normal), normer(vecteur_z)])

    if affichage == 1 :
        ax.plot(points_z[:, 0], points_z[:, 1], points_z[:, 2], '-', label='vecteur z obtenu par produit vectoriel')
        ax.plot(points_horz_normal[:, 0], points_horz_normal[:, 1], points_horz_normal[:, 2], '-',label='vecteur horizontal obtenu par produit vectoriel')
        ax.plot(points_vert_normal[:, 0], points_vert_normal[:, 1], points_vert_normal[:, 2], '-',label='vecteur vertical centré')
        # print('x scalaire y : ' + str(np.dot(normer(vecteur_vert_normal), normer(vecteur_horz_normal))))
        # print('x scalaire z : ' + str(np.dot(normer(vecteur_z), normer(vecteur_horz_normal))))
        # print('y scalaire z : ' + str(np.dot(normer(vecteur_z), normer(vecteur_vert_normal))))
        print('matrice de changement de base pour le fichier ' + trial_name + ': ')
        print(base)

    return base,vecteur_vert_normal,vecteur_horz_normal,vecteur_z

def position_prevue () :

    m=9
    n=15

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

    ind_milieu = int((m * n - 1) / 2)

    def Points_ancrage_repos():
        """
        :param dict_fixed_params: dictionnaire contenant les paramètres fixés
        :return: Pos_repos: cas.DM(n*m,3): coordonnées (2D) des points de la toile
        :return: Pt_ancrage: cas.DM(2*n+2*m,3): coordonnées des points du cadre
        """

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

        Pos_repos_new = np.copy(Pos_repos)
        for i in range (135) :
            Pos_repos_new[i] -= Pos_repos[67]

        # Pt_ancrage, Pos_repos_new = rotation_points(Pos_repos_new, Pt_ancrage)
        return Pt_ancrage, Pos_repos_new

    Pt_ancrage, Pt = Points_ancrage_repos()
    return Pt_ancrage,Pt

######################################################################################################################
#PARAMETRES :
participant=2 #2 #0=statique #1
trial_name='labeled_p2_sauthaut_01'

#ACTIONS
comparer_simulation = 1
visualiser_evolution=0
action = 'regression_3d'#'regression_3d' #'points_centres' #'points-cadre'
verification_2d = 1
#####################################################################################################################
#CENTRER LES POINTS LABELES :
c3d_experimental= open_c3d(participant,trial_name)


#donner l'ensemble des labeled markers et la position moyenne du point du milieu sur les 100 premiers frames
labels, ind_stop, ind_milieu, moyenne_milieu, named_positions = Named_markers(c3d_experimental)

# #calculer les positions de chaque point en ayant soustrait la moyenne et mettre les axes dans le bon sens
named_positions,position_moyenne = normalize_position(named_positions,moyenne_milieu)

longueur = np.size(named_positions[0,0])
if visualiser_evolution==1 : #exemple pour voir mouvement des points du cadre
#soustraire a chaque point sa valeur a vide pour voir l'amplitude du mouvement
    for i in range (longueur) :
        named_positions[:,:,i]-=position_moyenne

if action == 'points_centres' : #points avant la regression/rotation
    ind_print='C8'
    Pt_ancrage, Pt = position_prevue()
    # affichage_2d(named_positions[:,:,800])
    # affichage_2d(position_moyenne/1000, Pt_ancrage, Pt)
    affichage_3d(named_positions[:,:,100]/1000,Pt_ancrage, Pt)
    plt.show()

if action == 'raw' : #points labellés avant toute modif
    Pt_ancrage, Pt = position_prevue()
    affichage_2d(named_positions[:,:,100]/1000, Pt_ancrage, Pt)
    affichage_3d(named_positions[:,:,100]/1000, Pt_ancrage, Pt)
    plt.show()


####################################################################################################################
#REGRESSION ET CHANGEMENT DE BASE EN 3D

if action == 'regression_3d' :
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot(position_moyenne[0], position_moyenne[1], position_moyenne[2], '.', label='positions moyennes de base')

    #premiere regression a partir des positions de base (affichage des lignes et des points de la regression)
    p_vert = afficher_regression_3d(ax,position_moyenne,labels,'verticale',affichage=0)
    p_horz = afficher_regression_3d(ax,position_moyenne,labels,'horizontale',affichage=0)

    #vecteurs directeurs des regressions de base :
    points_vert,vecteur_vert = calcul_vecteur_3d(p_vert,'verticale', affichage = 0)
    points_horz,vecteur_horz = calcul_vecteur_3d(p_horz,'horizontale', affichage = 0)

    #calcul de la nouvelle base de vecteurs
    base,vecteur_vert_normal,vecteur_horz_normal,vecteur_z = produits_vectoriels_3d(vecteur_vert,vecteur_horz,trial_name,affichage = 1)

    #calcul des nouveaux points :
    for i in range (ind_stop) :
        for j in range (longueur) :
            named_positions[:,i,j] = np.dot (base,named_positions[:,i,j])
    #calcul et affichage des nouvelles positions moyennes :
    position_moyenne_new = np.zeros(np.shape(position_moyenne))
    for ind_print in range(ind_stop):
        position_moyenne_new[:, ind_print] = np.array(
            [np.mean(named_positions[i, ind_print, :100]) for i in range(3)])
    ax.plot(position_moyenne_new[0],position_moyenne_new[1],position_moyenne_new[2],'.r',label='nouvelles positions moyennes')

    #calcul des nouvelles regressions :
    p_vert_new = afficher_regression_3d(ax,position_moyenne_new,labels,'verticale',affichage=0)
    p_horz_new = afficher_regression_3d(ax,position_moyenne_new,labels,'horizontale',affichage=0)

    #vecteurs directeurs des nouvelles regressions :
    points_vert_new,vecteur_vert_new = calcul_vecteur_3d(p_vert_new,'verticale', affichage = 0)
    points_horz_new,vecteur_horz_new = calcul_vecteur_3d(p_horz_new,'horizontale', affichage = 0)
    vecteur_z_new=np.cross(vecteur_horz_new,vecteur_vert_new)

    #verifier que les points constituent bien une base orthonormee:
    if np.abs(np.max(np.cross(normer(vecteur_horz_new), [1, 0, 0]))) < 5e-02 and np.abs(
                np.max(np.cross(normer(vecteur_vert_new), [0, 1, 0]))) < 5e-02 and np.abs(
                np.max(np.cross(normer(vecteur_z_new), [0, 0, 1]))) < 5e-02:
        print('les points sont bien dans la base orthogonale')
    else :
        print ('points pas assez alignés')

    ax.axes.set_zlim3d(bottom=-20, top=20)
    plt.grid()
    plt.legend()
    plt.title('Fichier ' +str(trial_name))
    plt.show()

    #################################################################
    if verification_2d == 1 :
        Pt_ancrage, Pt = position_prevue()


        #VERIFICATION 2D XY
        #trace des points
        fig=plt.figure()
        plt.plot(Pt[:, 0], Pt[:, 1], '.r', label='positions simulées au repos')
        plt.plot(Pt_ancrage[:, 0], Pt_ancrage[:, 1], '.r')
        plt.plot(position_moyenne_new[0]/1000,position_moyenne_new[1]/1000,'.b',label = 'positions collectées après rotation')
        # plt.plot(position_moyenne_new[0],position_moyenne_new[1],'.',label='nouvelles positions après rotation')

        #trace des nouvelles regressions
        m_vert_nouveau,b_vert_nouveau,x_vert_nouveau,y_vert_nouveau=equation_droite_verticale(position_moyenne_new,labels)
        plt.plot(x_vert_nouveau/1000,y_vert_nouveau/1000, '-b',label='axes interpolés après rotation')
        m_horz_nouveau, b_horz_nouveau, x_horz_nouveau, y_horz_nouveau = equation_droite_horizontale(position_moyenne_new,labels)
        plt.plot(x_horz_nouveau/1000,y_horz_nouveau/1000,'-b')
        plt.legend()

        plt.grid()
        plt.legend()
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.axis('equal')
        plt.title('Axes en 2D après rotation \n pour l\'essai ' + str(trial_name))
        plt.show()
########################################

if action == 'points_cadre' :
    ind_cadre='C32'
    point_du_cadre(ind_cadre,named_positions,labels)

###################################################################

