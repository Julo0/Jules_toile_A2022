
import numpy as np
from ezc3d import c3d
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import seaborn as sns


def open_c3d(participant, trial_name):
    dossiers = ['statique', 'participant_01', 'participant_02']
    file_path = '/home/lim/Documents/Thea_final/Collecte/c3d_files/' + dossiers[participant]
    c3d_file = c3d(file_path + '/' + trial_name + '.c3d')
    return c3d_file

def positions_reelles (c3d_experimental) :
    labels = c3d_experimental['parameters']['POINT']['LABELS']['value']
    indices_supp = []
    for i in range(len(labels)):
        if '*' in labels[i]:
            indices_supp = np.append(indices_supp, i)

    if '*' not in labels[-1] : #cas ou tous les points sont labelles
        ind_stop = len(labels) - 1
    else :
        ind_stop = int(indices_supp[0])

    labels = c3d_experimental['parameters']['POINT']['LABELS']['value'][0:ind_stop]
    named_positions = c3d_experimental['data']['points'][0:3, 0:ind_stop, :]

    ind_milieu = labels.index('t67')
    moyenne_milieu = np.array([np.mean(named_positions[i, ind_milieu, :100]) for i in range(3)])
    return labels, ind_stop, ind_milieu, moyenne_milieu, named_positions

def point_le_plus_bas(points,ind_stop,labels) :
    argmin_marqueur=np.nanargmin(points[2])
    label_min=labels[argmin_marqueur]
    print('Altitude minimale obtenue au marqueur ' + str(label_min))
    return argmin_marqueur



participant=0 #2 #0=statique
instant = 2000/4
trial_name='labeled_statique_centrefront_D7'

c3d_experimental = open_c3d(participant,trial_name)
labels, ind_stop, ind_milieu, moyenne_milieu, named_positions = positions_reelles(c3d_experimental)

position_instant = named_positions[:,:,int(instant)]

argmin_marqueur= point_le_plus_bas(position_instant,ind_stop,labels)

print(position_instant[:,argmin_marqueur])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot(position_instant[0, :], position_instant[1, :], position_instant[2, :], '.', label='positions a instant ' + str(instant))
ax.plot(position_instant[0, argmin_marqueur], position_instant[1, argmin_marqueur], position_instant[2, argmin_marqueur], '.', label='Point le plus bas')
plt.grid()
plt.title('Positions a instant ' + str(instant*4))
ax.axes.set_zlim3d(bottom=500, top=1200)
plt.show()
