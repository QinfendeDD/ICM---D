import numpy as np
import networkx as nx
from pandas import read_csv
import matplotlib.pyplot as plt
from matplotlib import colors

try:
    f = open('passingevents.csv', encoding='UTF-8')
    data_passing_events = read_csv(f)
except:
    print('no file in dic!')


# 返回第matchID场次上（下）半场某两个球员之间的传球次数
def num_bt_players(matchID, matchPeriod, player_name1, player_name2):
    data1 = data_passing_events[data_passing_events['OriginPlayerID'] == player_name1]
    data2 = data1[data1['MatchID'] == matchID]
    data3 = data2[data2['MatchPeriod'] == matchPeriod]
    data4 = data3[data3['DestinationPlayerID'] == player_name2]
    return data4.shape[0]


# 返回邻接矩阵
def get_adjacency_matrix(players, matchID, matchPeriod):
    adjacency_matrix = np.zeros([11, 11])
    for i in range(11):
        for j in range(11):
            adjacency_matrix[i, j] = num_bt_players(matchID, matchPeriod, players[i], players[j])
    return adjacency_matrix


h_players = ['Huskies_D1', 'Huskies_D2', 'Huskies_D3', 'Huskies_D4',
             'Huskies_M1', 'Huskies_M2', 'Huskies_M3',
             'Huskies_F1', 'Huskies_F2', 'Huskies_F3',
             'Huskies_G1']
o_players = []

adjacency_matrix_1 = get_adjacency_matrix(h_players, 1, '1H')
print(adjacency_matrix_1)

G = nx.from_numpy_matrix(adjacency_matrix_1, create_using=nx.MultiDiGraph())
print(G.nodes())
coordinates = np.array(
    [[15, 80], [15, 60], [15, 40], [15, 20], [25, 75], [25, 50], [25, 25], [40, 75], [40, 50], [40, 25], [0, 50]])
pos = dict(zip(G.nodes(), coordinates))
print(pos)
nx.draw(G, pos)
nlabels = ['D1', 'D2', 'D3', 'D4', 'M1', 'M2', 'M3', 'F1', 'F2', 'F3', 'G1']
labels = dict(zip(G.nodes(), nlabels))
box = dict(facecolor='yellow', pad=5, alpha=0.2)
nx.draw_networkx_labels(G, pos, labels, font_size=15,bbox=box)
plt.show()
