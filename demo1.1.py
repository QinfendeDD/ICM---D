import numpy as np
import networkx as nx
from pandas import read_csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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


def set_size_by_degree(G):
    nsize = np.zeros(11)
    for _ in range(11):
        nsize[_] = G.degree(_)
    median = np.median(nsize)
    for _ in range(11):
        nsize[_] = 1000 + 450 * (nsize[_] - median)
    return nsize


def set_color_by_degree(G):
    ncolor = ['g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', ]
    nsize = set_size_by_degree(G)
    ncolor[np.argmax(nsize)] = 'r'
    ncolor[np.argmin(nsize)] = 'y'
    return ncolor


def get_average_position(matchID, matchPeriod, player_name):
    data1 = data_passing_events[data_passing_events['OriginPlayerID'] == player_name]
    data2 = data1[data1['MatchID'] == matchID]
    data3 = data2[data2['MatchPeriod'] == matchPeriod]
    data4 = data3.loc[:, 'EventOrigin_x']
    data5 = data3.loc[:, 'EventOrigin_y']
    return [data4.mean(), data5.mean()]


def clustering_by_position(position):
    ncolor = ['g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', ]
    kmeans = KMeans(n_clusters=4, random_state=9).fit(position)
    for _ in range(11):
        if kmeans.labels_[_] == 0:
            ncolor[_] = 'g'
        elif kmeans.labels_[_] == 1:
            ncolor[_] = 'b'
        elif kmeans.labels_[_] == 2:
            ncolor[_] = 'y'
        elif kmeans.labels_[_] == 3:
            ncolor[_] = 'r'
    return ncolor
# def gather_into_one():


h_players = ['Huskies_D1', 'Huskies_D2', 'Huskies_D3', 'Huskies_D4',
             'Huskies_M1', 'Huskies_M2', 'Huskies_M3',
             'Huskies_F1', 'Huskies_F2', 'Huskies_F3',
             'Huskies_G1']
o_players = ['Opponent1_D1', 'Opponent1_D2', 'Opponent1_D3', 'Opponent1_D4',
             'Opponent1_M1', 'Opponent1_M2', 'Opponent1_M3',
             'Opponent1_F1', 'Opponent1_F2', 'Opponent1_F3',
             'Opponent1_G1']

# 1.邻接矩阵 2.平均位置 3.设置标签 4.设置颜色
adjacency_matrix_1 = get_adjacency_matrix(h_players, 1, '1H')
# 球员平均位置
coordinates = np.zeros([11, 2])
for i in range(11):
    coordinates[i] = get_average_position(1, '1H', h_players[i])
plt.figure(1)
G = nx.from_numpy_matrix(adjacency_matrix_1, create_using=nx.Graph())
pos = dict(zip(G.nodes(), coordinates))
nlabels = ['D1', 'D2', 'D3', 'D4', 'M1', 'M2', 'M3', 'F1', 'F2', 'F3', 'G1']
labels = dict(zip(G.nodes(), nlabels))
ncolor = clustering_by_position(coordinates)
nx.draw(G, pos, node_size=set_size_by_degree(G), node_color=ncolor)
nx.draw_networkx_labels(G, pos, labels, font_size=15, )
nx.draw_networkx_edges(G, pos, width=[float(d['weight'] * 0.5) for (u, v, d) in G.edges(data=True)])
plt.show()
