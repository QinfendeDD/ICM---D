# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


####读取数据
data_dir = 'C:/Users/30920/Desktop/2020ICM/demo'
fullevents = pd.read_csv(data_dir+'/fullevents.csv')
matches = pd.read_csv(data_dir+'/matches.csv')
passingevents = pd.read_csv(data_dir+'/passingevents.csv')
win = matches[matches.Outcome == 'win'].shape[0]
loss = matches[matches.Outcome == 'loss'].shape[0]
tie = matches[matches.Outcome == 'tie'].shape[0]
huskies_passingevents = passingevents[passingevents.TeamID == 'Huskies' ] #从所有传球数据中选择huskies队队员数据
opponent_passingevents = passingevents[passingevents.TeamID != 'Huskies' ] #从所有传球数据中选择除了huskies队队员数据
def get_passing_graph(passingevents):
    team_name = sorted(list(set(passingevents['OriginPlayerID'])))
    passing_record = {}
    for item in passingevents.values:
        s=str(item)
        if s in passing_record.keys():
            passing_record[s] = passing_record[s] + 1
        else:
            passing_record[s] = 1    
    passing_graph = nx.DiGraph()
    passing_graph.add_nodes_from(team_name) #在有向图中添加节点
    for keys, value in passing_record.items():
        passing_graph.add_edges_from([tuple(keys.replace('[','').replace(']','').replace("'",'').split(" "))], weight = value) #添加有向边数据
    return passing_graph


##所有时间传球图
huskies_team = sorted(list(set(huskies_passingevents['OriginPlayerID']))) #huskies队所有队员名字
passing_graph = get_passing_graph(huskies_passingevents.loc[:,['OriginPlayerID','DestinationPlayerID']])
transitivity = nx.transitivity(passing_graph) #可传递性
average_shortest_path_length = nx.average_shortest_path_length(passing_graph,weight='weight') #平均最短路径长度
# pos=nx.spring_layout(passing_graph) # positions for all nodes
# nx.draw_networkx_edges(passing_graph,pos,alpha=0.4,width=[float(d['weight']*0.03) for (u,v,d) in passing_graph.edges(data=True)])
# nx.draw_networkx_nodes(passing_graph,pos,node_size=100)
# plt.savefig('passing_graph.png',dpi=800)
# plt.show()

##对于第一场
passing_event_1 = huskies_passingevents[huskies_passingevents.MatchID == 1]
passing_graph_1 = get_passing_graph(passing_event_1.loc[:,['OriginPlayerID','DestinationPlayerID']])
transitivity_1 = nx.transitivity(passing_graph_1)
print(passing_graph_1.edges(data = True))
pos=nx.spring_layout(passing_graph_1) # positions for all nodes
nx.draw_networkx_edges(passing_graph_1,pos,width=[float(d['weight']*0.05) for (u,v,d) in passing_graph_1.edges(data=True)])
nx.draw_networkx_nodes(passing_graph_1,pos,node_size=100)
plt.savefig('passing_graph_1.png',dpi=800)
plt.show()

##对于所有场次 得到indicator对于场次的函数
number_matches = huskies_passingevents.MatchID.max()
transitivity_record = np.zeros(number_matches)
average_shortest_path_length_record = np.zeros(number_matches)
for matchids in np.arange(0,number_matches):
    passing_event_itr = huskies_passingevents[huskies_passingevents.MatchID == matchids+1]
    passing_graph_itr = get_passing_graph(passing_event_itr.loc[:,['OriginPlayerID','DestinationPlayerID']])
    transitivity_itr = nx.transitivity(passing_graph_itr)
    average_shortest_path_length_itr = nx.average_shortest_path_length(passing_graph_itr)
    transitivity_record[matchids] = transitivity_itr
    average_shortest_path_length_record[matchids] = average_shortest_path_length_itr
    
# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax1.scatter(np.arange(1,number_matches+1),transitivity_record)
# plt.ylabel('Transitivity')
# plt.xlabel('Match')
# ax2 = fig.add_subplot(212)
# ax2.scatter(np.arange(1,number_matches+1),average_shortest_path_length_record)
# plt.ylabel('Average passing length')
# plt.xlabel('Match')
# plt.savefig('Matrices graph',dpi=800)

# lst = [1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 19, 20, 21, 22, 23]    # 连续数字

# fun = lambda x: x[1]-x[0]
# for k, g in groupby(enumerate(lst), fun):
#     l1 = [j for i, j in g]    # 连续数字的列表
#     if len(l1) > 1:
#         scop = str(min(l1)) + '-' + str(max(l1))    # 将连续数字范围用"-"连接
#     else:
#         scop = l1[0]
#     print("连续数字范围：{}".format(scop))

###问题二
##平均位置
#第一场平均位置    
# huskies_passingevents_match_1 = huskies_passingevents[huskies_passingevents.MatchID == 1]
# huskies_passingevents_match_1_Xmean = 0.5*(huskies_passingevents_match_1['EventOrigin_x'].mean()+huskies_passingevents_match_1['EventDestination_x'].mean())
# huskies_passingevents_match_1_Ymean = 0.5*(huskies_passingevents_match_1['EventOrigin_y'].mean()+huskies_passingevents_match_1['EventDestination_y'].mean())
#huskies队所有场次球平均位置
huskies_average_ball_position_x = np.zeros(number_matches)
huskies_average_ball_position_y = np.zeros(number_matches)
#对手所有场次球平均位置
opponent_average_ball_position_x = np.zeros(number_matches)
opponent_average_ball_position_y = np.zeros(number_matches)

for matches in np.arange(0, number_matches):
    huskies_passingevents_itr = huskies_passingevents[huskies_passingevents.MatchID == matches+1]
    opponent_passingevents_itr = opponent_passingevents[opponent_passingevents.MatchID == matches+1]
    
    huskies_passing_graph_itr = get_passing_graph(huskies_passingevents_itr.loc[:,['OriginPlayerID','DestinationPlayerID']])
    opponent_passing_graph_itr = get_passing_graph(opponent_passingevents_itr.loc[:,['OriginPlayerID','DestinationPlayerID']])
    
    huskies_passingevents_match_itr_Xmean = 0.5*(huskies_passingevents_itr['EventOrigin_x'].mean()+huskies_passingevents_itr['EventDestination_x'].mean())
    huskies_passingevents_match_itr_Ymean = 0.5*(huskies_passingevents_itr['EventOrigin_y'].mean()+huskies_passingevents_itr['EventDestination_y'].mean())
    opponent_passingevents_match_itr_Xmean = 0.5*(opponent_passingevents_itr['EventOrigin_x'].mean()+opponent_passingevents_itr['EventDestination_x'].mean())
    opponent_passingevents_match_itr_Ymean = 0.5*(opponent_passingevents_itr['EventOrigin_y'].mean()+opponent_passingevents_itr['EventDestination_y'].mean())
        
    huskies_average_ball_position_x[matches] = huskies_passingevents_match_itr_Xmean
    huskies_average_ball_position_y[matches] = huskies_passingevents_match_itr_Ymean
    opponent_average_ball_position_x[matches] = opponent_passingevents_match_itr_Xmean
    opponent_average_ball_position_y[matches] = opponent_passingevents_match_itr_Xmean


# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax1.scatter(np.arange(1,number_matches+1),huskies_average_ball_position_x,label='Huskies')
# ax1.scatter(np.arange(1,number_matches+1),opponent_average_ball_position_x,label='Opponent')
# plt.xlim(1,number_matches)
# plt.ylabel('Average X')
# plt.xlabel('Match')
# plt.legend()
#
# ax2 = fig.add_subplot(212)
# ax2.scatter(np.arange(1,number_matches+1),huskies_average_ball_position_y,label='Huskies')
# ax2.scatter(np.arange(1,number_matches+1),opponent_average_ball_position_y,label='Opponent')
# plt.xlim(1,number_matches)
# plt.ylabel('Average Y')
# plt.xlabel('Match')
# plt.legend()
# plt.savefig('Average position graph',dpi=800)
    
##对于第一场的图分析
#得到第一场双方的传球图
huskies_passingevents_match_1 = huskies_passingevents[huskies_passingevents.MatchID == 1]
opponent_passingevents_match_1 = opponent_passingevents[opponent_passingevents.MatchID == 1]
huskies_passing_graph_1 = get_passing_graph(huskies_passingevents_match_1.loc[:,['OriginPlayerID','DestinationPlayerID']])
opponent_passing_graph_1 = get_passing_graph(opponent_passingevents_match_1.loc[:,['OriginPlayerID','DestinationPlayerID']])
#得到下列描述图的特征
huskies_degree_centrality = nx.degree_centrality(huskies_passing_graph_1)
opponent_degree_centrality = nx.degree_centrality(opponent_passing_graph_1)
huskies_betweeness_centrality = nx.betweenness_centrality(huskies_passing_graph_1,weight='weight')
opponent_betweeness_centrality = nx.betweenness_centrality(opponent_passing_graph_1,weight='weight')
huskies_closeness_centrality = nx.closeness_centrality(huskies_passing_graph_1,distance='weight')
opponent_closeness_centrality = nx.closeness_centrality(opponent_passing_graph_1,distance='weight')
huskies_eigenvector_centrality = nx.eigenvector_centrality(huskies_passing_graph_1,weight='weight')
opponent_eigenvector_centrality = nx.eigenvector_centrality(opponent_passing_graph_1,weight='weight')
huskies_clustering_coef = nx.clustering(huskies_passing_graph_1,weight='weight')
opponent_clustering_coef = nx.clustering(opponent_passing_graph_1,weight='weight')

huskies_degree_centrality_mean = np.mean(list(huskies_degree_centrality.values()))
huskies_betweeness_centrality_mean = np.mean(list(huskies_betweeness_centrality.values()))
huskies_closeness_centrality_mean = np.mean(list(huskies_closeness_centrality.values()))
huskies_eigenvector_centrality_mean = np.mean(list(huskies_eigenvector_centrality.values()))

huskies_clustering_coef = np.mean(list(huskies_clustering_coef.values()))

opponent_degree_centrality_mean = np.mean(list(opponent_degree_centrality.values()))
opponent_betweeness_centrality_mean = np.mean(list(opponent_betweeness_centrality.values()))
opponent_closeness_centrality_mean = np.mean(list(opponent_closeness_centrality.values()))
opponent_eigenvector_centrality_mean = np.mean(list(opponent_eigenvector_centrality.values()))
opponent_clustering_coef = np.mean(list(opponent_clustering_coef.values()))

graph_matrics_size = [len(list(huskies_degree_centrality.values())),len(list(opponent_degree_centrality.values()))]

                           
graph_matrics = pd.DataFrame(np.zeros((109,3)),columns=['Graph_Matrics','Value','Team'])
graph_matrics.loc[0:13,['Graph_Matrics']] = 'Degree Centrality'
graph_matrics.loc[0:13,['Value']] = list(huskies_degree_centrality.values())
graph_matrics.loc[14:27,['Graph_Matrics']] = 'Betweenness Centrality'
graph_matrics.loc[14:27,['Value']] = list(huskies_betweeness_centrality.values())
graph_matrics.loc[28:41,['Graph_Matrics']] = 'Closeness Centrality'
graph_matrics.loc[28:41,['Value']] = list(huskies_closeness_centrality.values())
graph_matrics.loc[42:55,['Graph_Matrics']] = 'Eigenvector Centrality'
graph_matrics.loc[42:55,['Value']] = list(huskies_eigenvector_centrality.values())

graph_matrics.loc[56,['Graph_Matrics']] = 'Clustering Coefficient'
graph_matrics.loc[56,['Value']] = opponent_clustering_coef
graph_matrics.loc[57:69,['Graph_Matrics']] = 'Degree Centrality'
graph_matrics.loc[57:69,['Value']] = list(opponent_degree_centrality.values())
graph_matrics.loc[70:82,['Graph_Matrics']] = 'Betweenness Centrality'
graph_matrics.loc[70:82,['Value']] = list(opponent_betweeness_centrality.values())
graph_matrics.loc[83:95,['Graph_Matrics']] = 'Closeness Centrality'
graph_matrics.loc[83:95,['Value']] = list(opponent_closeness_centrality.values())
graph_matrics.loc[96:108,['Graph_Matrics']] = 'Eigenvector Centrality'
graph_matrics.loc[96:108,['Value']] = list(opponent_eigenvector_centrality.values())
graph_matrics.loc[109,['Graph_Matrics']] = 'Clustering Coefficient'
graph_matrics.loc[109,['Value']] = opponent_clustering_coef
graph_matrics.loc[0:56,['Team']] = 'Huskies'
graph_matrics.loc[57:109,['Team']] = 'Opponent'

# f, ax = plt.subplots()
# plt.xticks(rotation='45')
# sns.barplot(x='Graph_Matrics',y='Value',data=graph_matrics,hue='Team')
# plt.savefig('Graph Matrics',dpi=800)
# plt.show()
