# -*- coding: utf-8 -*-
# @Time    : 2018/3/1 9:18
# @Author  : Liunux
# @Email   : 103996977@qq.com
# @File    : Nba_Predict.py
# @Software: PyCharm

import pandas as pd

team_elo={}
base_elo=1600

def get_elo(team):
    try:
        return team_elo[team]
    except:
        team_elo[team]=1600
        return team_elo[team]

def elo_score(winner,loser):
    winner_rank=get_elo(winner)
    loser_rank=get_elo(loser)
    if winner_rank < 2100:
        k = 32
    elif winner_rank >= 2100 and winner_rank < 2400:
        k = 24
    else:
        k = 16
    E_winner=1/(1+pow(10,(loser_rank-winner_rank)/400))
    E_loser=1-E_winner
    new_winner_rank=winner_rank+k*(1-E_winner)
    new_loser_rank=loser_rank+k*(1-E_loser)
    return round(new_winner_rank,2),round(new_winner_rank,2)

def init_data():
    filepath="D:/_Liunux/study/GitHub/ml/NBA_Predit/"
    Mstat=pd.read_csv(filepath+"Miscellaneous_Stats.txt")
    Ostat=pd.read_csv(filepath+"Team_Per_Game_Stats.txt")
    Tstat=pd.read_csv(filepath+"Opponent_Per_Game_Stats.txt")
    new_Mstat = Mstat.drop(['Rk', 'Arena'], axis=1)
    new_Ostat = Ostat.drop(['Rk', 'G', 'MP'], axis=1)
    new_Tstat = Tstat.drop(['Rk', 'G', 'MP'], axis=1)

    team_stats=pd.merge(new_Mstat,new_Ostat,how='left',on='Team')
    team_stats=pd.merge(team_stats,new_Tstat,how='left',on='Team')
    return team_stats.set_index('Team',inplace=False,drop=True)
