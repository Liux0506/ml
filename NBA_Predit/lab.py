
# coding: utf-8

# # 实验名称：利用 Python 进行 NBA 比赛数据分析
# 
# ## 一、实验介绍
# 
# ### 1.1 内容简介
# 
# 不知道你是否朋友圈被刷屏过 nba 的某场比赛进度或者结果？或者你就是一个 nba 狂热粉，比赛中的每个进球，抢断或是逆转压哨球都能让你热血沸腾。除去观赏精彩的比赛过程，我们也同样好奇比赛的结果会是如何。因此本节课程，将给同学们展示如何使用 nba 比赛的以往统计数据，判断每个球队的战斗力，及预测某场比赛中的结果。
# 
# 我们将基于 2015-2016 年的 **NBA** 常规赛及季后赛的比赛统计数据，预测在当下正在进行的 2016-2017 常规赛每场赛事的结果。
# 
# ![](https://dn-anything-about-doc.qbox.me/document-uid291340labid2647timestamp1489325776168.png/wm)
# 
# ### 1.2 实验知识点
# 
# *   nba 球队的`Elo score`计算
# *   特征向量
# *   逻辑回归
# 
# ### 1.3 实验流程
# 
# 本次课程我们将按照下面的流程实现 **NBA** 比赛数据分析的任务：
# 
# 1.  获取比赛统计数据
# 2.  比赛数据分析，得到代表每场比赛每支队伍状态的特征表达
# 3.  利用**机器学习**方法学习每场比赛与胜利队伍的关系，并对 2016-2017 的比赛进行预测

# ## 二、获取 NBA 比赛统计数据
# 
# ### 2.1 比赛数据介绍
# 
# 在本次实验中，我们将采用 [Basketball Reference.com](http://www.basketball-reference.com/) 中的统计数据。在这个网站中，你可以看到不同球员、队伍、赛季和联盟比赛的基本统计数据，如得分，犯规次数等情况，胜负次数等情况。而我们在这里将会使用 **2015-16 NBA Season Summary** 中数据。
# 
# ![](https://dn-anything-about-doc.qbox.me/document-uid291340labid2647timestamp1489325761209.png/wm)
# 
# 在这个 2015-16 总结的所有表格中，我们将使用的是以下三个数据表格：
# 
# *   **Team Per Game Stats**：每支队伍平均每场比赛的表现统计
# 
# | 数据名 | 含义 |
# | --- | --- |
# | Rk -- Rank | 排名 |
# | G -- Games | 参与的比赛场数（都为 82 场） |
# | MP -- Minutes Played | 平均每场比赛进行的时间 |
# | FG--Field Goals | 投球命中次数 |
# | FGA--Field Goal Attempts | 投射次数 |
# | FG%--Field Goal Percentage | 投球命中次数 |
# | 3P--3-Point Field Goals | 三分球命中次数 |
# | 3PA--3-Point Field Goal Attempts | 三分球投射次数 |
# | 3P%--3-Point Field Goal Percentage | 三分球命中率 |
# | 2P--2-Point Field Goals | 二分球命中次数 |
# | 2PA--2-point Field Goal Attempts | 二分球投射次数 |
# | 2P%--2-Point Field Goal Percentage | 二分球命中率 |
# | FT--Free Throws | 罚球命中次数 |
# | FTA--Free Throw Attempts | 罚球投射次数 |
# | FT%--Free Throw Percentage | 罚球命中率 |
# | ORB--Offensive Rebounds | 进攻篮板球 |
# | DRB--Defensive Rebounds | 防守篮板球 |
# | TRB--Total Rebounds | 篮板球总数 |
# | AST--Assists | 助攻 |
# | STL--Steals | 抢断 |
# | BLK -- Blocks | 封盖 |
# | TOV -- Turnovers | 失误 |
# | PF -- Personal Fouls | 个犯 |
# | PTS -- Points | 得分 |
# 
# *   **Opponent Per Game Stats**：所遇到的对手平均每场比赛的统计信息，所包含的统计数据与 **Team Per Game Stats** 中的一致，只是代表的该球队对应的对手的
# 
# *   **Miscellaneous Stats**：综合统计数据
# 
# | 数据项 | 数据含义 |
# | --- | --- |
# | Rk (Rank) | 排名 |
# | Age | 队员的平均年龄 |
# | W (Wins) | 胜利次数 |
# | L (Losses) | 失败次数 |
# | PW (Pythagorean wins) | 基于毕达哥拉斯理论计算的赢的概率 |
# | PL (Pythagorean losses) | 基于毕达哥拉斯理论计算的输的概率 |
# | MOV (Margin of Victory) | 赢球次数的平均间隔 |
# | SOS (Strength of Schedule) | 用以评判对手选择与其球队或是其他球队的难易程度对比，0 为平均线，可以为正负数 |
# | SRS (Simple Rating System) | 3 |
# | ORtg (Offensive Rating) | 每 100 个比赛回合中的进攻比例 |
# | DRtg (Defensive Rating) | 每 100 个比赛回合中的防守比例 |
# | Pace (Pace Factor) | 每 48 分钟内大概会进行多少个回合 |
# | FTr (Free Throw Attempt Rate) | 罚球次数所占投射次数的比例 |
# | 3PAr (3-Point Attempt Rate) | 三分球投射占投射次数的比例 |
# | TS% (True Shooting Percentage) | 二分球、三分球和罚球的总共命中率 |
# | eFG% (Effective Field Goal Percentage) | 有效的投射百分比（含二分球、三分球） |
# | TOV% (Turnover Percentage) | 每 100 场比赛中失误的比例 |
# | ORB% (Offensive Rebound Percentage) | 球队中平均每个人的进攻篮板的比例 |
# | FT/FGA | 罚球所占投射的比例 |
# | eFG% (Opponent Effective Field Goal Percentage) | 对手投射命中比例 |
# | TOV% (Opponent Turnover Percentage) | 对手的失误比例 |
# | DRB% (Defensive Rebound Percentage) | 球队平均每个球员的防守篮板比例 |
# | FT/FGA (Opponent Free Throws Per Field Goal Attempt) | 对手的罚球次数占投射次数的比例 |
# 
# > 毕达哥拉斯定律：
# > 
# > ![](https://dn-anything-about-doc.qbox.me/document-uid291340labid2647timestamp1490108859908.png/wm)
# 
# 我们将用这三个表格来评估球队过去的战斗力，另外还需 **2015-16 NBA Schedule and Results** 中的 2015~2016 年的 nba 常规赛及季后赛的每场比赛的比赛数据，用以评估`Elo score`（在之后的实验小节中解释）。在`Basketball Reference.com`中按照从常规赛至季后赛的时间。列出了 2015 年 10 月份至 2016 年 6 月份的每场比赛的比赛情况。
# 
# ![](https://dn-anything-about-doc.qbox.me/document-uid291340labid2647timestamp1489325843917.png/wm)
# 
# 可在上图中，看到 2015 年 10 月份的部分比赛数据。在每个 _Schedule_ 表格中所包含的数据为：
# 
# | 数据项 | 数据含义 |
# | --- | --- |
# | Date | 比赛日期 |
# | Start (ET) | 比赛开始时间 |
# | Visitor/Neutral | 客场作战队伍 |
# | PTS | 客场队伍最后得分 |
# | Home/Neutral | 主场队伍 |
# | PTS | 主场队伍最后得分 |
# | Notes | 备注，表明是否为加时赛等 |
# 
# 在预测时，我们同样也需要在 **2016-17 NBA Schedule and Results** 中 2016~2017 年的 NBA 的常规赛比赛安排数据。

# ### 2.2 获取比赛数据
# 
# 我们将以获取 **Team Per Game Stats** 表格数据为例，展示如何获取这三项统计数据。
# 
# 1.  进入到用 [Basketball Reference.com](http://www.basketball-reference.com/) 中，在导航栏中选择`Season`并选择`2015~2016`赛季中的`Summary`：

# ![](https://dn-anything-about-doc.qbox.me/document-uid291340labid2647timestamp1489325866617.png/wm)

# 2.  进入到 2015~2016 年的`Summary`界面后，滑动窗口找到`Team Per Game Stats`表格，并选择左上方的 **Share & more**，在其下拉菜单中选择 **Get table as CSV (for Excel)**：

# ![](https://dn-anything-about-doc.qbox.me/document-uid291340labid2647timestamp1489325884435.png/wm)

# 3.  复制在界面中生的的 csv 格式数据，并复制粘贴至一个文本编辑器保存为 csv 文件即可：

# ![](https://dn-anything-about-doc.qbox.me/document-uid291340labid2647timestamp1489325928669.png/wm)

# 为了方便同学们进行实验，我们已经将数据全部都保存成 _csv_ 文件上传至实验楼的云环境中。在后续的**代码实现**小节里，我们将给出获取这些文件的地址。
# 
# ## 三、数据分析
# 
# 在获取到数据之后，我们将利用每支队伍**过去的比赛情况**和 **Elo 等级分**来判断每支比赛队伍的可胜概率。在评价到每支队伍过去的比赛情况时，我们将使用到 **Team Per Game Stats**，**Opponent Per Game Stats** 和 **Miscellaneous Stats**（之后简称为 **T、O 和 M** 表）这三个表格的数据，作为代表比赛中某支队伍的比赛特征。我们最终将实现针对每场比赛，预测比赛中哪支队伍最终将会获胜，但并不是给出绝对的胜败情况，而是预判胜利的队伍有多大的获胜概率。因此我们将建立一个代表比赛的**特征向量**。由两支队伍的以往比赛情况统计情况（**T、O 和Ｍ**表），和两个队伍各自的 **Elo** 等级分构成。
# 
# 关于 **Elo score** 等级分，不知道同学们是否看过`《社交网络》`这部电影，在这部电影中，**Mark**（主人公原型就是扎克伯格，FaceBook 创始人）在电影起初开发的一个美女排名系统就是利用其好友 **Eduardo** 在窗户上写下的排名公式，对不同的女生进行等级制度对比，最后 PK 出胜利的一方。
# 
# ![](https://dn-anything-about-doc.qbox.me/document-uid291340labid2647timestamp1489325997201.png)
# 
# 这条对比公式就是 **Elo Score 等级分**制度。Elo 的最初为了提供国际象棋中，更好地对不同的选手进行等级划分。在现在很多的竞技运动或者游戏中都会采取 Elo 等级分制度对选手或玩家进行等级划分，如足球、篮球、棒球比赛或 LOL，DOTA 等游戏。
# 
# 在这里我们将基于国际象棋比赛，大致地介绍下 Elo 等级划分制度。在上图中 **Eduardo** 在窗户上写下的公式就是根据`Logistic Distribution`计算 PK 双方（A 和 B）对各自的胜率期望值计算公式。假设 A 和 B 的当前等级分为 <math><semantics><mrow><msub><mi>R</mi><mi>A</mi></msub></mrow><annotation encoding="application/x-tex">R_A</annotation></semantics></math>R​A​​何 <math><semantics><mrow><msub><mi>R</mi><mi>B</mi></msub></mrow><annotation encoding="application/x-tex">R_B</annotation></semantics></math>R​B​​，则 A 对 B 的胜率期望值为：
# 
# ![](https://dn-anything-about-doc.qbox.me/document-uid291340labid2647timestamp1490108812711.png)
# 
# B 对 A 的胜率期望值为
# 
# ![](https://dn-anything-about-doc.qbox.me/document-uid291340labid2647timestamp1490108823190.png)
# 
# 如果棋手 A 在比赛中的真实得分 <math><semantics><mrow><msub><mi>S</mi><mi>A</mi></msub></mrow><annotation encoding="application/x-tex">S_A</annotation></semantics></math>S​A​​（胜 1 分，和 0.5 分，负 0 分）和他的胜率期望值 <math><semantics><mrow><msub><mi>E</mi><mi>A</mi></msub></mrow><annotation encoding="application/x-tex">E_A</annotation></semantics></math>E​A​​不同，则他的等级分要根据以下公式进行调整：
# 
# ![](https://dn-anything-about-doc.qbox.me/document-uid291340labid2647timestamp1490108835437.png)
# 
# 在国际象棋中，根据等级分的不同 K 值也会做相应的调整：
# 
# *   <math><semantics><mrow><mo>≥</mo><mn>2</mn><mn>4</mn><mn>0</mn><mn>0</mn></mrow><annotation encoding="application/x-tex">\ge2400</annotation></semantics></math>≥2400，K=16
# *   2100~2400 分，K=24
# *   <math><semantics><mrow><mo>≤</mo><mn>2</mn><mn>1</mn><mn>0</mn><mn>0</mn></mrow><annotation encoding="application/x-tex">\le2100</annotation></semantics></math>≤2100，K=32
# 
# 因此我们将会用以表示某场比赛数据的**特征向量**为（加入 A 与 B 队比赛）：[A 队 Elo score, A 队的 **T,O 和 M 表**统计数据，B 队 Elo score, B 队的 **T,O 和 M 表**统计数据]

# ## 四、基于数据进行模型训练和预测
# 
# ### 4.1 实验前期准备
# 
# 在本次实验环境中，我们将会使用到 python 的`pandas`，`numpy`，`scipy`和`sklearn`库，实验楼的在线环境中都已经具备，无需手动安装。

# 接下来，我们下载相应的数据文件并解压。

# **☞ 示例代码：**

# ```bash
# # 获取数据文件
# !wget http://labfile.oss.aliyuncs.com/courses/782/data.zip
# 
# # 安装 unzip
# apt-get install unzip
# 
# # 解压data压缩包并且删除该压缩包
# !unzip data.zip 
# !rm -r data.zip
# ```

# **☞ 动手练习：**

# 在`data`文件夹中，包含了 2015~2016 年的 NBA 数据 **T,O 和 M** 表，及经处理后的常规赛和挑战赛的比赛数据`2015~16result.csv`，这个数据文件是我们通过在`basketball-reference.com`的 2015-16 Schedule and result 的几个月份比赛数据中提取得到的，其中包括三个字段：
# 
# *   WTeam: 比赛胜利队伍
# *   LTeam: 失败队伍
# *   WLoc: 胜利队伍一方所在的为主场或是客场 另外一个文件就是`16-17Schedule.csv`，也是经过我们加工处理得到的 NBA 在 2016~2017 年的常规赛的比赛安排，其中包括两个字段：
# *   Vteam: 访问 / 客场作战队伍
# *   Hteam: 主场作战队伍
# 
# ### 4.2 代码实现
# 
# 下载完实验数据后，就可以正式开始实验了。
# 
# 首先，引入实验相关模块：

# **☞ 示例代码：**

# ```python
# # -*- coding:utf-8 -*-
# import pandas as pd
# import math
# import csv
# import random
# import numpy as np
# from sklearn import linear_model
# from sklearn.model_selection import cross_val_score
# ```

# **☞ 动手练习：**

# 设置回归训练时所需用到的参数变量：

# **☞ 示例代码：**

# ```python
# # 当每支队伍没有elo等级分时，赋予其基础elo等级分
# base_elo = 1600
# team_elos = {} 
# team_stats = {}
# X = []
# y = []
# folder = 'data' #存放数据的目录
# ```

# **☞ 动手练习：**

# 在最开始需要初始化数据，从 **T、O 和 M** 表格中读入数据，去除一些无关数据并将这三个表格通过`Team`属性列进行连接：

# **☞ 示例代码：**

# ```python
# # 根据每支队伍的Miscellaneous Opponent，Team统计数据csv文件进行初始化
# def initialize_data(Mstat, Ostat, Tstat):
#     new_Mstat = Mstat.drop(['Rk', 'Arena'], axis=1)
#     new_Ostat = Ostat.drop(['Rk', 'G', 'MP'], axis=1)
#     new_Tstat = Tstat.drop(['Rk', 'G', 'MP'], axis=1)
# 
#     team_stats1 = pd.merge(new_Mstat, new_Ostat, how='left', on='Team')
#     team_stats1 = pd.merge(team_stats1, new_Tstat, how='left', on='Team')
#     return team_stats1.set_index('Team', inplace=False, drop=True)
# ```

# **☞ 动手练习：**

# 获取每支队伍的`Elo Score`等级分函数，当在开始没有等级分时，将其赋予初始`base_elo`值：

# **☞ 示例代码：**

# ```python
# def get_elo(team):
#     try:
#         return team_elos[team]
#     except:
#         # 当最初没有elo时，给每个队伍最初赋base_elo
#         team_elos[team] = base_elo
#         return team_elos[team]
# ```

# **☞ 动手练习：**

# 定义计算每支球队的`Elo等级分`函数：

# **☞ 示例代码：**

# ```python
# # 计算每个球队的elo值
# def calc_elo(win_team, lose_team):
#     winner_rank = get_elo(win_team)
#     loser_rank = get_elo(lose_team)
# 
#     rank_diff = winner_rank - loser_rank
#     exp = (rank_diff  * -1) / 400
#     odds = 1 / (1 + math.pow(10, exp))
#     # 根据rank级别修改K值
#     if winner_rank < 2100:
#         k = 32
#     elif winner_rank >= 2100 and winner_rank < 2400:
#         k = 24
#     else:
#         k = 16
#     new_winner_rank = round(winner_rank + (k * (1 - odds)))
#     new_rank_diff = new_winner_rank - winner_rank
#     new_loser_rank = loser_rank - new_rank_diff
# 
#     return new_winner_rank, new_loser_rank
# ```

# **☞ 动手练习：**

# 基于我们初始好的统计数据，及每支队伍的 **Elo score** 计算结果，建立对应 2015~2016 年常规赛和季后赛中每场比赛的数据集（在主客场比赛时，我们认为主场作战的队伍更加有优势一点，因此会给主场作战队伍相应加上 100 等级分）：

# **☞ 示例代码：**

# ```python
# def  build_dataSet(all_data):
#     print("Building data set..")
#     X = []
#     skip = 0
#     for index, row in all_data.iterrows():
# 
#         Wteam = row['WTeam']
#         Lteam = row['LTeam']
# 
#         #获取最初的elo或是每个队伍最初的elo值
#         team1_elo = get_elo(Wteam)
#         team2_elo = get_elo(Lteam)
# 
#         # 给主场比赛的队伍加上100的elo值
#         if row['WLoc'] == 'H':
#             team1_elo += 100
#         else:
#             team2_elo += 100
# 
#         # 把elo当为评价每个队伍的第一个特征值
#         team1_features = [team1_elo]
#         team2_features = [team2_elo]
# 
#         # 添加我们从basketball reference.com获得的每个队伍的统计信息
#         for key, value in team_stats.loc[Wteam].iteritems():
#             team1_features.append(value)
#         for key, value in team_stats.loc[Lteam].iteritems():
#             team2_features.append(value)
# 
#         # 将两支队伍的特征值随机的分配在每场比赛数据的左右两侧
#         # 并将对应的0/1赋给y值
#         if random.random() > 0.5:
#             X.append(team1_features + team2_features)
#             y.append(0)
#         else:
#             X.append(team2_features + team1_features)
#             y.append(1)
# 
#         if skip == 0:
#             print('X',X)
#             skip = 1
# 
#         # 根据这场比赛的数据更新队伍的elo值
#         new_winner_rank, new_loser_rank = calc_elo(Wteam, Lteam)
#         team_elos[Wteam] = new_winner_rank
#         team_elos[Lteam] = new_loser_rank
# 
#     return np.nan_to_num(X), y
# ```

# **☞ 动手练习：**

# 最终在 main 函数中调用这些数据处理函数，使用 sklearn 的`Logistic Regression`方法建立回归模型：

# **☞ 示例代码：**

# ```python
# if __name__ == '__main__':
# 
#     Mstat = pd.read_csv(folder + '/15-16Miscellaneous_Stat.csv')
#     Ostat = pd.read_csv(folder + '/15-16Opponent_Per_Game_Stat.csv')
#     Tstat = pd.read_csv(folder + '/15-16Team_Per_Game_Stat.csv')
# 
#     team_stats = initialize_data(Mstat, Ostat, Tstat)
# 
#     result_data = pd.read_csv(folder + '/2015-2016_result.csv')
#     X, y = build_dataSet(result_data)
# 
#     # 训练网络模型
#     print("Fitting on %d game samples.." % len(X))
# 
#     model = linear_model.LogisticRegression()
#     model.fit(X, y)
# 
#     #利用10折交叉验证计算训练正确率
#     print("Doing cross-validation..")
#     print(cross_val_score(model, X, y, cv = 10, scoring='accuracy', n_jobs=-1).mean())
# ```

# **☞ 动手练习：**

# 最终利用训练好的模型在 16~17 年的常规赛数据中进行预测。
# 
# 利用模型对一场新的比赛进行胜负判断，并返回其胜利的概率：

# **☞ 示例代码：**

# ```python
# def predict_winner(team_1, team_2, model):
#     features = []
# 
#     # team 1，客场队伍
#     features.append(get_elo(team_1))
#     for key, value in team_stats.loc[team_1].iteritems():
#         features.append(value)
# 
#     # team 2，主场队伍
#     features.append(get_elo(team_2) + 100)
#     for key, value in team_stats.loc[team_2].iteritems():
#         features.append(value)
# 
#     features = np.nan_to_num(features)
#     return model.predict_proba([features])
# ```

# **☞ 动手练习：**

# 在 main 函数中调用该函数，并将预测结果输出到`16-17Result.csv`文件中：

# ```python
# # 利用训练好的model在16-17年的比赛中进行预测
# 
# print('Predicting on new schedule..')
# schedule1617 = pd.read_csv(folder + '/16-17Schedule.csv')
# result = []
# for index, row in schedule1617.iterrows():
#     team1 = row['Vteam']
#     team2 = row['Hteam']
#     pred = predict_winner(team1, team2, model)
#     prob = pred[0][0]
#     if prob > 0.5:
#         winner = team1
#         loser = team2
#         result.append([winner, loser, prob])
#     else:
#         winner = team2
#         loser = team1
#         result.append([winner, loser, 1 - prob])
# 
# with open('16-17Result.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(['win', 'lose', 'probability'])
#     writer.writerows(result)
#     print('done.')
# ```

# **☞ 动手练习：**

# 最后，我们实验 Pandas 预览生成预测结果文件`16-17Result.csv`文件：

# **☞ 示例代码：**

# ```python
# pd.read_csv('16-17Result.csv',header=0)
# ```

# **☞ 动手练习：**

# ## 五、总结
# 
# 在本节课程中，我们利用`Basketball-reference.com`的部分统计数据，计算每支 nba 比赛队伍的`Elo socre`，和利用这些基本统计数据评价每支队伍过去的比赛情况，并且根据国际等级划分方法`Elo Score`对队伍现在的战斗等级进行评分，最终结合这些不同队伍的特征判断在一场比赛中，哪支队伍能够占到优势。但在我们的预测结果中，与以往不同，我们没有给出绝对的正负之分，而是给出胜算较大一方的队伍能够赢另外一方的概率。当然在这里，我们所采用评价一支队伍性能的数据量还太少（只采用了 15~16 年一年的数据），如果想要更加准确、系统的判断，有兴趣的你当然可以从各种统计数据网站中获取到更多年份，更加全面的数据。结合不同的**回归**、**决策**机器学习模型，搭建一个更加全面，预测准确率更高的模型。在 [kaggle](https://www.kaggle.com/) 中有相关的篮球预测比赛项目，有兴趣的同学可尝试一下。
# 
# ## 六、参考阅读
# 
# *   [知乎：在哪能看到最全面细致的 NBA 数据统计](https://www.zhihu.com/question/19952290)
# *   [How I won my NCAA tournament bracket pool using machine learning](http://ftw.usatoday.com/2016/04/villanova-bracket-winner)
# 
# ## 七、课后习题
# 
# 本次课程中，我们只是利用了`scikit-learn`提供的`Logisitc Regression`方法进行回归模型的训练，你可否尝试`scikit-learn`中的其他机器学习方法，或者其他类似于`TensorFlow`的开源框架，结合我们所提供的数据集进行训练。若采用`Scikit-learn`中的方法，可参看实验楼的课程：[ebay 在线拍卖数据分析](https://www.shiyanlou.com/courses/714)。或是结合下图进行模型的尝试：
# 
# ![](https://dn-anything-about-doc.qbox.me/document-uid291340labid2647timestamp1489329035758.png)
# 
# ## 八、常见疑问解答补充
# 
# 在这里我们将对之前在文档中解释得比较模糊的部分做一点补充，之后有疑问的欢迎同学们在课程讨论区提出讨论。
# 
# `Q1：` 在生成训练集时，“将特征值随机分配在每场比赛数据的左右侧” 是什么意思？为什么要做如下的随机分配：
# 
# ![](https://dn-anything-about-doc.qbox.me/document-uid291340labid2647timestamp1491839921983.png)
# 
# `Ans:` 将胜利队伍和失败队伍的特征值随机分配到每场比赛数据的左右侧意思是，为了随机产生 [winTeam, loseTeam]（胜利队伍特征值在左侧，对应的 y 值标签为 0），[loseTeam, winTeam]（失败队伍在左侧， 对应的 y 值标签为 1）这样的训练样本。你也可以固定利用数据集前一半为 [winTeam, loseTeam]，后一半为 [loseTeam, winTeam] 这样来生成数据。只要保证两类数据的分布比较均衡，且在训练时随机得取到两类训练样本即可。
# 
# `Q2:` 为什么按照 **X: [winTeam, loseTeam]** 对应标签 **Y: 0**，**X: [loseTeam, winTeam]** 对应标签 **Y: 1**，这样的取法在后边进行预测 **[team1, team2]** 的比赛结果时，是否应该按照 <math><semantics><mrow><mi>p</mi><mi>r</mi><mi>o</mi><mi>b</mi><mo><</mo><mn>0</mn><mi mathvariant="normal">.</mi><mn>5</mn></mrow><annotation encoding="application/x-tex">prob<0.5</annotation></semantics></math>prob<0.5 时 **team1** 胜出，胜出概率为 <math><semantics><mrow><mn>1</mn><mo>−</mo><mi>p</mi><mi>r</mi><mi>o</mi><mi>b</mi></mrow><annotation encoding="application/x-tex">1-prob</annotation></semantics></math>1−prob，<math><semantics><mrow><mi>p</mi><mi>r</mi><mi>o</mi><mi>b</mi><mo>></mo><mn>0</mn><mi mathvariant="normal">.</mi><mn>5</mn></mrow><annotation encoding="application/x-tex">prob>0.5</annotation></semantics></math>prob>0.5 时 **team2** 胜出，胜出概率为 <math><semantics><mrow><mi>p</mi><mi>r</mi><mi>o</mi><mi>b</mi></mrow><annotation encoding="application/x-tex">prob</annotation></semantics></math>prob？
# 
# `Ans:` 可查看 [sklearn logistic regression api](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) 中有关`predict_prob`的函数定义。
# 
# `predict_prob`返回的是对于测试样本中跟不同类别的匹配程度（概率，Probability estimates）, 对于 **N** 个类别，`predict_prob`返回的则是一个 **N 维数组**，按照类别顺序对应这个测试样本分别属于这些类别的可能性。在实验中我取的是`predict_prob[0]`，所以是测试样本属于**类别 1[winTeam, loseTeam]** 的可能性。因此当返回的值大于`0.5`时，我们则认为是 **team1** 胜出，其概率为`predict_prob[0]`，否则是 **team2** 胜出，胜出概率为`1-predict_prob[0]`
# 
# * 本课程内容，由作者授权实验楼发布，未经允许，禁止转载、下载及非法传播。
