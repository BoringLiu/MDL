import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
# pd.set_option('future.no_silent_downcasting', True)

# 全局设置字体
rcParams['font.family'] = 'SimHei'  # 或 'SimSun'，视系统而定
rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块

data = pd.read_csv('DATA/mcdonalds.csv')

# 查看数据维度
print(data.shape)

# 查看数据信息
print(data.info())

# 查看各列缺失值
print(data.isna().sum())

# 查看重复值
print(data.duplicated().sum())


print('"Like"特征的唯一值：')
print(data['Like'].unique())
print('-'*50)
print('"VisitFrequency"特征的唯一值：')
print(data['VisitFrequency'].unique())

#！！！！！把"Like"特征转成数值型，有助于后续分析：（后续的聚类分析都会用到，一直保持运行即可）
like_mapping = {
    'I love it!+5': 5,
    '+4': 4,
    '+3': 3,
    '+2': 2,
    '+1': 1,
    '0': 0,
    '-1': -1,
    '-2': -2,
    '-3': -3,
    '-4': -4,
    'I hate it!-5': -5
}

data['Like'] = data['Like'].map(like_mapping)
# print(data.head())

######################################################################

# #描述性统计

print(data.describe(include='all'))

# yummy: 顾客是否觉得麦当劳的食物美味，Yes: 803 次, No: 650 次。
# convenient: 顾客是否觉得麦当劳方便，Yes: 1319 次, No: 134 次。
# spicy: 顾客是否觉得麦当劳的食物辣，Yes: 136 次, No: 1317 次。
# fattening: 顾客是否觉得麦当劳的食物使人发胖，Yes: 1260 次, No: 193 次。
# greasy: 顾客是否觉得麦当劳的食物油腻，Yes: 765 次, No: 688 次。
# fast: 顾客是否觉得麦当劳的服务快速，Yes: 1308 次, No: 145 次。
# cheap: 顾客是否觉得麦当劳便宜，Yes: 870 次, No: 583 次。
# tasty: 顾客是否觉得麦当劳的食物可口，Yes: 936 次, No: 517 次。
# expensive: 顾客是否觉得麦当劳昂贵，Yes: 520 次, No: 933 次。
# healthy: 顾客是否觉得麦当劳的食物健康，Yes: 289 次, No: 1164 次。
# disgusting: 顾客是否觉得麦当劳的食物令人厌恶，Yes: 353 次, No: 1100 次。
# Like: 对麦当劳的整体喜好评分，平均值为0.777, 最小值为-5, 最大值为5。
# Age: 平均年龄为44.60岁，最小年龄为18岁，最大年龄为71岁。
# VisitFrequency: 受访者光顾麦当劳的频率，以“每月一次”最多，共有439次。
# Gender: 受访者的性别，788名女性和665名男性。


# ###############################################################################

# 定义一个函数，在countplot上添加数量文本
def add_count_labels(ax):
    for p in ax.patches:
        height = int(p.get_height())
        ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom')


VisitFrequency_order = ['Never', 'Once a year', 'Every three months', 'Once a month', 'Once a week',
                        'More than once a week']

plt.figure(figsize=(20, 15))
plt.subplot(3, 4, 1)
sns.histplot(data['Age'], kde=True)
plt.title('受访者的年龄分布')
plt.xlabel('受访者的年龄')
plt.ylabel('人数')

plt.subplot(3, 4, (2, 4))
ax = sns.countplot(x=data['VisitFrequency'], order=VisitFrequency_order)
plt.title('受访者光顾麦当劳的频率分布')
plt.xlabel('受访者光顾麦当劳的频率')
plt.ylabel('人数')

plt.subplot(3, 4, 5)
gender_counts = data['Gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('受访者的性别占比')

plt.subplot(3, 4, (6, 8))
ax = sns.countplot(x=data['Like'])
plt.title('对麦当劳的整体喜好评分分布')
plt.xlabel('对麦当劳的整体喜好评分')
plt.ylabel('数量')
add_count_labels(ax)

plt.subplot(3, 4, (9, 12))
yes_no_vars = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap', 'tasty', 'expensive', 'healthy',
               'disgusting']

# 计算每个分类变量的频数
yes_no_counts = data[yes_no_vars].apply(lambda x: x.value_counts())
yes_no_counts = yes_no_counts.transpose()

# 绘制堆叠条形图
bar_plot = yes_no_counts.plot(kind='bar', stacked=True, ax=plt.gca(), color=['#1f77b4', '#ff7f0e'])
plt.title('二元分类变量分布情况')
plt.ylabel('频数')
plt.xlabel('特征')
plt.legend(loc='upper right')
plt.xticks(rotation=0)
# 在每个颜色段上标注商品名称和数量
for i, category in enumerate(yes_no_counts.index):
    y_offset = 0
    for item, count in yes_no_counts.loc[category].items():
        if count > 0:
            bar_plot.text(i, y_offset + count / 2, f'{count}', ha='center', va='center', fontsize=12, color='white',
                          fontweight='bold')
            y_offset += count

plt.tight_layout()
plt.show()

################################################################################

# 将性别和访问频率编码为数值
data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})
frequency_mapping = {
    'More than once a week': 5,
    'Once a week': 4,
    'Once a month': 3,
    'Every three months': 2,
    'Once a year': 1,
    'Never': 0
}
data['VisitFrequency'] = data['VisitFrequency'].map(frequency_mapping)

data = data.replace({'No': 0,'Yes': 1})

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop('Index',axis=1))

# 使用肘部法则来确定最佳聚类数
inertia = []
silhouette_scores = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=10).fit(data_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data_scaled, kmeans.labels_))

plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o')
plt.xlabel('聚类中心数目')
plt.ylabel('惯性')
plt.title('肘部法则图')

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o')
plt.xlabel('聚类中心数目')
plt.ylabel('轮廓系数')
plt.title('轮廓系数图')

plt.tight_layout()
plt.show()


####################################################################
# K_Means聚类分析

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)
kmeans_final = KMeans(n_clusters=4, random_state=15)
kmeans_final.fit(data_scaled)
# 获取聚类标签
cluster_labels = kmeans_final.labels_
# 将聚类标签添加到原始数据中以进行分析
data['Cluster'] = cluster_labels

# 计算每个特征在所有簇中心的方差，方差越大，说明该特征对区分不同簇的重要性越大。

# 获取聚类中心
cluster_centers = kmeans.cluster_centers_
# 计算特征的重要性：每个特征在所有簇中心的方差
feature_variances = np.var(cluster_centers, axis=0)

min_lenth = min(len(data.drop(['Index','Cluster'],axis=1).columns),len(feature_variances))
feature_variances = feature_variances[:min_lenth]

# 将特征及其方差放入DataFrame，并按方差降序排列
feature_importance = pd.DataFrame({
    'Feature': data.drop(['Index','Cluster'],axis=1).columns,
    'Variance': feature_variances
})

feature_importance = feature_importance[feature_importance['Feature'] != 'Cluster']
feature_importance = feature_importance.sort_values(by='Variance', ascending=False)
print(feature_importance)


#########################################################################################
# 四类受访者之间的对比
# 将分类变量的分布按聚类分组
def plot_bar_chart(feature, title, xlabel, ylabel, position, data, fig):
    feature_and_cluster = pd.crosstab(data['Cluster'], data[feature])
    feature_and_cluster_percent = feature_and_cluster.div(feature_and_cluster.sum(axis=1), axis=0) * 100
    ax = plt.subplot(2, 4, position)
    feature_and_cluster_percent[1].plot(kind='bar', ax=ax)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    # 添加数据标签
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.figure(figsize=(20, 10))

plot_bar_chart('convenient', '不同类型受访者认为麦当劳方便的比例', '顾客类别', '比例', 1, data, plt)

plot_bar_chart('spicy', '不同类型受访者认为麦当劳食物辛辣的比例', '顾客类别', '比例', 2, data, plt)

plot_bar_chart('fattening', '不同类型受访者认为麦当劳食物增肥的比例', '顾客类别', '比例', 3, data, plt)

plot_bar_chart('disgusting', '不同类型受访者认为麦当劳食物令人厌恶的比例', '顾客类别', '比例', 4, data, plt)

plot_bar_chart('yummy', '不同类型受访者认为麦当劳食物美味的比例', '顾客类别', '比例', 5, data, plt)

plot_bar_chart('tasty', '不同类型受访者认为麦当劳食物可口的比例', '顾客类别', '比例', 6, data, plt)

plot_bar_chart('expensive', '不同类型受访者认为麦当劳昂贵的比例', '顾客类别', '比例', 7, data, plt)

ax8 = plt.subplot(2, 4, 8)
sns.boxplot(x='Cluster', y='Like', data=data, ax=ax8)
plt.title('不同类型受访者的整体喜好评分')
plt.xlabel('顾客类别')
plt.ylabel('喜好评分')

plt.tight_layout()
plt.show()

#################################################################################################
# 斯皮尔曼相关性分析
def plot_spearmanr(data,features,title,wide,height):
    # 计算斯皮尔曼相关性矩阵和p值矩阵
    spearman_corr_matrix = data[features].corr(method='spearman')
    pvals = data[features].corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(len(data[features].columns))

    # 转换 p 值为星号
    def convert_pvalue_to_asterisks(pvalue):
        if pvalue <= 0.001:
            return "***"
        elif pvalue <= 0.01:
            return "**"
        elif pvalue <= 0.05:
            return "*"
        return ""

    # 应用转换函数
    pval_star = pvals.applymap(lambda x: convert_pvalue_to_asterisks(x))

    # 转换成 numpy 类型
    corr_star_annot = pval_star.to_numpy()

    # 定制 labels
    corr_labels = spearman_corr_matrix.to_numpy()
    p_labels = corr_star_annot
    shape = corr_labels.shape

    # 合并 labels
    labels = (np.asarray(["{0:.2f}\n{1}".format(data, p) for data, p in zip(corr_labels.flatten(), p_labels.flatten())])).reshape(shape)

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(height, wide), dpi=100, facecolor="w")
    sns.heatmap(spearman_corr_matrix, annot=labels, fmt='', cmap='coolwarm',
                vmin=-1, vmax=1, annot_kws={"size":10, "fontweight":"bold"},
                linecolor="k", linewidths=.2, cbar_kws={"aspect":13}, ax=ax)

    ax.tick_params(bottom=False, labelbottom=True, labeltop=False,
                left=False, pad=1, labelsize=12)
    ax.yaxis.set_tick_params(labelrotation=0)

    # 自定义 colorbar 标签格式
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(direction="in", width=.5, labelsize=10)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(["-1.00", "-0.50", "0.00", "0.50", "1.00"])
    cbar.outline.set_visible(True)
    cbar.outline.set_linewidth(.5)

    plt.title(title)
    plt.show()

    features = data.drop(['Index', 'Cluster'], axis=1).columns.tolist()
    plot_spearmanr(data, features, '各变量之间的斯皮尔曼相关系数热力图', 10, 13)