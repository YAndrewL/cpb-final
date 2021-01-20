#!/usr/bin/env python
# coding: utf-8

# ## 计算生物学 期末作业 第一题
# 对于给定的如下三条蛋白质序列，试采用SP评分函数，给出相应的最优比对路径，其中精确匹配（exact match）定义为3 分，错配（mismatch）为-1，且空位（gap）罚分为-2。
# 
# >d1g08b_ a.1.1.2 (B:) Hemoglobin, beta-chain {Cow (Bos taurus)}
# 
# mltaeekaavtafwgkvkvdevggealgrllvvypwtqrffesfgdlstadavmnnpkvk
# 
# ahgkkvldsfsngmkhlddlkgtfaalselhcdklhvdpenfkllgnvlvvvlarnfgke
# 
# ftpvlqadfqkvvagvanalahryh
# 
# >d1itha_ a.1.1.2 (A:) Hemoglobin {Innkeeper worm (Urechis caupo)}
# 
# gltaaqikaiqdhwflnikgclqaaadsiffkyltaypgdlaffhkfssvplyglrsnpa
# 
# ykaqtltvinyldkvvdalggnagalmkakvpshdamgitpkhfgqllklvggvfqeefs
# 
# adpttvaawgdaagvlvaamk
# 
# >d1cqxa1 a.1.1.2 (A:1-150) Flavohemoglobin, N-terminal domain {Alcaligenes eutrophus}
# 
# mltqktkdivkatapvlaehgydiikcfyqrmfeahpelknvfnmahqeqgqqqqalara
# 
# vyayaeniedpnslmavlkniankhaslgvkpeqypivgehllaaikevlgnaatddiis
# 
# awaqaygnladvlmgmeselyersaeqpgg
# 
# 

# ### 题目简析
# 
# 根据题目，对于多重序列比对，其SP(Some of Pairs)评分为对于比对结果的所有列的两两评分结果求和
# 特别地，对于第i列，我们需要对这N个序列的两两求和，所以第i列我们需要对 $C(N,2)=N(N-1)/2$ 个评分进行求和
# 对于本题，N=3，所以每列需要对3个评分进行求和。
# 
# SP的目标函数为：
# 
# $SP = \sum_{i}S(m_i) = \sum_{i} \sum_{k<j}s(m^{k}_i, m^{j}_i)$
# 
# 对于本题，我们需要最大化SP的值，对于每一对的评分计算如下：
# 
# s(a, -) = s(-, a) = gap score = -2 (gap penalty为常数的模式)
# 
# s(-, -) = 0
# 
# s(a, a) = 3 (exact match, 即正确匹配)
# 
# s(a, a) = -1 (mismatch, 即错误匹配)
# 
# 首先，引入numpy，并初始化三个序列：

# In[83]:


import numpy as np


# In[84]:


seq1 = 'mltaeekaavtafwgkvkvdevggealgrllvvypwtqrffesfgdlstadavmnnpkvkahgkkvldsfsngmkhlddlkgtfaalselhcdklhvdpenfkllgnvlvvvlarnfgkeftpvlqadfqkvvagvanalahryh'
seq2 = 'gltaaqikaiqdhwflnikgclqaaadsiffkyltaypgdlaffhkfssvplyglrsnpaykaqtltvinyldkvvdalggnagalmkakvpshdamgitpkhfgqllklvggvfqeefsadpttvaawgdaagvlvaamk'
seq3 = 'mltqktkdivkatapvlaehgydiikcfyqrmfeahpelknvfnmahqeqgqqqqalaravyayaeniedpnslmavlkniankhaslgvkpeqypivgehllaaikevlgnaatddiisawaqaygnladvlmgmeselyersaeqpgg'


# ### 初始化边界情况
# 
#  对于任意的三个序列，我们需要初始化边界情况，由于计算的是最大值，所以我们需要把所有便捷情况置为最小的可能值
#  
#  对于每一列，最小的可能值为-5 （一个错配 + 2个gap penalty = -5）
# 
#  因此对于任意三个序列，其最差的SP匹配得分为最长序列的长度乘以-5

# In[99]:


def init_score(i, j, k):
    return -5 * max(i, j, k)


# In[100]:


l1 = len(seq1)
l2 = len(seq2)
l3 = len(seq3)
init = init_score(l1, l2, l3)
# print(init)


# ### 技巧性处理 exact match 和 mismatch
# 
#  对于mismatch和exact match的计算，因为一个是-1分，1个是3分
#  
#  所以只需进行布尔判断两个字符是否相等，然后用 4 * TRUE/FALSE - 1 即可计算得分，以简化后面的共识过程
#  
#  两个字符的比对得分用score函数来表示

# In[102]:


def score(c1, c2):
    return(c1==c2) * 4 - 1


# ### 动态规划（Dynamic Programming）求解过程
# 
# 寻找最优解并记录路径
# 
# find_opt 函数为核心Dynamic programming算法:
# 
# (1) 用前面定义的init_score初始化边界情况，即 $opt[i,j,k]$ 的值
# 
# （2）使用Needleman-Wunsch算法进行求解，记录中间状态，同时记录达到中间状态的状态转移情况
# 
# $align[i,j,k]$ 记录了得到每个中间状态$opt[i,j,k]$ 最优解的状态转移情况
# 
# （3）整体的动态规划求解过程总共需要分为$2^{N}-1 = 7$ 种情况进行讨论来求解并记录状态转移情况：（我们用使得状态的记录和二级制保持一致）
# 
# 7 - case(1,1,1) ：三个序列都向前推动
# 
# opt[i,j,k] = opt[i-1,j-1,k-1] + score(s_1[i-1], s_2[j-1]) + score(s_1[i-1], s_3[k-1]) + score(s_2[j-1], s_3[k-1])
# 
# 6 - case(1,1,0) ：序列1和序列2向前推动，序列3为gap ’-‘
# 
# opt[i,j,k] = opt[i-1,j-1,k] + score(s_1[i-1], s_2[j-1]) + one_gap * 2
# 
# 5 - case(1,0,1) ：序列1和序列3向前推动，序列2为gap ’-‘
# 
# opt[i,j,k] = opt[i-1,j,k-1] + score(s_1[i-1], s_3[k-1]) + one_gap * 2
# 
# 3 - case(0,1,1) ：序列2和序列3向前推动，序列1为gap ’-‘
# 
# opt[i,j,k] = opt[i,j-1,k-1] + score(s_2[j-1], s_3[k-1]) + one_gap * 2
# 
# 4 - case(1,0,0) ：只有序列1向前推动，序列2和序列3位为gap ’-‘
# 
# opt[i,j,k] = opt[i-1,j,k] + two_gaps + one_gap * 2
# 
# 2 - case(0,1,0) ：只有序列2向前推动，序列1和序列3位为gap ’-‘
# 
# opt[i,j,k] = opt[i,j-1,k] + two_gaps + one_gap * 2
# 
# 1 - case(0,0,1) ：只有序列3向前推动，序列1和序列2位为gap ’-‘
# 
# opt[i,j,k] = opt[i,j,k-1] + two_gaps + one_gap * 2
# 

# In[105]:


def find_opt(s_1, s_2, s_3):
    exact_match = 3
    mismatch = -1
    one_gap = -2
    two_gaps = 0
    len1, len2, len3 = len(s_1), len(s_2), len(s_3)
    
    align = np.ones((len1 + 1, len2 + 1, len3 + 1)) * (-1)
    opt = np.zeros((len1 + 1, len2 + 1, len3 + 1))
    
    for i in range(0, len1 + 1):
        for j in range(0,len2 + 1):
            for k in range(0, len3 + 1):
                opt[i, j, k] = init_score(i, j, k)
                align[i, j, k] = -1
            
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            for k in range(1, len3 + 1):
                
                # case (1,1,1)
                if opt[i,j,k] < opt[i-1,j-1,k-1] + score(s_1[i-1], s_2[j-1]) + score(s_1[i-1], s_3[k-1]) + score(s_2[j-1], s_3[k-1]):
                    opt[i,j,k] = opt[i-1,j-1,k-1] + score(s_1[i-1], s_2[j-1]) + score(s_1[i-1], s_3[k-1]) + score(s_2[j-1], s_3[k-1])
                    align[i,j,k] = 7
                
                # case (1,1,0)
                if opt[i,j,k] < opt[i-1,j-1,k] + score(s_1[i-1], s_2[j-1]) + one_gap * 2:
                    opt[i,j,k] = opt[i-1,j-1,k] + score(s_1[i-1], s_2[j-1]) + one_gap * 2
                    align[i,j,k] = 6
                    
                # case (1,0,1)
                if opt[i,j,k] < opt[i-1,j,k-1] + score(s_1[i-1], s_3[k-1]) + one_gap * 2:
                    opt[i,j,k] = opt[i-1,j,k-1] + score(s_1[i-1], s_3[k-1]) + one_gap * 2
                    align[i,j,k] = 5
                    
                # case (0,1,1)
                if opt[i,j,k] < opt[i,j-1,k-1] + score(s_2[j-1], s_3[k-1]) + one_gap * 2:
                    opt[i,j,k] = opt[i,j-1,k-1] + score(s_2[j-1], s_3[k-1]) + one_gap * 2
                    align[i,j,k] = 3
                    
                # case (1,0,0)
                if opt[i,j,k] < opt[i-1,j,k] + two_gaps + one_gap * 2:
                    opt[i,j,k] = opt[i-1,j,k] + two_gaps + one_gap * 2
                    align[i,j,k] = 4
                    
                # case (0,1,0)
                if opt[i,j,k] < opt[i,j-1,k] + two_gaps + one_gap * 2:
                    opt[i,j,k] = opt[i,j-1,k] + two_gaps + one_gap * 2
                    align[i,j,k] = 2
                    
                # case (0,0,1)
                if opt[i,j,k] < opt[i,j,k-1] + two_gaps + one_gap * 2:
                    opt[i,j,k] = opt[i,j,k-1] + two_gaps + one_gap * 2
                    align[i,j,k] = 1
                
    return opt, align


# In[106]:


s1 = seq1
s2 = seq2
s3 = seq3

# s1 = 'a'
# s2 = 'b'
# s3 = 'abc'

len1 = len(s1)
len2 = len(s2)
len3 = len(s3)


# ### 使用 find_opt 函数求解，最优的得分为 -123

# In[107]:


(opt,align) = find_opt(s1,s2,s3)
print(opt[len1, len2, len3])


# ### 输出路径
# 
# 打印完整路径，从最后的状态往前倒推， path字符串记录了从最开始的状态$[0,0,0]$ 到 $[len_{s1}, len_{s2}, len_{s3}]$的完整路径
# 
# index1,index2,index3分别标记了s1,s2,s3三个序列的目前回溯到的位置
# 
# 每次添加路径，往path的最前面插入路径，直到index1,index2,index3同时回溯到0

# In[108]:


index1 = len1
index2 = len2
index3 = len3
path = ''
while index1 + index2 + index3 > 0:
    if align[index1, index2, index3] == 7:
        path = '7' + path
        index1 = index1 - 1
        index2 = index2 - 1
        index3 = index3 - 1
    elif align[index1, index2, index3] == 6:
        path = '6' + path
        index1 = index1 - 1
        index2 = index2 - 1   
    elif align[index1, index2, index3] == 5:
        path = '5' + path
        index1 = index1 - 1
        index3 = index3 - 1
    elif align[index1, index2, index3] == 3:
        path = '3' + path
        index2 = index2 - 1
        index3 = index3 - 1
    elif align[index1, index2, index3] == 4:
        path = '4' + path
        index1 = index1 - 1
    elif align[index1, index2, index3] == 2:
        path = '2' + path
        index2 = index2 - 1
    elif align[index1, index2, index3] == 1:
        path = '1' + path
        index3 = index3 - 1
    else:
        path = '0' + path

print(path)


# ### 根据路径还原到比对结果
# 需要对每一个不同的路径值判断情况并进行还原

# In[109]:


align_s1 = ''
align_s2 = ''
align_s3 = ''
index1 = 0
index2 = 0
index3 = 0
len_path = len(path)

for i in range(0, len_path):
    if path[i] == '7':
        align_s1 = align_s1 + s1[index1]
        align_s2 = align_s2 + s2[index2]
        align_s3 = align_s3 + s3[index3]
        index1 = index1 + 1
        index2 = index2 + 1
        index3 = index3 + 1
    
    elif path[i] == '6':
        align_s1 = align_s1 + s1[index1]
        align_s2 = align_s2 + s2[index2]
        align_s3 = align_s3 + '-'
        index1 = index1 + 1
        index2 = index2 + 1
    
    elif path[i] == '5':
        align_s1 = align_s1 + s1[index1]
        align_s2 = align_s2 + '-'
        align_s3 = align_s3 + s3[index3]
        index1 = index1 + 1
        index3 = index3 + 1
        
    elif path[i] == '3':
        align_s1 = align_s1 + s1[index1]
        align_s2 = align_s2 + s2[index2]
        align_s3 = align_s3 + '-'
        index2 = index2 + 1
        index3 = index3 + 1
        
    elif path[i] == '4':
        align_s1 = align_s1 + s1[index1]
        align_s2 = align_s2 + '-'
        align_s3 = align_s3 + '-'
        index1 = index1 + 1
        
    elif path[i] == '2':
        align_s1 = align_s1 + '-'
        align_s2 = align_s2 + s2[index2]
        align_s3 = align_s3 + '-'
        index2 = index2 + 1
    
    elif path[i] == '1':
        align_s1 = align_s1 + '-'
        align_s2 = align_s2 + '-'
        align_s3 = align_s3 + s3[index3]
        index3 = index3 + 1
        
    else:
        align_s1 = align_s1 + '-'
        align_s2 = align_s2 + '-'
        align_s3 = align_s3 + '-'


# In[114]:


# print(align_s1[:100])
# print(align_s2[:100])
# print(align_s3[:100])

print(align_s1)
print(align_s2)
print(align_s3)
print(len(align_s1))
print(len(align_s2))
print(len(align_s3))


# ### 将比对结果输出到文件中 -- 问题（1）答案
# 
# 序列1 的比对结果输出在：question1_align.txt 中

# In[113]:


align_output = open("question1_align.txt", "w").write(str(align_s1) + '\n' + str(align_s2) + '\n' + str(align_s3))

