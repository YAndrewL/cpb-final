# -*- coding: utf-8 -*-
# @Time    : 2021/1/9
# @Author: Computational Biology group 2


from needleman_wunsch import NeedlemanWunsch
from friend import FriendClass
from uwpgma import Xpgma
from Bio import Phylo
import matplotlib.pyplot as plt
import pylab
import random
import sys
from time import time


def nobracket(s):
    # 去掉字符串s中的括号和逗号
    return s.replace('(', '').replace(')', '').replace(',', '')


class FengDoolittle():
    def __init__(self):
        self.MultSeqAlign = []  # 存储多重序列比对结果
        self.aligned = {}  # 存储目前已经比对过的引导树的部分

    def pairwiseAlignment(self,s1,s2):
        # 创建对象
        nw = NeedlemanWunsch()
        gma = Xpgma()
        # 计算NeedlemanWunsch矩阵回溯矩阵
        traceback, optimalScore = nw.buildMatrices(s1,s2,self.subsMat,self.gapOpenCost)
        # 根据回溯矩阵进行比对
        alignment_strings = nw.getAlignmentsFromTracebacks(s1,s2,traceback)
        num_alignments = len(alignment_strings)
        # 根据回溯矩阵可以得到多条路径 随机选取一个进行计算
        randomNum = random.randint(0,num_alignments-1)
        alignment = alignment_strings[randomNum]
        # 根据相似性计算距离 作业中要求使用的公式计算在下面这个函数里面
        distance = gma.distanceMatrixGenerate(optimalScore,s1,s2,nw,alignment,self.subsMat,self.gapOpenCost)
        return distance,alignment

    def addEquivalentGaps(self,s1,s2):
        # 在s2中 s1有空位的位置 加上相同的空位
        for i,c in enumerate(s1):
            if c == '-':
                s2 = s2[:i] + c + s2[i:]
        return s2

    def alignAndCombineGroups(self,group1,group2):
        # 两组序列之间的比对 合并这两组序列
        # 创建对象
        nw=NeedlemanWunsch()
        gma=Xpgma()
        minDistAlignment = ["","",""]
        minDist = 1.1 # 距离矩阵中的元素均在0~1之间
        # 两组的序列之间 两两比对 记录距离最小的两个序列
        for id1,seq1 in enumerate(group1):
            for id2,seq2 in enumerate(group2):
                dist,alignment = self.pairwiseAlignment(seq1,seq2)
                del alignment[2]
                if dist < minDist:
                    minDist = dist
                    minDistAlignment = alignment
                    minIndex1 = id1
                    minIndex2 = id2
        # 保存距离最小的比对结果
        group1[minIndex1] = minDistAlignment[0]
        group2[minIndex2] = minDistAlignment[1]
        # 给同组序列相同位置加上空位
        for id1,seq1 in enumerate(group1):
            if id1 != minIndex1:
                changed1 = self.addEquivalentGaps(group1[minIndex1],seq1)
                group1[id1] = changed1
        for id2,seq2 in enumerate(group2):
            if id2 != minIndex2:
                changed2  = self.addEquivalentGaps(group2[minIndex2],seq2)
                group2[id2] = changed2
        # 合并两组序列
        group1.extend(group2)
        return group1

    def createGroups(self,groups,seqClusterMap):
        # 把一个字符串分组处理
        if groups.count('C') == 2: # 如果字符串中就包含两个序列
            groupList = groups[1:len(groups)-1].split(',') # 字符串分割
            g1 = seqClusterMap[groupList[0]] # 根据序号取序列
            g2 = seqClusterMap[groupList[1]]
            gp1=[g1]
            gp2=[g2]

            # 对两组序列进行比对合并
            combinedAlignment = self.alignAndCombineGroups(gp1,gp2)
            combinedAlignment = [ele.replace('-','X') for ele in combinedAlignment]

            # 如果当前多重序列比对结果为空 则直接将combinedAlignment添加到结果中
            if self.MultSeqAlign == []:
                for ele in combinedAlignment:
                    self.MultSeqAlign.append(ele)
                # 更新引导树键值对
                self.aligned.clear()
                self.aligned['({},{})'.format(groupList[0],groupList[1])] = self.MultSeqAlign
                return
            # 如果当前多重序列比对结果不为空
            else:
                # 将groupList的第2个值改成这个函数的输入 完整的两个
                groupList[1] = '({},{})'.format(groupList[0],groupList[1])
                # 第二组序列为刚刚两个序列的比对结果
                gp2 = combinedAlignment

                # 第一组序列为已有的多重序列比对结果
                gp1 = self.MultSeqAlign
                # groupList的第1个值为已经比对过的序列序号
                groupList[0] = ''.join([i for i in self.aligned.keys()])
                
                # 对已有的多重序列比对结果 和 刚刚比对出的2组序列结果 进行比对合并
                combinedAlignment = self.alignAndCombineGroups(gp1,gp2)
                combinedAlignment = [ele.replace('-','X') for ele in combinedAlignment]

                # 更新比对结果
                self.MultSeqAlign.clear()
                for ele in combinedAlignment:
                    self.MultSeqAlign.append(ele)
                # 更新引导树键值对
                self.aligned.clear()
                self.aligned['({},{})'.format(groupList[0],groupList[1])] = self.MultSeqAlign
                return

        if groups.count('C') > 2:  # 字符串中有多于两个序列
            # 如果这组已经比对过了就直接返回
            if self.aligned.get(groups) != None:
                return

            # 去掉左右括号
            s=groups
            for myl in range(len(s)):
                if s[myl]!='(' and s[myl]!=')':
                    break
            for myr in range(len(s)-1, -1, -1):
                if s[myr]!='(' and s[myr]!=')':
                    break
            s= s[myl:myr+1]

            groupList = []
            leftmostComma = groups.find(',')
            groupList.append(groups[1:leftmostComma]) # 最左边的序列编号
            groupList.append(groups[leftmostComma + 1:groups.rfind(')')]) # 剩下右边的序列编号
            for myk in self.aligned.keys():  # 最左边的1个是新加的
                if nobracket(groupList[1]) in nobracket(myk) and not nobracket(groupList[0]) in nobracket(myk):
                    g1 = seqClusterMap[nobracket(groupList[0])]  # 取出最左边编号对应的序列
                    gp1=[g1]  # 第一组是这个新加的
                    gp2=self.aligned.get(myk)  # 第二组是已有的
                    # 这两组的比对合并
                    combinedAlignment = self.alignAndCombineGroups(gp1,gp2)
                    combinedAlignment = [ele.replace('-','X') for ele in combinedAlignment]
                    
                    # 更新已有比对结果
                    self.MultSeqAlign.clear()
                    for ele in combinedAlignment:
                        self.MultSeqAlign.append(ele)
                    # 更新引导树键值对
                    groupList[1]=''.join([i for i in self.aligned.keys()])  # wjy
                    self.aligned.clear()
                    self.aligned['({},{})'.format(groupList[0],groupList[1])] = self.MultSeqAlign
                    return

            groupList = []
            rightMostComma = groups.rfind(',')
            groupList.append(groups[1:rightMostComma])
            groupList.append(groups[rightMostComma + 1:groups.rfind(')')])
            for myk in self.aligned.keys():  # 最右边的1个是新加的 大概语义同上 不再注释说明
                if nobracket(groupList[0]) in nobracket(myk) and not nobracket(groupList[1]) in nobracket(myk):
                    g2 = seqClusterMap[nobracket(groupList[1])]
                    gp2=[g2]
                    gp1 = self.aligned.get(myk)
                    combinedAlignment = self.alignAndCombineGroups(gp1,gp2)
                    combinedAlignment = [ele.replace('-','X') for ele in combinedAlignment]

                    self.MultSeqAlign.clear()
                    for ele in combinedAlignment:
                        self.MultSeqAlign.append(ele)
                    groupList[0]=''.join([i for i in self.aligned.keys()])  # wjy
                    self.aligned.clear()
                    self.aligned['({},{})'.format(groupList[0],groupList[1])] = self.MultSeqAlign
                    return

    def processNewickString(self,newick,seqClusterMap):
        # 处理引导树
        openBrackets = closeBrackets = 0
        openInd = []
        closeInd = []
        for i,item in enumerate(newick):
            if item == '(': # 一组开始
                openBrackets += 1
                openInd.append(i)
            if item == ')': # 一组结束
                closeBrackets += 1
                openBrackets -= 1
                start = openInd[openBrackets]
                end = i
                groups = newick[start:end+1]

                del(openInd[openBrackets])
                ifcontinue=False
                for myk in self.aligned.keys():
                    if nobracket(groups) in nobracket(myk):
                        ifcontinue=True # 如果这组已经比对过了 就不需要重新createGroups了
                if ifcontinue:
                    continue
                self.createGroups(groups,seqClusterMap)  # 分组进行比对

    def outputGenerator(self, newick):
        """ This function prints the multiple sequence alignment and its sum of
        pairs score."""
        with open("guide_tree.txt", 'w') as f:
            f.write(newick)
            f.close()
        guideTree = Phylo.read("guide_tree.txt", "newick").as_phyloxml()
        fig = plt.figure(figsize=(10, 20), dpi=100)
        axes = fig.add_subplot(1, 1, 1)
        Phylo.draw(guideTree, axes=axes)
        plt.savefig('guide_tree_plot.pdf', dpi=100)
        print("The guide tree in Newick format is written in 'guide_tree.txt', and plot is saved as 'guide_tree_plot.pdf'.")

        self.MultSeqAlign = [ele.replace('X', '-') for ele in self.MultSeqAlign]
        with open("sequence_alignment.txt", 'w') as f:
            for i, ele in enumerate(self.MultSeqAlign):
                f.write('{} <- Sequence {}'.format(ele, i + 1))
                f.write('\n')
            f.close()
        print("Sequence alignment result is written in 'sequence_alignment.txt.'")

    def run(self,seq_fasta_file,cost_gap_open,subst_matrix_fn='pam250',clustering='UPGMA'):

        self.subsMat = subst_matrix_fn
        self.gapOpenCost = cost_gap_open
        
        # 读取fasta文件并处理            
        lines=open(seq_fasta_file, 'r').readlines()
        ones = ''
        ids, s = [], []
        i = 0
        for line in lines:
            if line.find('>') == 0:
                if ones!= '':
                    i += 1
                    ids.append('ID%d'%i)
                    s.append(ones.upper())
                    ones = ''
                continue
            ones += line[:-1]
        ids.append('ID%d'%(i+1))
        s.append(ones.upper())
        num_sequences=len(s)
        print('Parse amino acid sequences, running time: %.2f mins.' % ((time() - tstart) / 60))

        # 创建引导树
        gma = Xpgma()
        newick, newickNoDistance, distanceMatrix, newickIds = gma.UandWpgma(
            ids, s, num_sequences, subst_matrix_fn,cost_gap_open, clustering, tstart)
        # 键为序列的编号 值为序列字符串
        seqClusterMap = {}
        cl = 0
        for seq in s:
            seqClusterMap['C' + str(cl)] = seq
            cl += 1
        print('Cluster from paired alignments, running time: %.2f mins.' % ((time() - tstart) / 60))
        
        # 处理引导树字符串 对比合并
        self.processNewickString(newickNoDistance, seqClusterMap)
        self.outputGenerator(newickIds)
        print('All processes done, running time: %.2f mins.' % ((time() - tstart) / 60))
        return newick, newickNoDistance


if __name__ == '__main__':
    tstart = time()
    fd = FengDoolittle()
    fasta_file='mult.fasta'
    fasta_file='P1_fd.fasta'
    fd.run(fasta_file, cost_gap_open=-1)
