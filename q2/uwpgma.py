# -*- coding: utf-8 -*-
# @Time    : 2021/1/9
# @Author: Computational Biology group 2

from needleman_wunsch import NeedlemanWunsch
from friend import FriendClass
import numpy as np
import pandas as pd
import random
import math
from decimal import Decimal
from time import time


class Xpgma:

    def distanceMatrixGenerate(self, s_ab, a, b, nw, alignment, subsMat, gapOpenCost):
        L = len(alignment[0]) 
        N_g = alignment[0].count('-') + alignment[2].count('-')
        fr = FriendClass()
        sum_xy = 0
        
        # s_ab_rand
        list_a = list(a)
        list_b = list(b)
        random.shuffle(list_a)
        random.shuffle(list_b)
        rand_a = "".join(list_a)
        rand_b = "".join(list_b)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                s_xy = fr.getSubsMatScore(rand_a[i], rand_b[j], subsMat, gapOpenCost)
                Na_x = a.count(x)
                Nb_y = b.count(y)
                sum_xy += (Na_x * Nb_y * s_xy)
        s_ab_rand = (sum_xy / L) + (N_g * gapOpenCost)

        # s_ab_max
        (traceback_aa, s_aa) = nw.buildMatrices(a, a, subsMat, gapOpenCost)
        (traceback_bb, s_bb) = nw.buildMatrices(b, b, subsMat, gapOpenCost)
        s_ab_max = (s_aa + s_bb) / 2

        # s_ab_eff
        s_ab_eff = (s_ab - s_ab_rand) / (s_ab_max - s_ab_rand)
        d = - math.log(s_ab_eff)
        return d

    def pairwiseAlignment(self, s1, s2, subsMat, gapOpenCost):
        nw = NeedlemanWunsch()
        (traceback, optimalScore) = nw.buildMatrices(s1, s2, subsMat, gapOpenCost)
        alignment_strings = nw.getAlignmentsFromTracebacks(s1, s2, traceback)
        num_alignments = len(alignment_strings)
        # Get only one random optimal alignment
        randomNum = random.randint(0, num_alignments - 1)
        alignment = alignment_strings[randomNum]  # alignment is a list: ['','','']
        distance = self.distanceMatrixGenerate(optimalScore, s1, s2, nw, alignment, subsMat, gapOpenCost)
        return distance

    def getMin(self, distanceMatrix):
        minimum = Decimal('inf')
        for i in range(len(distanceMatrix)):
            for j in range(len(distanceMatrix)):
                # We need to consider only the upper triangle matrix as it is symmetric.
                if i < j and distanceMatrix[i, j] < minimum:
                    # Round to 3 digits
                    minimum = Decimal(distanceMatrix[i, j]).quantize(Decimal('.010'))
                    row = i
                    col = j

        return row, col, minimum

    def updateMatrix(self, distanceMatrix, clustering, clust1, clust2, row, col):
        if clustering == "WPGMA":
            clust1 = clust2 = 1
        for j in range(0, len(distanceMatrix)):
            if j != col and j != row:
                distanceMatrix[row, j] = ((clust1 * distanceMatrix[row, j]) +
                                          (clust2 * distanceMatrix[col, j])) / (clust1 + clust2)
                distanceMatrix[j, row] = distanceMatrix[row, j]
        distanceMatrix = np.delete(distanceMatrix, col, axis=0)
        distanceMatrix = np.delete(distanceMatrix, col, axis=1)
        return distanceMatrix

    def updateMatrixAndClusters(self, c, c2, distanceMatrix, row, col, dmin, next_cluster, clustering, mapping,
                                idNewick):
        clust1 = len(c[row].split(','))
        clust2 = len(c[col].split(','))

        # Update the UPGMA/WPGMA distance matrix
        distanceMatrix = self.updateMatrix(distanceMatrix, clustering, clust1, clust2, row, col)

        if clust1 == 1 and clust2 == 1:
            added = '({}:{},{}:{})'.format(c[row], dmin / 2, c[col], dmin / 2)
            addedId = '({}:{},{}:{})'.format(idNewick[row], dmin / 2, idNewick[col], dmin / 2)
        elif clust1 == 1 and clust2 != 1:
            added = '({}:{},{}:{})'.format(c[row], dmin / 2, c[col], dmin / 2 - mapping[c[col]])
            addedId = '({}:{},{}:{})'.format(idNewick[row], dmin / 2, idNewick[col], dmin / 2 - mapping[c[col]])
        elif clust1 != 1 and clust2 == 1:
            added = '({}:{},{}:{})'.format(c[row], dmin / 2 - mapping[c[row]], c[col], dmin / 2)
            addedId = '({}:{},{}:{})'.format(idNewick[row], dmin / 2 - mapping[c[row]], idNewick[col], dmin / 2)
        elif clust1 != 1 and clust2 != 1:
            added = '({}:{},{}:{})'.format(c[row], dmin / 2 - mapping[c[row]], c[col], dmin / 2 - mapping[c[col]])
            addedId = '({}:{},{}:{})'.format(idNewick[row], dmin / 2 - mapping[c[row]], idNewick[col],
                                             dmin / 2 - mapping[c[col]])

        added2 = '({},{})'.format(c2[row], c2[col])

        mapping[added] = dmin / 2
        c[row] = added
        del c[col]

        c2[row] = added2
        del c2[col]

        idNewick[row] = addedId
        del idNewick[col]
        new_cluster = 'C{}'.format(str(next_cluster))
        return c, c2, distanceMatrix, idNewick

    def UandWpgma(self, ids, s, n, subsMat, gapOpenCost, clustering, tstart):
        c = []
        c2 = []

        idNewick = []
        cToIdDict = dict()
        for index, id in enumerate(ids):
            cToIdDict['C' + str(index)] = id

        for i in range(0, n):
            c.append('C' + str(i))
            c2.append('C' + str(i))
            idNewick.append(cToIdDict.get(c[i]))

        distanceMatrix = np.zeros((n, n))
        next_cluster = n
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                distanceMatrix[i, j] = self.pairwiseAlignment(s[i], s[j], subsMat, gapOpenCost)
                distanceMatrix[j, i] = distanceMatrix[i, j]
        distanceMatrix_pd = pd.DataFrame(distanceMatrix)
        xlswriter = pd.ExcelWriter('distance_matrix.xlsx')
        distanceMatrix_pd.to_excel(xlswriter)
        xlswriter.save()
        print("The distance matrix of alignments is written in 'distance_matirx.xlsx'.")
        mapping = dict()
        while len(c) > 1:
            row, col, dmin = self.getMin(distanceMatrix)
            distanceMatrix = np.round(distanceMatrix, 3)
            c, c2, distanceMatrix, idNewick = self.updateMatrixAndClusters(c, c2,
                                                                           distanceMatrix, row, col, dmin, next_cluster,
                                                                           clustering, mapping, idNewick)
            next_cluster += 1
        print('Clustering from distance matrix, running time: %.2f mins.' % ((time() - tstart) / 60))
        c += ';'
        c2 += ';'
        idNewick += ';'
        newick = ''.join([cluster for cluster in c])
        newickNoDistance = ''.join([cluster for cluster in c2])
        newickIds = ''.join([cluster for cluster in idNewick])
        return (newick, newickNoDistance, distanceMatrix, newickIds)

