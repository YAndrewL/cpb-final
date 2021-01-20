# -*- coding: utf-8 -*-
# @Time    : 2021/1/9
# @Author: Computational Biology group 2


from friend import FriendClass
import numpy as np
from pickle import loads, dumps
import warnings

warnings.filterwarnings('ignore')

my_deepcopy = lambda x: loads(dumps(x))


class NeedlemanWunsch:

    def buildMatrices(self, s1, s2, subst_matrix_fn, gap_cost):
        s1_length = len(s1)
        s2_length = len(s2)
        nw_matrix = np.zeros((s1_length + 1, s2_length + 1), dtype=int)  # initialize a matrix size
        traceback = np.zeros((s1_length, s2_length),
                             dtype=int)  # use a traceback matrix to memorize which position it came from

        fr = FriendClass()

        for i in range(1, s1_length + 1):
            nw_matrix[i, 0] = nw_matrix[i - 1, 0] + gap_cost
        for j in range(1, s2_length + 1):
            nw_matrix[0, j] = nw_matrix[0, j - 1] + gap_cost
        for i in range(1, s1_length + 1):
            for j in range(1, s2_length + 1):
                seq1_gap = nw_matrix[i, j - 1] + gap_cost
                seq2_gap = nw_matrix[i - 1, j] + gap_cost
                substitution = nw_matrix[i - 1, j - 1] + fr.getSubsMatScore(s1[i - 1], s2[j - 1], subst_matrix_fn,
                                                                            gap_cost)

                nw_matrix[i][j] = max(seq1_gap, seq2_gap, substitution)
                if seq1_gap == max(seq1_gap, seq2_gap, substitution):
                    traceback[i - 1][j - 1] += 1
                if seq2_gap == max(seq1_gap, seq2_gap, substitution):
                    traceback[i - 1][j - 1] += 2
                if substitution == max(seq1_gap, seq2_gap, substitution):
                    traceback[i - 1][j - 1] += 4
                    
        traceback[(traceback == 5) | (traceback == 6) | (traceback == 7)] = 4
        traceback[traceback == 3] = 1
        optimalScore = nw_matrix[s1_length][s2_length]
        return traceback, optimalScore

    def getAlignmentsFromTracebacks(self, s1, s2, traceback):
        indices_list = [[]]
        trace_list = [[]]
        i = len(traceback) - 1
        j = len(traceback[0]) - 1

        indices_list[0] = [i, j]
        trace_list[0] = ["", "", ""]
        indices_duplicate = my_deepcopy(indices_list) 
        while True:
            completed_counter = 0  # This counter will be set to the number of tracebacks found.
            for index, [i, j] in enumerate(indices_duplicate):

                if i == -1 and j == -1:
                    # We reach here only when we have got the complete sequence
                    completed_counter +=1 #increment indicates that we have got 1 more complete traceback
                    continue

                if i ==-1 and j >= 0:
                    # We reach here only when s1 has reached the beginning of the sequence
                    trace_list[index][0] += '-'
                    trace_list[index][1] += s2[j]
                    trace_list[index][2] += ' '
                    indices_list[index][1] -= 1
                    continue

                if i >= 0 and j == -1:
                    # We reach here only when s2 has reached the beginning of the sequence
                    trace_list[index][0] += s1[i]
                    trace_list[index][1] += '-'
                    trace_list[index][2] += ' '
                    indices_list[index][0] -= 1
                    continue

                if traceback[i][j] == 1:                #case 1: we get the traceback from the left
                    trace_list[index][0] += '-'
                    trace_list[index][1] += s2[j]
                    trace_list[index][2] += ' '
                    # index of i stays as it is. Check if index of j is already less than 0. If yes, don't do anything, otherwise decrement it.

                    if indices_list[index][1] >= 0:
                        indices_list[index][1] -= 1

                elif traceback[i][j] == 2:                #case 2: we get the traceback from top
                    trace_list[index][0] += s1[i]
                    trace_list[index][1] += '-'
                    trace_list[index][2] += ' '
                    # index of j stays as it is. Check if index of i is already less than 0. If yes, don't do anything, otherwise decrement it.

                    if indices_list[index][0] >= 0:
                        indices_list[index][0] -= 1


                elif traceback[i][j] == 3:                #case 3: we get 2 tracebacks: from top and left
                    # we need to split the traceback sublist (and indices sublist) into 2 equal lists. deepcopy is used, because the
                    # normal shallow copy will result in both copies being updated whenever one is updated.
                    trace_list.append(copy.deepcopy(trace_list[index]))
                    indices_list.append(copy.deepcopy(indices_list[index]))
                    # treat traceback[index] as the list where the traceback has come from the left (decrease j)
                    trace_list[index][0] += '-'
                    trace_list[index][1] += s2[j]
                    trace_list[index][2] += ' '

                    # treat traceback[second] as the list where the traceback has come from the top (decrease i)
                    # second will store the index of the newly duplicated list (it will always be at the end because that's how append works)
                    second = len(trace_list) - 1
                    trace_list[second][0] += s1[i]
                    trace_list[second][1] += '-'
                    trace_list[second][2] += ' '

                    if indices_list[index][1] >= 0:
                        indices_list[index][1] -= 1
                    if indices_list[second][0] >= 0:
                        indices_list[second][0] -= 1

                elif traceback[i][j] == 4:                #case 4: we get the traceback from the diagonal
                    trace_list[index][0] += s1[i]
                    trace_list[index][1] += s2[j]
                    if s1[i] == s2[j]:
                        trace_list[index][2] += '*'
                    else:
                        trace_list[index][2] += ':'

                    # index of j and i need to be decremented if they are not already less than 0

                    if indices_list[index][0] >= 0 and indices_list[index][1] >= 0:
                        indices_list[index][0] -= 1
                        indices_list[index][1] -= 1


                elif traceback[i][j] == 5:                #case 3: we get 2 tracebacks: from diagonal and left
                    # split trace_list and indices_list into 2 equal lists
                    trace_list.append(copy.deepcopy(trace_list[index]))
                    indices_list.append(copy.deepcopy(indices_list[index]))
                    # treat traceback[index] as the list where the traceback has come from the left (decrease j)
                    trace_list[index][0] += '-'
                    trace_list[index][1] += s2[j]
                    trace_list[index][2] += ' '

                    # treat traceback[second] as the list where the traceback has come from the diagonal (decrease i and j)
                    #second will store the index of the newly duplicated list (it will always be at the end because that's how append works)
                    second = len(trace_list) - 1
                    trace_list[second][0] += s1[i]
                    trace_list[second][1] += s2[j]
                    if s1[i] == s2[j]:
                        trace_list[second][2] += '*'
                    else:
                        trace_list[second][2] += ':'

                    if indices_list[index][1] >= 0:
                        indices_list[index][1] -= 1
                    if indices_list[second][0] >= 0 and indices_list[second][1] >= 0:
                        indices_list[second][0] -= 1
                        indices_list[second][1] -= 1

                elif traceback[i][j] == 6:                #case 6: we get 2 tracebacks: from top and diagonal
                    # split trace_list and indices_list into 2 equal lists
                    trace_list.append(copy.deepcopy(trace_list[index])) # we need to split the traceback sublist into 2 lists
                    indices_list.append(copy.deepcopy(indices_list[index]))
                    # treat traceback[index] as the list where the traceback has come from the top (decrease i)
                    trace_list[index][0] += s1[i]
                    trace_list[index][1] += '-'
                    trace_list[index][2] += ' '

                    # treat traceback[second] as the list where the traceback has come from the left (decrease j)
                    #second will store the index of the newly duplicated list (it will always be at the end because that's how append works)
                    second = len(trace_list) - 1
                    trace_list[second][0] += s1[i]
                    trace_list[second][1] += s2[j]
                    if s1[i] == s2[j]:
                        trace_list[second][2] += '*'
                    else:
                        trace_list[second][2] += ':'

                    if indices_list[index][1] >= 0:
                        indices_list[index][0] -= 1
                    if indices_list[second][0] >= 0 and indices_list[second][0] >= 0:
                        indices_list[second][0] -= 1
                        indices_list[second][1] -= 1

                elif traceback[i][j] == 7:                #case 6: we get 3 tracebacks: from top, left and diagonal
                    # split trace_list and indices_list into 3 equal lists
                    trace_list.append(copy.deepcopy(trace_list[index])) # we need to split the trace_list sublist into 3 lists
                    trace_list.append(copy.deepcopy(trace_list[index]))
                    indices_list.append(copy.deepcopy(indices_list[index])) #first copy
                    indices_list.append(copy.deepcopy(indices_list[index])) #second copy
                    # treat traceback[index] as the list where the traceback has come from the top (decrease i)
                    trace_list[index][0] += s1[i]
                    trace_list[index][1] += '-'
                    trace_list[index][2] += ' '

                    # treat traceback[second] as the list where the traceback has come from the left (decrease j)
                    #second will store the index of the newly duplicated list (it will always be at the end because that's how append works)
                    second = len(trace_list) - 1
                    trace_list[second][0] += '-'
                    trace_list[second][1] += s2[j]
                    trace_list[second][2] += ' '

                    # treat traceback[third] as the list where the traceback has come from the diagonal (decrease i and j)
                    third = len(trace_list) - 2
                    trace_list[third][0] += s1[i]
                    trace_list[third][1] += s2[j]
                    if s1[i] == s2[j]:
                        trace_list[third][2] += '*'
                    else:
                        trace_list[third][2] += ':'

                    if indices_list[index][0] >= 0:
                        indices_list[index][0] -= 1
                    if indices_list[second][1] >= 0:
                        indices_list[second][1] -= 1
                    if indices_list[third][0] >= 0 and indices_list[third][1] >= 0:
                        indices_list[third][0] -= 1
                        indices_list[third][1] -= 1

            # indices_duplicate, the for loop variable, needs to store the updated value of indices_list before the next loop starts
            indices_duplicate = my_deepcopy(indices_list)
            # when the number of indices (same as no. of tracebacks) is equal to the 'done counter', which is incremented once for
            # each traceback, we can break out of the while(True) infinite loop
            if completed_counter == len(indices_duplicate):
                break
        # As trace_list contains all the strings (S1, S2 and connect) in the opposite order, they need to be reversed.
        alignment_strings = [[string[::-1] for string in trace] for trace in trace_list]
        return alignment_strings
