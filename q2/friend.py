# -*- coding: utf-8 -*-
# @Time    : 2021/1/9
# @Author: Computational Biology group 2


import re
from Bio.SubsMat import MatrixInfo
from Bio import SeqIO
import os
import sys
import warnings
warnings.filterwarnings('ignore')


class FriendClass():
    """ This class contains functions to validate an amino acid sequence,
    to get the PAM250/BLOSUM62 substitution matrix score between
    2 amino acids, and to parse FASTA files."""

    def validateAminoSequence(self, sequence_list):
        """This function checks if any of the sequences in the passed list
        contain an invalid amino acid character."""

        # Create a list of invalid amino acid characters for later validation
        # all except the specified characters are invalid.
        invalid = re.compile(r'[^ARNDBCEQZGHILKMFPSTWYV]')
        for string in sequence_list:
            if re.search(invalid, string) is not None:
                return 0
        return 1

    def getSubsMatScore(self, amino1, amino2, subst_matrix_fn, gap_cost):
        """This function gets gets the match/mismatch score using the
        appropriate substitution matrix"""

        # use try-except loop to get interchange score
        if subst_matrix_fn == 'pam250':
            try:
                score = MatrixInfo.pam250[amino1, amino2]
            except KeyError:
                score = MatrixInfo.pam250[amino2, amino1]

        elif subst_matrix_fn == 'blosum62':
            try:
                score = MatrixInfo.blosum62[amino1, amino2]
            except KeyError:
                score = MatrixInfo.blosum62[amino2, amino1]
        return score

    def parseFastaFiles(self, seq1_fasta_file, seq2_fasta_file):
        """ This function takes the location of 2 Fasta files, builds absolute paths
        and retrieves them. It returns the strings and ids in both files in a 2 lists.
        This is meant for pairwise alignments, so only 1 record is expected per file"""
        # Parse the fasta files. Get 2 sequences out of them

        # split input file path by '/' in Unix based system
        file1_loc = seq1_fasta_file.split('/')
        file2_loc = seq2_fasta_file.split('/')
        # get the location from which the program was run
        pwd = os.path.dirname(os.path.abspath(__file__))
        # * unpacks the list
        file1_path = os.path.join(pwd, *file1_loc)
        file2_path = os.path.join(pwd, *file2_loc)

        try:
            record1 = list(SeqIO.parse(file1_path, "fasta"))
        except FileNotFoundError:
            print("Your first Fasta file doesn't exist. Exiting.")
            sys.exit(1)

        try:
            record2 = list(SeqIO.parse(file2_path, "fasta"))
        except FileNotFoundError:
            print("Your second Fasta file doesn't exist. Exiting.")
            sys.exit(1)

        return record1, record2

    def parseMultSequenceFastaFile(self, seq_fasta_fn):
        """ This function takes the location of a single fasta file, creates an
        absolute path, retrieves the ids and strings out of it, and puts them in
        a list, which is returned. Multiple ids/strings may be present in the file."""
        # Convert to absolute paths to ensure portability.
        file_loc = seq_fasta_fn.split('/')
        # get the location from which the program was run
        pwd = os.path.dirname(os.path.abspath(__file__))
        # * unpacks the list
        file_path = os.path.join(pwd, *file_loc)
        record = list(SeqIO.parse(file_path, "fasta"))
        return record
