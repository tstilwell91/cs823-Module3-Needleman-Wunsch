# cs823-Module3-Needleman-Wunsch

## Option 1: Jupyter Notebook
You can run this code interactively using the Jupyter notebook. The only dependency is numpy. I ran this from Ondemand using the Waterfield HPC cluster. 

Steps:
1. Login using your ODU credentials at ondemand.waterfield.hpc.odu.edu
2. Click Interactive Apps -> Jupyter Notebook
3. Options: Python 3.10, pytorch 2.8.0 (numpy already included), partition cpu-2
4. Click Launch
5. Download Neddleman-Wunsch.ipynb from the git repo and copy to Waterfield. 
6. Execute the cells top to bottom. 

This code has two modes. If use_pam250 is set to false, then it expects user defined values for match_score, mismatch_score, and gap_penalty. If set to true, then gap_penalty is still required but it will use the PAM250 matrix for scoring matches and mismatches. 

For validation, I ran the simple matching example using the same sequences in the lecture, where S1:ACTCG and S2:ACAGTAG.

For PAM250 validation, I used the same sequences from the homework assignment and checked it against my results. S1:PRKVV and S2:DPLVR

## Option 2: Python Code
To run from the HPC clusters:
1. Download Neddleman-Wunsch.py to the cluster. 
2. Setup environment by running: module load pytorch-gpu/2.8.0
3. Run code: crun python3 Neddleman-Wunsch.py --arguments
4. View results from terminal

Sample Run using simple mode:
crun python3 Neddleman-Wunsch.py --seq1 ACTCG --seq2 ACAGTAG --gap -1 --simple --match 1 --mismatch 0
```
Optimal alignment score: 2
AC--TCG
ACAGTAG
```

Sample Run using pam250 mode:
crun python3 Neddleman-Wunsch.py --seq1 PRKVV --seq2 DPLVR --gap -2 --pam250
```
Optimal alignment score: 4
-PRKVV-
DP--LVR
```
