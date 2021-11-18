 Cover: unsupervised training set sampling and compression using a diversity strategy
 Copyright Rodrigo Silva (rmsilva @ dcc.ufmg.br) 2018
 Parameters are: original training set, emst file, size of selection (optional: "noqid" to indicate that this is a classification training set)
 OUTPUT: file named "selectedset-SELSIZE.txt" written to current directory and containing the selected instances
 
 EXAMPLE:
 #> ./cover.pl train.txt emst.csv 5000 
 where ẗrain.txt is the original training set (in LETOR/MSLR10K format), "emst.csv" is the output of the "emst" program (See below) and "5000" is the desired number of instances in the final selected set.

 
More on the input files:
 
 TRAINING FILE
 each line in the training set must be in the following (LETOR) format "LABEL QID:QIDNUM FEAT1:VAL1 FEAT2:VAL2 ... FEATM:VALM". For classification training sets, add a "noqid" parameter to the command line (i.e. "./cover.pl train.txt emst.csv 5000 noqid"). The format is similar, except that the "QID:QIDNUM" item in the line is missing.

 EXAMPLE TRAINING FILE LINE (L2R):
 0 qid:1 1:0.174545 2:0.083333 3:0.062500 4:0.000000 5:0.176044 6:0.000000 7:0.000000 8:0.000000 9:0.000000 10:0.000000 11:0.178009 12:0.083333 13:0.060569 14:0.000000 15:0.179183 16:0.084787 17:0.095238 18:0.054054 19:0.250000 20:0.085190 21:1.000000 22:0.282700 23:0.374494 24:0.000000 25:1.000000 26:0.987850 27:0.341306 28:0.183738 29:0.000000 30:0.984973 31:0.873810 32:0.009026 33:0.006893 34:0.000000 35:0.859789 36:1.000000 37:0.108855 38:0.112037 39:0.000000 40:1.000000 41:1.000000 42:1.000000 43:0.883207 44:0.960067 45:0.037611 46:1.000000 47:1.000000 48:0.417875 49:0.033155 50:0.000000 51:0.000050 52:0.369140 53:0.000000 54:0.023064 55:0.000000 56:0.002740 57:0.008475 58:0.428571 59:0.209790 60:0.000000 61:0.000000 62:0.000000 63:0.000000 64:0.000000

 EXAMPLE TRAINING FILE LINE (Classification): Same thing, with no "qid:QIDNUM"
 0 1:0.174545 2:0.083333 3:0.062500 4:0.000000 5:0.176044 6:0.000000 7:0.000000 8:0.000000 9:0.000000 10:0.000000 11:0.178009 12:0.083333 13:0.060569 14:0.000000 15:0.179183 16:0.084787 17:0.095238 18:0.054054 19:0.250000 20:0.085190 21:1.000000 22:0.282700 23:0.374494 24:0.000000 25:1.000000 26:0.987850 27:0.341306 28:0.183738 29:0.000000 30:0.984973 31:0.873810 32:0.009026 33:0.006893 34:0.000000 35:0.859789 36:1.000000 37:0.108855 38:0.112037 39:0.000000 40:1.000000 41:1.000000 42:1.000000 43:0.883207 44:0.960067 45:0.037611 46:1.000000 47:1.000000 48:0.417875 49:0.033155 50:0.000000 51:0.000050 52:0.369140 53:0.000000 54:0.023064 55:0.000000 56:0.002740 57:0.008475 58:0.428571 59:0.209790 60:0.000000 61:0.000000 62:0.000000 63:0.000000 64:0.000000

 EMST FILE
 emst file must contain the list of edges with ascending order by size and indexed by 0 (i.e. first instance in training file is numbered 0)

 EXAMPLE EMST FILE LINE:
 # instance X	     , instance Y	  , edge length
 1.89830000000000e+04,1.90430000000000e+04,0.00000000000000e+00

 EMST file can be created using the "emst" program in mlpack (https://github.com/mlpack/mlpack or http://www.mlpack.org)
 to run emst, take the training file and convert it into a CSV containing only feature values for each instance (one vector per line), then use the program to produce a list of EMST edges  in ascending order of length

 EXAMPLE of running EMST on a Comma separated file containing all unlabeled instances:
 #> ./emst --input_file=INPUT.csv --output_file=EMST-EDGES.txt -v
