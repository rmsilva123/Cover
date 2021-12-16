#!/usr/bin/perl

# Cover: unsupervised training set sampling and compression using a diversity strategy
# Copyright Rodrigo Silva (rmsilva @ dcc.ufmg.br) 2018
# Parameters are: original training set, emst file, size of selection (optional: "noqid" to indicate that this is a classification training set)
# OUTPUT: file named "selectedset-SELSIZE.txt" written to current directory and containing the selected instances
# EXAMPLE:
# ./cover.pl train.txt emst.csv 5000 
# where ẗrain.txt is the original training set, "emst.csv" is the output of the "emst" program (See below) and "5000" is the desired number of instances in the final selected set.

# TRAINING FILE
# each line in the training set must be in the following (LETOR) format "LABEL QID:QIDNUM FEAT1:VAL1 FEAT2:VAL2 ... FEATM:VALM". For classification training sets, add a "noqid" parameter to the command line (i.e. "./cover.pl train.txt emst.csv 5000 noqid"). The format is similar, except that the "QID:QIDNUM" item in the line is missing.

# EXAMPLE TRAINING FILE LINE (L2R):
# 0 qid:1 1:0.174545 2:0.083333 3:0.062500 4:0.000000 5:0.176044 6:0.000000 7:0.000000 8:0.000000 9:0.000000 10:0.000000 11:0.178009 12:0.083333 13:0.060569 14:0.000000 15:0.179183 16:0.084787 17:0.095238 18:0.054054 19:0.250000 20:0.085190 21:1.000000 22:0.282700 23:0.374494 24:0.000000 25:1.000000 26:0.987850 27:0.341306 28:0.183738 29:0.000000 30:0.984973 31:0.873810 32:0.009026 33:0.006893 34:0.000000 35:0.859789 36:1.000000 37:0.108855 38:0.112037 39:0.000000 40:1.000000 41:1.000000 42:1.000000 43:0.883207 44:0.960067 45:0.037611 46:1.000000 47:1.000000 48:0.417875 49:0.033155 50:0.000000 51:0.000050 52:0.369140 53:0.000000 54:0.023064 55:0.000000 56:0.002740 57:0.008475 58:0.428571 59:0.209790 60:0.000000 61:0.000000 62:0.000000 63:0.000000 64:0.000000

# EXAMPLE TRAINING FILE LINE (Classification): Same thing, with no "qid:QIDNUM"
# 0 1:0.174545 2:0.083333 3:0.062500 4:0.000000 5:0.176044 6:0.000000 7:0.000000 8:0.000000 9:0.000000 10:0.000000 11:0.178009 12:0.083333 13:0.060569 14:0.000000 15:0.179183 16:0.084787 17:0.095238 18:0.054054 19:0.250000 20:0.085190 21:1.000000 22:0.282700 23:0.374494 24:0.000000 25:1.000000 26:0.987850 27:0.341306 28:0.183738 29:0.000000 30:0.984973 31:0.873810 32:0.009026 33:0.006893 34:0.000000 35:0.859789 36:1.000000 37:0.108855 38:0.112037 39:0.000000 40:1.000000 41:1.000000 42:1.000000 43:0.883207 44:0.960067 45:0.037611 46:1.000000 47:1.000000 48:0.417875 49:0.033155 50:0.000000 51:0.000050 52:0.369140 53:0.000000 54:0.023064 55:0.000000 56:0.002740 57:0.008475 58:0.428571 59:0.209790 60:0.000000 61:0.000000 62:0.000000 63:0.000000 64:0.000000

# EMST FILE
# emst file must contain the list of edges with ascending order by size and indexed by 0 (i.e. first instance in training file is numbered 0)

# EXAMPLE EMST FILE LINE:
# 1.89830000000000e+04,1.90430000000000e+04,0.00000000000000e+00

# emst can be created using the "emst" program in mlpack (http://svn.cc.gatech.edu/fastlab/mlpack/tags/mlpack-1.0.9/build/bin/emst)
# to run emst, take the training file and convert it into a CSV containing only feature values for each instance, then use the program to produce a list of EMST edges 
# in ascending order of length

# EXAMPLE of running EMST on a Comma separated file containing all unlabeled instances:
# ./emst --input_file=INPUT.csv --output_file=EMST-EDGES.txt -v

use strict;
use warnings;
use Math::Random::ISAAC qw(rand);
use Heap::MinMax;
use DBI;

sub log2 {
	my $n = shift;
	return log($n)/log(2);
}

# calculates the euclidean distance between two points
sub Dist {
      my $arr1ref = shift;
      my $arr2ref = shift;
      
      my $i = 0;
      my $dist = 0;
      foreach my $p (@$arr1ref) {
	  my $diff = $p - @$arr2ref[$i];
	  $dist = $dist + ($diff*$diff);
	  $i++;
      }
      return $dist;
}

# command line parameters 
my $train_file = $ARGV[0]; # the original training data we want to select instances from
my $emstfile = $ARGV[1];   # the file containing edges that was calculated by "emst"
my $selsize = $ARGV[2];	   # the target selection size (number of instances to be selected)
my $hasqid = $ARGV[3];	   # "noqid" indicates a classification training set (no Query ID); leave blank for LETOR (L2R) style datasets
if (! defined $hasqid) { $hasqid = ""; } 
if (! defined $train_file || ! defined $emstfile || ! defined $selsize) { print "USAGE: \"cover.pl TRAINFILE EMSTFILE SELSIZE [noquid]\"\n"; exit 1; }

my $resultfolder = "./";
my $row = 0;
my @unlabeledset; # unlabeled set with only feature values (this makes it easier to calculate distances later on)
my %clustermap;	  # this hash makes it easier to know which cluster an instance belongs to during the cluster merge process
my @clusters;	  # an array of pointers to the clusters' list
my @last;	  # array containing a pointer to the last instance of each cluster (for easier cluster merging)
my @size;	  # array with the size of each cluster (makes it simple to decide the fastest way to merge 2 clusters; i.e. merge smaller cluster into larger one)
my @labels;	  # we keep an extra array with the labels
my @qids;	  # and the qids for each training set instance. 	

$|++;

my $file = "$train_file";
open (F1, $file) || die ("Could not open $file!"); 

print "loading unlabeled dataset...\n";
# $starttime = `echo \$(date +%s)`;

# read the training file and create arrays with the coordinates for each instance
while (my $linetrain = <F1>) {
# 	  $linetrain = <F1>;
	my $label; my $qid; my @features;
	if ($hasqid eq "noqid") {
		($label, @features) = split ' ', $linetrain; # Classification set. format "LABEL feat1:val1 feat2:val2 ... featm:valm"
	}
	else {
		($label, $qid, @features) = split ' ', $linetrain; # LETOR set: "LABEL QID feat1:val1 feat2:val2 ... featm:valm"
		# save the qid for this instance (to be able to "reconstruct" the instance after selection)
		push (@qids, $qid);
	}
	
	my @allfeat;
	for (my $i=0; $i<=$#features; $i++) { # each line must end with a feature or this loop will grab garbage from the end of the line.
		my ($fid, $val) = split ":", $features[$i];
		# array containing each feature value (coordinate)
		if (defined($val)) { # checks anyway if we have a valid fid:value pair
			push (@allfeat, $val); 
		} 
	}
	# array of arrays with every instance coordinates
	push (@unlabeledset, \@allfeat);
	# we keep the label
	push (@labels, $label);
	
	# Initially, each instance is a cluster. Here we initialize the arrays we need
	my $cluster = [$row, undef];
	push (@clusters, $cluster); # initialize cluster pointer array
	push (@last, $cluster);	    # initialize last instance pointer array
	push (@size, 1);	    # initialize cluster size array
	$clustermap{$row} = $row;   # map the current instance to its own cluster
	$row++;
	if ($row % 10000 == 0) { print "      $row docs processed\r"; }
}
close(F1);
print "Total instances $#clusters\n";

# a few control variables
my @distances; # contains the size of the last edge added to a cluster 
my $maxd = 0;
my $mind = 99999;
my $clustercount = keys %clustermap;
my %deletedclusters;
my $percent = 100;

open (F1, $emstfile) || die ("Could not open $emstfile!"); 

# main loop: go through emst file and merge clusters; emst file contains edeges sorted ascending by size
while (my $linedist = <F1>) {

	my ($inst1,$inst2,$dist) = split ',', $linedist;
	$inst1 = int($inst1); # these are the instances connected by the edge
	$inst2 = int($inst2);
	
	# checks if the two instances in the edge are ALREADY in the same cluster
	if ($clustermap{$inst1} ne $clustermap{$inst2}) {
		# if instances are in distinct clusters, merge these clusters together
		my $source;
		my $dest;
		
		# always append the SMALLER ($source) cluster to the LARGER ($dest) one
		if ($size[$clustermap{$inst1}] >= $size[$clustermap{$inst2}]) {
			$source = $clustermap{$inst2};
			$dest = $clustermap{$inst1};
		} else {
			$source = $clustermap{$inst1};
			$dest = $clustermap{$inst2};
		}
		
		# go through (smaller) cluster instances to update clustermap for each 
		my $current = $clusters[$source];
		while (defined $current) {
			$clustermap{$current->[0]} = $dest;
			$current = $current->[1];
		}
		
		# update pointer to last instance in list; update $dest cluster size
		$last[$dest]->[1] = $clusters[$source];
		$last[$dest] = $last[$source];
		$size[$dest] += $size[$source];

		# delete $source cluster reference
		$deletedclusters{$source} = 1;
 		undef($clusters[$source]);

	}
		
	my $currsize = ($#clusters+1) - (keys %deletedclusters);
# 	my $currsize = $#clusters;
	print "Current number of clusters: $currsize     \r";
	
	# do we have the target number of clusters (number of instances to be selected)?
	if ($currsize == $selsize) { 
# 		print "\n$dist";
		$dist =~ s/\n//;
		
		# the current edge size indicates the epsilon used in the hierarchical clustering
		# we keep this as additional information. This is not needed by Cover.
		push(@distances,$dist); 
		
		$percent = int($currsize*100/$#clusters);
		
		# Now we need to calculate the centroid for each cluster
		my @centroids;
# 			
		# we also keep a few statistics about the clusters 
		my @clustrel; 	# number of relevant instances (label > 0)
		
		# centroid calculation loop
		for (my $i = 0; $i<=$#clusters; $i++) {
			# only process remaining clusters
			if (defined $clusters[$i]) {
				# initialize cluster metrics
				if (! exists $clustrel[$i]) { $clustrel[$i] = 0; }
				
				# initialize centroid for current cluster
				my @zerocentroid;
				for (my $j=0; $j<=$#{$unlabeledset[0]}; $j++) {
					push(@zerocentroid,0);
				} 
				$centroids[$i] = \@zerocentroid;
				
				# calculate cluster metrics and cummulative centroid values 
				my $current = $clusters[$i];
				while (defined $current) {
					if ($labels[$current->[0]] > 0) {
						$clustrel[$i]++; # instance with label > 0
					}
					for (my $j=0; $j<=$#{$unlabeledset[$current->[0]]}; $j++) {
						$centroids[$i][$j] += $unlabeledset[$current->[0]][$j];
					}
					$current = $current->[1];
				}
			}
		}
		
		# now process the centroids
		open (F4, ">$resultfolder/clusthist-$currsize.txt") || die ("Could not open file!"); 	# file containing cluster statistics
		open (F5, ">$resultfolder/selectedset-$currsize.tmp") || die ("Could not open file!");  # file containing selected training set
		for (my $i = 0; $i<=$#clusters; $i++) {
			if (defined $clusters[$i]) {
				# calculate final centroid
				for (my $j=0; $j<=$#{$unlabeledset[0]}; $j++) {
					$centroids[$i][$j] = $centroids[$i][$j]/$size[$i];
				}
				my $mindist = 99999;
				my $mininst = -1;
				
				# find instance closest to centroid
				my $current = $clusters[$i];
				while (defined $current) {
					my $dist = Dist($unlabeledset[$current->[0]], $centroids[$i]);
					if ($dist < $mindist) {
						$mindist = $dist;
						$mininst = $current->[0];
					}
					$current = $current->[1];
				}
				undef($centroids[$i]);
				# print centroid data to file
				my $clustnrel = $size[$i] - $clustrel[$i]; # calculate how many instances with label = 0
				print F4 "$i $size[$i] $clustrel[$i] $clustnrel\n";
				# print label and qid (if the dataset has them)
				if ($hasqid eq "noqid") {
					print F5 "$labels[$mininst] ";
				}
				else {
					print F5 "$labels[$mininst] $qids[$mininst] ";
				}
				# recreate instance in the original format using the feature values
				for (my $j=0; $j<=$#{$unlabeledset[$mininst]}; $j++) {
					my $feat = $j+1;
					print F5 "$feat:$unlabeledset[$mininst][$j] "; # prints feature as "featnum:featval "
				}
				print F5 "\n";
			}
		}
		undef(@centroids);
		close (F4);
		close (F5);
		
		# sort final training file by queryid (for L2R datasets with qids)
		if ($hasqid eq "") {
			system("cat $resultfolder/selectedset-$currsize.tmp | sort -t \: -n -k 2 > $resultfolder/selectedset-$currsize.txt");
			system("rm -f $resultfolder/selectedset-$currsize.tmp");
		}
		else {
			# classification ("noqid): no sorting needed; Notice that selected instances appear in no particular order
			system("mv $resultfolder/selectedset-$currsize.tmp $resultfolder/selectedset-$currsize.txt");
		}
		
	}
	
	# finish execution and write epsilon (this is the minimum edge size necessary for hierarchical clustering)
	# this is just additional information and is not needed or used by Cover.
	if ($currsize < $selsize) { 
		open (F4, ">$resultfolder/epsilon-$currsize.txt") || die ("Could not open file!");
		print F4 "\nEpsilon array:\n";
		for (my $i=0; $i<=$#distances; $i++) {
			print F4 "$distances[$i],";
		}
		print F4 "\n";
		close(F4);
		last; 
	}
}
$|=0;

	
print "\n\nSize of clusters: $#clusters\n"; 

