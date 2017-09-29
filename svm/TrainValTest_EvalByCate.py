# this program is to split original deceptive and authentic reviews into train, val and test dataset.
# based on some rules to split
import random
import pickle
from shutil import copyfile
import operator
import glob
import os
from multiprocessing import Process
from files_remove import *

class Track:
	def __init__(self, line):
		words = line.split()
		self.memberId = words[0]
		self.productId = words[1]
		self.cat = words[2]

def load_syntactic_complexity(src_folder):
	dec_syn_comp = {}
	tru_syn_comp = {}

	file = open(src_folder + "syntactic_complexity/dec_syntactic_complexity", 'r')
	for line in file:
		words = line.split(",")
		if words[0] == "Filename":
			continue
		dec_syn_comp[words[0]] = line
	file.close()
	for true_file in glob.glob(src_folder + "syntactic_complexity/tru_syntactic_complexity*"):
		file = open(true_file, 'r')
		for line in file:
			words = line.split(",")
			if words[0] == "Filename":
				continue
			tru_syn_comp[words[0]] = line
		file.close()
	return dec_syn_comp, tru_syn_comp

#if __name__ == "__main__":
# max_size controls the total number of train deceptive and val deceptive
# sample_ratio controls percentages of max_size to be considered
# tru_dec_ratio controls authentic:deceptive reviews
def TrainValTest_EvalByCate_main(src_folder, dst_folder, rand_seed, dec_reviewers, dec_products, valid_train_cat, valid_test_cat, \
	max_size = None, sample_ratio = 1.0, train_tru_dec_ratio = 2, test_tru_dec_ratio = None):

	random.seed(rand_seed)
	print rand_seed
	#print valid_reviewer
	

	#test_dec_output = open("test-dec.txt", 'w')
	#test_tru_output = open("test-tru.txt", 'w')
	val_dec_output = open(dst_folder + "val-dec.txt", 'w')
	val_tru_output = open(dst_folder + "val-tru.txt", 'w')
	train_dec_output = open(dst_folder + "train-dec.txt", 'w')
	train_tru_output = open(dst_folder + "train-tru.txt", 'w')

	#test_dec_track = open("test-dec_track.txt", 'w')
	#test_tru_track = open("test-tru_track.txt", 'w')
	val_dec_track = open(dst_folder + "val-dec_track.txt", 'w')
	val_tru_track = open(dst_folder + "val-tru_track.txt", 'w')
	train_dec_track = open(dst_folder + "train-dec_track.txt", 'w')
	train_tru_track = open(dst_folder + "train-tru_track.txt", 'w')

	val_dec_SynComp = open(dst_folder + "val-dec_SynComp.txt", 'w')
	val_tru_SynComp = open(dst_folder + "val-tru_SynComp.txt", 'w')
	train_dec_SynComp = open(dst_folder + "train-dec_SynComp.txt", 'w')
	train_tru_SynComp = open(dst_folder + "train-tru_SynComp.txt", 'w')

	dec_syn_comp, tru_syn_comp = load_syntactic_complexity(src_folder)

	dec_count = 0
	tru_count = 0

	train_dec_count = 0
	val_dec_count = 0
	#test_dec_count = 0
	train_tru_count = 0
	val_tru_count = 0
	#test_tru_count = 0


	dec_file = open(src_folder + "dec.txt", 'r')
	dec_track_lines = open(src_folder + "dec_track.txt", 'r').readlines()
	files_remove(dst_folder)


	for line in dec_file:
		dec_count += 1
		track = Track(dec_track_lines[dec_count - 1])

		# the second filter
		if track.productId not in dec_products:
			continue
		# category level
		
		if track.cat in valid_test_cat:
			
			val_dec_count += 1
			val_dec_output.write(line)
			val_dec_track.write(dec_track_lines[dec_count - 1])
			src = src_folder + "dec_folder_coreNLP/" + str(dec_count) + ".txt.out"
			dst = dst_folder + "val-dec_folder_coreNLP/" + str(val_dec_count) + ".txt.out"
			copyfile(src, dst)
			val_dec_SynComp.write(dec_syn_comp[str(dec_count) + ".txt"])
		elif track.cat in valid_train_cat:
			if max_size != None and train_dec_count > max_size: # sample ratio only apply to training data
				continue
			if random.randint(0, 100) > sample_ratio * 100:
				continue
			train_dec_count += 1
			train_dec_output.write(line)
			train_dec_track.write(dec_track_lines[dec_count - 1])
			src = src_folder + "dec_folder_coreNLP/" + str(dec_count) + ".txt.out"
			dst = dst_folder + "train-dec_folder_coreNLP/" + str(train_dec_count) + ".txt.out"
			copyfile(src, dst)
			train_dec_SynComp.write(dec_syn_comp[str(dec_count) + ".txt"])
		
		
		

	tru_file = open(src_folder + "tru.txt", 'r')
	tru_track_lines = open(src_folder + "tru_track.txt", 'r').readlines()

	for line in tru_file:
		tru_count += 1
		track = Track(tru_track_lines[tru_count - 1])


		# the second filter
		if track.productId in dec_products:
			continue

		rand_num = random.randint(0, 10)
		if track.cat in valid_test_cat:

			if test_tru_dec_ratio != None and val_tru_count > test_tru_dec_ratio * val_dec_count:
				continue

			val_tru_count += 1
			val_tru_output.write(line)
			val_tru_track.write(tru_track_lines[tru_count - 1])
			src = src_folder + "tru_folder_coreNLP/" + str(tru_count) + ".txt.out"
			dst = dst_folder + "val-tru_folder_coreNLP/" + str(val_tru_count) + ".txt.out"
			copyfile(src, dst)
			val_tru_SynComp.write(tru_syn_comp[str(tru_count) + ".txt"])
		elif track.cat in valid_train_cat:
			#if random.randint(0, 2) != 0:
			#	continue
			if train_tru_count > train_tru_dec_ratio * train_dec_count:
				continue
			if random.randint(0, 100) > sample_ratio * 100:
				continue
			train_tru_count += 1
			train_tru_output.write(line)
			train_tru_track.write(tru_track_lines[tru_count - 1])
			src = src_folder + "tru_folder_coreNLP/" + str(tru_count) + ".txt.out"
			dst = dst_folder + "train-tru_folder_coreNLP/" + str(train_tru_count) + ".txt.out"
			copyfile(src, dst)
			train_tru_SynComp.write(tru_syn_comp[str(tru_count) + ".txt"])

	#print "train_tru_count, train_dec_count:", train_tru_count, train_dec_count
	return train_dec_count, train_tru_count, val_dec_count, val_tru_count
	val_dec_output.close()
	val_tru_output.close()
	train_dec_output.close()
	train_tru_output.close()

	val_dec_track.close()
	val_tru_track.close()
	train_dec_track.close()
	train_tru_track.close()

	val_dec_SynComp.close()
	val_tru_SynComp.close()
	train_dec_SynComp.close()
	train_tru_SynComp.close()

		
