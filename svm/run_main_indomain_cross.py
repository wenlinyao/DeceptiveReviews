# in-domain cross validation
from TrainValTest_IndomainCross import *
from nbsvm2 import *
from error_analysis2 import *
from multiprocessing import Process
import time


if __name__ == '__main__':
	start = time.time()

	#src_folder = "../real_data/preprocess_1_onefile/"
	src_folder = "../real_data/preprocess_merge/"

	with open(src_folder + "dec_reviewers.p", 'rb') as f:
		dec_reviewers = set(pickle.load(f))
	with open(src_folder + "dec_products.p", 'rb') as f:
		dec_products = set(pickle.load(f))
	#with open(src_folder + "cat2review_num.p", 'rb') as f:
	#	cat2review_num = pickle.load(f)
	#with open(src_folder + "dec_reviewers_cat2num.p", 'rb') as f:
	#	dec_reviewers_cat2num = pickle.load(f)
	
	parameterList = []

	#invalid_POS_tags = set(["NN", "NNS", "NNP", "NNPS"])
	invalid_POS_tags = set()

	#top_f_num = 500 # top_feature_selection_num
	top_f_num = None
	parameterList.append({"unlexicalized_production": 1})
	#parameterList.append({"unlexicalized_production": 1, "POS": 1, "LIWC": 1, "advertising_phrases": 1, "product_name_overlap": 12})
	#parameterList.append({"unlexicalized_production": 1, "POS": 1, "LIWC": 1, "advertising_phrases": 1, "product_name_overlap": 12, "token_emotion_trans": 1})
	#parameterList.append({"unlexicalized_production": 1, "POS": 1, "LIWC": 1, "advertising_phrases": 1, "product_name_overlap": 12, "token_emotion_trans": 1, \
	#	"syntactic_complexity": [1, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]})

	"""
	parameterList.append({"ngram": 1})
	parameterList.append({"POS": 1})
	parameterList.append({"LIWC": 1})
	parameterList.append({"syntax_production": 1})
	parameterList.append({"syntax_production": 1, "dependencies": 1, "emotion_trans": 1, "emotion_trans_lexical": 1, "product_name_overlap": 12, "advertising_phrases": 1, "syntactic_complexity": [3, 8, 9, 10]})
	parameterList.append({"dependencies": 1, "emotion_trans": 1, "emotion_trans_lexical": 1, "product_name_overlap": 12, "advertising_phrases": 1, "syntactic_complexity": [3, 8, 9, 10]})
	"""
	"""
	parameterList.append({"ngram": 1})
	parameterList.append({"ngram": 12})
	parameterList.append({"ngram": 123})
	
	
	parameterList.append({"ngram": 1, "POS": 1})
	parameterList.append({"ngram": 1, "POS": 12})
	parameterList.append({"ngram": 1, "POS": 123})
	
	parameterList.append({"ngram": 1, "advertising_phrases": 1})

	parameterList.append({"ngram": 1, "product_name_overlap": 12})
	parameterList.append({"ngram": 1, "product_name_ngram": 2})

	
	parameterList.append({"ngram": 1, "LIWC": 1})
	parameterList.append({"ngram": 1, "syntax_production": 1})
	parameterList.append({"ngram": 1, "passive_voice": 1})
	parameterList.append({"ngram": 1, "dependencies": 1})
	parameterList.append({"ngram": 1, "emotion_trans": 1})
	parameterList.append({"ngram": 1, "emotion_trans_lexical": 1})

	for i in range (1, 22):
		parameterList.append({"ngram": 1, "syntactic_complexity": [i]})
	"""
	#parameterList.append({"ngram": 1, "POS": 1, "LIWC": 1})
	#parameterList.append({"ngram": 1, "POS": 1, "LIWC": 1, "syntax_production": 1, "dependencies": 1, "emotion_trans": 1, "emotion_trans_lexical": 1, "product_name_overlap": 12, "advertising_phrases": 1, "syntactic_complexity": [3, 8, 9, 10]})

	

	cross_num = 5

	classifier = "MaxEnt"
	#classifier = "SVM"

	#feature_value_type = "nb"
	feature_value_type = "bool"
	#feature_value_type = "freq"

	# multi-threads
	
	valid_reviewer = []
	output = open('different_cat_figure_matrix', 'w', 0)
	#valid_catList = [['Paperback', 'Hardcover', 'Kindle_Edition'], ['Health_and_Beauty']]

	all_cate = set(['Health_and_Beauty', 'Paperback', 'Hardcover', 'Kindle_Edition', 'Personal_Computers', 'Apparel', 'Music', \
	'Office_Product', 'Electronics', 'DVD', 'Accessory', 'Sports' , 'Toy' , 'CD-ROM' , 'Audio_CD', 'Audio_Cassette', 'Kitchen', 'Jewelry', \
	'Tools_&_Home_Improvement' , 'Grocery', 'Video_Game', 'Software', 'Baby_Product', 'Camera', 'Blu-ray', 'Automotive', 'App', 'Phone', \

	"Kindle_Store", "Books", "Health_&_Personal_Care", "Appstore_for_Android", "Movies_&_TV", "Electronics", "Cell_Phones_&_Accessories", "Computers_&_Accessories",\
	"Beauty", "Shoes", "Toys_&_Games", "Kitchen_&_Dining", "Home_&_Kitchen", "Jewelry", "Grocery_&_Gourmet_Food", "Video_Games", "Baby", "Home_Improvement", "Watches", "Arts", \
	"Musical_Instruments", "Sports_&_Outdoors", "MP3_Albums", "MP3_Songs", "Music", "Software", "Clothing", "Office_Products", "Pet_Supplies", "Automotive", "Patio", "Camera_&_Photo", "Misc"])

	cate_setList = []
	
	cate_setList.append(set(["Paperback", "Hardcover", "Kindle_Edition", "Books", "Kindle_Store"]))
	cate_setList.append(set(["Health_and_Beauty", "Beauty", "Health_&_Personal_Care"]))
	cate_setList.append(set(["Electronics", "Personal_Computers", "Computers_&_Accessories", "Cell_Phones_&_Accessories", "CD-ROM"]))
	#cate_setList.append(set(["Kitchen", "Home_&_Kitchen", "Kitchen_&_Dining"]))
	cate_setList.append(set(["DVD", "Movies_&_TV"]))
	#cate_setList.append(set(["Audio_CD", "Music", "MP3_Songs"]))
	#cate_setList.append(set(["App", "Appstore_for_Android", "Software", "Video_Games"]))

	rest_cate = all_cate
	for cate_set in cate_setList:
		rest_cate = rest_cate - cate_set

	cate_setList.append(rest_cate)

	#additional_valid_train_cat = rest_cate
	additional_valid_train_cat = None
	#tru2dec_ratio = 2


	#for valid_cat in valid_catList:
	for tru2dec_ratio in [3]:
		#output.write("tru2dec_ratio: " + str(tru2dec_ratio) + "\n")
		for valid_cat in cate_setList:
			print valid_cat
			output.write(str(valid_cat) + '\n')
			#valid_cat = all_cate
			for cross_id in range(0, cross_num):
				dst_folder = 'thread' + str(cross_id) + '/'
				train_dec_count, train_tru_count, val_dec_count, val_tru_count, additional_dec_count, additional_tru_count \
				= TrainValTest_IndomainCross_main(src_folder, dst_folder, cross_id, cross_num, dec_reviewers, dec_products, valid_cat, additional_valid_train_cat, tru2dec_ratio)
			output.write("train_dec_count, train_tru_count, val_dec_count, val_tru_count, additional_dec_count, additional_tru_count: ")
			output.write(str(train_dec_count) + " " + str(train_tru_count) + " " + str(val_dec_count) + " " + str(val_tru_count) + " "\
				+ str(additional_dec_count) + " " + str(additional_tru_count) + "\n")

			for parameter_setting in parameterList:
				for p in parameter_setting: # output valid parameters
					if parameter_setting[p] != 0:
						output.write(str(p) + ': ' + str(parameter_setting[p]) + ' ')
				output.write('\n')
				
				R_sum = 0.0
				P_sum = 0.0
				F_sum = 0.0

				Top_w_R_sum = 0.0
				Top_w_P_sum = 0.0
				Top_w_F_sum = 0.0

				processV = []
				
				for cross_id in range(0, cross_num):
					dst_folder = 'thread' + str(cross_id) + '/'
					processV.append(Process(target = classifier_main, args = (classifier, dst_folder, 'train-dec.txt', 'train-tru.txt', 'val-dec.txt', 'val-tru.txt', \
							'model_output', 'liblinear-1.96', parameter_setting, invalid_POS_tags, top_f_num, feature_value_type)))
					#processV.append(Process(target = nb_svm_main, args = (dst_folder, 'train-dec.txt', 'train-tru.txt', 'val-dec.txt', 'val-tru.txt', 'model_output', 'liblinear-1.96', parameter_setting, invalid_POS_tags)))
				for i in range(0, cross_num):
					processV[i].start()
				for i in range(0, cross_num):
					processV[i].join()
				
				for cross_id in range(0, cross_num):
					dst_folder = 'thread' + str(cross_id) + '/'
					R, P, F, true_pos, dec_test_count, total_labeled = error_analysis2_main(dst_folder, "model_output")
					print cross_id, 'R, P, F, dec_test_count:', R, P, F, dec_test_count
					output.write(str(cross_id) + ': R, P, F, dec_test_count: ' + str(R) + ' ' + str(P) + ' ' + str(F) + ' ' + str(dec_test_count) + '\n')
					R_sum += R
					P_sum += P
					F_sum += F
					if top_f_num != None:
						R, P, F, true_pos, dec_test_count, total_labeled = error_analysis2_main(dst_folder, "model_output" + "_TOP" + str(top_f_num))
						print 'R, P, F, dec_test_count (Top n weight):', R, P, F, dec_test_count
						Top_w_R_sum += R
						Top_w_P_sum += P
						Top_w_F_sum += F
						output.write(str(cross_id) + ': R, P, F, dec_test_count(Top n weight): '+ str(R) + ' ' + str(P) + ' ' + str(F) + ' ' + str(dec_test_count) + '\n')

				output.write('R, P, F, dec_test_count(Category level): ' + str(R_sum/float(cross_num)) + ' ' + str(P_sum/float(cross_num)) + ' ' + str(F_sum/float(cross_num))\
								 + ' ' + str(dec_test_count) + '\n')

				if top_f_num != None:
					output.write('R, P, F, dec_test_count(Top n weight): ' + str(Top_w_R_sum/float(cross_num)) + ' ' + str(Top_w_P_sum/float(cross_num)) + ' ' + str(Top_w_F_sum/float(cross_num))\
							 + ' ' + str(dec_test_count) + '\n')

				output.write('\n')
	end = time.time()
	print"Time: ", end - start

	# remove parsing files
	for cross_id in range(0, cross_num):
		dst_folder = 'thread' + str(cross_id) + '/'
		files_remove(dst_folder)
	output.close()	
	
	