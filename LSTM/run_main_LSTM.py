
import sys
sys.path.append("../nbsvm/")
sys.path.append("../NN_functions/")
from process_text import *
from TrainValTest_EvalByCate import *
from process_data import *
from multiprocessing import Process
from NN_error_analysis import *
from lstm import *
import time


if __name__ == '__main__':
    start = time.time()
    src_folder = "../real_data/preprocess_merge/"

    with open(src_folder + "dec_reviewers.p", 'rb') as f:
        dec_reviewers = set(pickle.load(f))
    with open(src_folder + "dec_products.p", 'rb') as f:
        dec_products = set(pickle.load(f))
    
    
    # this is category level evaluation
    repeat_num = 1

    # multi-threads

    valid_reviewer = []
    output = open('different_cat_figure_matrix', 'w', 0)

    all_cate = set(['Health_and_Beauty', 'Paperback', 'Hardcover', 'Kindle_Edition', 'Personal_Computers', 'Apparel', 'Music', \
    'Office_Product', 'Electronics', 'DVD', 'Accessory', 'Sports' , 'Toy' , 'CD-ROM' , 'Audio_CD', 'Audio_Cassette', 'Kitchen', 'Jewelry', \
    'Tools_&_Home_Improvement' , 'Grocery', 'Video_Game', 'Software', 'Baby_Product', 'Camera', 'Blu-ray', 'Automotive', 'App', 'Phone', \

    "Kindle_Store", "Books", "Health_&_Personal_Care", "Appstore_for_Android", "Movies_&_TV", "Electronics", "Cell_Phones_&_Accessories", "Computers_&_Accessories",\
    "Beauty", "Shoes", "Toys_&_Games", "Kitchen_&_Dining", "Home_&_Kitchen", "Jewelry", "Grocery_&_Gourmet_Food", "Video_Games", "Baby", "Home_Improvement", "Watches", "Arts", \
    "Musical_Instruments", "Sports_&_Outdoors", "MP3_Albums", "MP3_Songs", "Music", "Software", "Clothing", "Office_Products", "Pet_Supplies", "Automotive", "Patio", "Camera_&_Photo", "Misc"])

    cate_setList = []
    
    cate_setList.append(set(["Electronics", "Personal_Computers", "Computers_&_Accessories", "Cell_Phones_&_Accessories", "CD-ROM"]))
    cate_setList.append(set(["Paperback", "Hardcover", "Kindle_Edition", "Books", "Kindle_Store"]))
    cate_setList.append(set(["Health_and_Beauty", "Beauty", "Health_&_Personal_Care"]))
    #
    #cate_setList.append(set(["Kitchen", "Home_&_Kitchen", "Kitchen_&_Dining"]))
    cate_setList.append(set(["DVD", "Movies_&_TV"]))
    #cate_setList.append(set(["Audio_CD", "Music", "MP3_Songs"]))
    #cate_setList.append(set(["App", "Appstore_for_Android", "Software", "Video_Games"]))

    rest_cate = all_cate
    for cate_set in cate_setList:
        rest_cate = rest_cate - cate_set

    #cate_setList.append(rest_cate)

    #valid_train_cat = set(['Paperback', 'Hardcover', 'Kindle_Edition'])
    #valid_test_cat = set(['Electronics'])
    #valid_test_cat = set(['Health_and_Beauty'])

    
    current_folder = ""

    
    #invalid_POS_tags = set(["NN", "NNS", "NNP", "NNPS"])
    invalid_POS_tags = set()
    epoch_num = 50
    train_tru_dec_ratio = 3
    test_tru_dec_ratio = 3
    train_sample_percent = 1
    

    for valid_train_cat in cate_setList:
        output.write("train: " + str(valid_train_cat) + "\n")

        macro_R_sum = 0.0
        macro_P_sum = 0.0

        valid_test_cat_idxList = []
        for i, valid_test_cat in enumerate(cate_setList):
            if valid_train_cat == valid_test_cat:
                continue
            valid_test_cat_idxList.append(i)
            output.write("test_cat_idx: " + str(i) + " ")
            R_sum = 0.0
            P_sum = 0.0
            F_sum = 0.0
            processV = []
            for rand_seed in range(0, repeat_num):
                dst_folder = 'thread' + str(rand_seed) + '/'
                
                #train_dec_count, train_tru_count, val_dec_count, val_tru_count = \
                #TrainValTest_EvalByCate_main(src_folder, dst_folder, rand_seed, dec_reviewers, dec_products, valid_train_cat, valid_test_cat, None, train_sample_percent, train_tru_dec_ratio, test_tru_dec_ratio)
                #train_dec_count, train_tru_count, val_dec_count, val_tru_count = \
                #TrainValTest_EvalByCate_main(src_folder, dst_folder, rand_seed, dec_reviewers, dec_products, valid_train_cat | rest_cate, valid_test_cat, None, train_sample_percent, train_tru_dec_ratio, test_tru_dec_ratio)

                #process_text_main(dst_folder, invalid_POS_tags)


            print "process_data_main(...)"
            #process_data_main(current_folder, repeat_num, "new_", "../../tools/GoogleNews-vectors-negative300.bin")
            

            print "LSTM_main(...)"
            
            LSTM_main(current_folder, repeat_num, epoch_num)
            
            for rand_seed in range(0, repeat_num):
                dst_file = 'thread' + str(rand_seed) + '/LSTM_true_and_pred_value'
                R, P, F, dec_test_count = error_analysis_main(dst_file)
                print 'R, P, F, dec_test_count:', R, P, F, dec_test_count
                R_sum += R
                P_sum += P
                F_sum += F

            macro_R_sum += R_sum/float(repeat_num)
            macro_P_sum += P_sum/float(repeat_num)

            output.write('R, P, F, dec_test_count(Category level): ' + str(R_sum/float(repeat_num)) + ' ' + str(P_sum/float(repeat_num)) + ' ' + str(F_sum/float(repeat_num))\
                         + ' ' + str(dec_test_count) + '\n')

        macro_R_ave = macro_R_sum / float(len(valid_test_cat_idxList))
        macro_P_ave = macro_P_sum / float(len(valid_test_cat_idxList))
        macro_F_ave = 2 * macro_P_ave * macro_R_ave / (macro_P_ave + macro_R_ave)

        output.write('R, P, F, dec_test_count(cross domain average): ' + str(macro_R_ave) + ' ' + str(macro_P_ave) + ' ' + str(macro_F_ave) + '\n')
            
    end = time.time()
    print"Time: ", end - start
    # remove parsing files
    for rand_seed in range(0, repeat_num):
        dst_folder = 'thread' + str(rand_seed) + '/'
        files_remove(dst_folder)

    output.close()  
    
        
