import sys
sys.path.append("../nbsvm/")
sys.path.append("../NN_functions/")
from process_text import *
from TrainValTest_EvalByCate import *
from process_data_pytorch import *
from multiprocessing import Process
from NN_error_analysis import *
from train import *
import time


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #self.parser.add_argument("--rnn_type", dest="rnn_type", type=str, metavar='<str>', default='LSTM', help="Recurrent unit type (lstm|gru|simple) (default=lstm)")
    parser.add_argument("--opt", dest="opt", type=str, metavar='<str>', default='RMS', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
    parser.add_argument("--emb_size", dest="embedding_size", type=int, metavar='<int>', default=300, help="Embeddings dimension (default=300)")
    parser.add_argument("--rnn_size", dest="rnn_size", type=int, metavar='<int>', default=128, help="RNN dimension. '0' means no RNN layer (default=300)")
    parser.add_argument("--batch-size", dest="batch_size", type=int, metavar='<int>', default=20, help="Batch size (default=50)")
    parser.add_argument("--rnn_layers", dest="rnn_layers", type=int, metavar='<int>', default=1, help="Number of RNN layers")
    parser.add_argument("--aggregation", dest="aggregation", type=str, metavar='<str>', default='mean', help="The aggregation method for regp and bregp types (mot|attsum|attmean) (default=mot)")
    parser.add_argument("--dropout", dest="dropout", type=float, metavar='<float>', default=0.5, help="The dropout probability. To disable, give a negative number (default=0.5)")
    #parser.add_argument("--pretrained", dest="pretrained", type=int, metavar='<int>', default=1, help="Whether to use pretrained or not")
    parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=70, help="Number of epochs (default=20)")
    parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
    parser.add_argument('--gpu', dest='gpu', type=int, metavar='<int>', default=0, help="Specify which GPU to use (default=0)")
    parser.add_argument("--hdim", dest='hidden_layer_size', type=int, metavar='<int>', default=300, help="Hidden layer size (default=300)")
    parser.add_argument("--lr", dest='learn_rate', type=float, metavar='<float>', default=0.0005, help="Learning Rate")
    parser.add_argument("--clip", dest='clip', type=float, metavar='<float>', default=5.0, help="Gradient clipping")
    parser.add_argument("--trainable", dest='trainable', type=bool, metavar='<bool>', default=False, help="Trainable Word Embeddings (default=False)")
    parser.add_argument('--l2_reg', dest='l2_reg', type=float, metavar='<float>', default=0.0, help='L2 regularization, default=0')
    parser.add_argument('--eval', dest='eval', type=int, metavar='<int>', default= 10, help='Epoch to evaluate results')
    parser.add_argument('--dev', dest='dev', type=int, metavar='<int>', default=1, help='1 for development set 0 to train-all')
    parser.add_argument('--toy', dest='toy', type = bool, metavar='<bool>', default=False, help="Use toy dataset (for fast testing), True means use toy dataset")
    parser.add_argument('--metafeature_dim', dest='metafeature_dim', type = int, metavar='<int>', default=25, help="Meta feature dimension (default=25)")
    #parser.add_argument('--cuda', action='store_true', help='use CUDA')
    args = parser.parse_args()

    start = time.time()
    src_folder = "../real_data/preprocess_merge/"

    with open(src_folder + "dec_reviewers.p", 'rb') as f:
        dec_reviewers = set(pickle.load(f))
    with open(src_folder + "dec_products.p", 'rb') as f:
        dec_products = set(pickle.load(f))
    with open(src_folder + "productId2name.p", 'rb') as file:
        productId2name = pickle.load(file)
    with open(src_folder + "productId2cate.p", 'rb') as file:
        productId2cate = pickle.load(file)
    
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
    cate_setList.append(set(["DVD", "Movies_&_TV"]))
    cate_setList.append(set(["Paperback", "Hardcover", "Kindle_Edition", "Books", "Kindle_Store"]))
    cate_setList.append(set(["Health_and_Beauty", "Beauty", "Health_&_Personal_Care"]))
    
    #cate_setList.append(set(["Kitchen", "Home_&_Kitchen", "Kitchen_&_Dining"]))
    
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

    train_tru_dec_ratio = 3
    test_tru_dec_ratio = 3
    train_sample_percent = 1



    #print "process_data_main(...)"
    #process_data_main(current_folder, repeat_num, "../../tools/glove.6B/glove.6B.300d.txt", productId2name)
    #print "LSTM_main(...)"
    #LSTM_main(current_folder, repeat_num, args)
    
    
    for valid_train_cat in cate_setList:
        output.write("train: " + str(valid_train_cat) + "\n")
        print ("train: " + str(valid_train_cat))
        macro_R_sum = 0.0
        macro_P_sum = 0.0

        valid_test_cat_idxList = []
        #trained_model_save_path = "trained_model.pt"
        
        for i, valid_test_cat in enumerate(cate_setList):

            if valid_train_cat == valid_test_cat:
                continue
            output.write("test_cat_idx: " + str(i) + " ")
            valid_test_cat_idxList.append(i)

            valid_test_cat_idx = i
            
            #for valid_test_cat_idx in valid_test_cat_idxList:
            dst_folder = 'thread' + str(valid_test_cat_idx) + '/'
            
            train_dec_count, train_tru_count, val_dec_count, val_tru_count = \
            TrainValTest_EvalByCate_main(src_folder, dst_folder, 1, dec_reviewers, dec_products, valid_train_cat, valid_test_cat, None, train_sample_percent, train_tru_dec_ratio, test_tru_dec_ratio)
            #train_dec_count, train_tru_count, val_dec_count, val_tru_count = \
            #TrainValTest_EvalByCate_main(src_folder, dst_folder, 1, dec_reviewers, dec_products, valid_train_cat | rest_cate, valid_test_cat, None, train_sample_percent, train_tru_dec_ratio, test_tru_dec_ratio)

            #process_text_main(dst_folder, invalid_POS_tags)
            print "train_dec_count, train_tru_count, val_dec_count, val_tru_count:", train_dec_count, train_tru_count, val_dec_count, val_tru_count
        
        
        print "process_data_main(...)"
        process_data_main(current_folder, valid_test_cat_idxList, "../../tools/glove.6B/glove.6B.300d.txt", productId2name)
            

        print "LSTM_main(...)"
        LSTM_main(current_folder, valid_test_cat_idxList, args)
            
        for valid_test_cat_idx in valid_test_cat_idxList:
            dst_file = 'thread' + str(valid_test_cat_idx) + '/LSTM_true_and_pred_value'
            R, P, F, dec_test_count = error_analysis_main(dst_file)
            print 'R, P, F, dec_test_count:', R, P, F, dec_test_count
            output.write(str(valid_test_cat_idx) + ' R, P, F, dec_test_count(Category level): ' + str(R) + ' ' + str(P) + ' ' + str(F) + ' ' + str(dec_test_count) + '\n')
            macro_R_sum += R
            macro_P_sum += P

            

        macro_R_ave = macro_R_sum / float(len(valid_test_cat_idxList))
        macro_P_ave = macro_P_sum / float(len(valid_test_cat_idxList))
        macro_F_ave = 2 * macro_P_ave * macro_R_ave / (macro_P_ave + macro_R_ave)

        output.write('R, P, F, dec_test_count(cross domain average): ' + str(macro_R_ave) + ' ' + str(macro_P_ave) + ' ' + str(macro_F_ave) + '\n')
            
    end = time.time()
    print"Time: ", end - start
    # remove parsing files
    for valid_test_cat_idx in valid_test_cat_idxList:
        dst_folder = 'thread' + str(valid_test_cat_idx) + '/'
        files_remove(dst_folder)

    output.close()  
    
        
