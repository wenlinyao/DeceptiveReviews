# this program is to read LIWC dic into memory
def LIWC_load(file_name):
	file = open(file_name, 'r')
	label_read_flag = 0
	LIWC_dic = {}
	for line in file:
		if not line.strip():
			continue
		words = line.split()
		if words[0] == '%':
			label_read_flag += 1
			continue
		if label_read_flag == 2:
			LIWC_dic[words[0]] = words[1:]
	return LIWC_dic

"""
if __name__ == "__main__":
	LIWC_dic = LIWC_load("../dic/LIWC2015_English.dic")
	for word in LIWC_dic:
		print word, LIWC_dic[word]
		raw_input("continue?")
"""
