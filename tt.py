



fw = open("query_deal.txt", 'w')  # 将要输出保存的文件地址

for line in range(20):  # 读取的文件
    fw.write('0000000000000000000000000000000000000000')  # 将字符串写入文件中
    # line.rstrip("\n")为去除行尾换行符
    fw.write("\n")  # 换行
fw.close()