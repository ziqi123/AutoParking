import os
path = '/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/point/'

# 获取该目录下所有文件，存入列表中
fileList = os.listdir(path)

for i in fileList:

    num = i.count('_OA')
    if num == 0:
        newname = i.strip('.txt')+'_OA.txt'
    else:
        newname = i
    # # 设置新文件名
    # newname = i.strip('.txt') + '_OA'+'.txt'

    os.rename(os.path.join(path, i), os.path.join(
        path, newname))  # 用os模块中的rename方法对文件改名
    # print(i, '======>', newname)
