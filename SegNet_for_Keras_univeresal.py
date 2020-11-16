"""Программа для обучения и прогнозирования задачи по семантической сегментации
1. Использована сеть SegNet Взято с сайта https://github.com/ykamikawa/tf-keras-SegNet/blob/master/model.py
2. Код оптимизирован в 2020 году.
"""

# Импорт библеотеки по работе с матрицами
import numpy as np

# Импорт библеотеки keras для нейросетей
from keras import backend as K
# Импорт библеотеки для работы с .mat файлами
import scipy.io
# Импорт собственного модуля с функциями для поддержки
import support_function_for_SegNet as sffU
from SegNet_example2 import segnet
import time
#1. Переменная для загрузки различных значений
pl_sphere_cyl = np.array([3, 2]) #%1 - какая серия эксперментов; 2 - вид из серии экспериментов
#1, 2 - серия с олисарус, теория и практика; 1 - 1 Деталь Олеса Рус ; 2 - 5
#Деталь Олеса Рус; 3 - 4 Деталь Олеса Рус 4 - 2 деталь Олеса Рус
#3, 4 - серия с лопатками, теория и практика; 1 - единичная лопатка РК турбины; 2 - блок из 3 лопаток СА
#Blade_turbine_ideal, Лопатка_турбины_август_1_обр,40.412.007_идеал, 40.412.007_упрощенная_1
#2. Имя файла для загрузки
name_file=sffU.name_of_results(pl_sphere_cyl)

# Формат входных данных (каналы*размер*размер или размер*размер*каналы)
if (1==0):
    K.set_image_data_format('channels_first') #Формат входных данных
    axis_direction = 1 #Направление при декодировании каналов-размера
else:
    K.set_image_data_format('channels_last')
    axis_direction=-1

#Основные управляющие переменные
pl=0
pl_save = 1
pl_real_teor=1 #реальные тестовые данные (1) или теоретические (0)
# Количество тестовых данных (реальных) для работы с нейросетью
N_test=50
# Количество обучающих данных (в названии)
fileIndex = 1000
imgtype="tif"

num_classes = 5
pl_norm = 1
batch_size = 1
epochs = 10

# Загрузка всех необходимых путей
fileNameTest, fileNameTrain, fileNameTest_cat, fileNameTrain_cat, fileNameResult1, fileNameTemplate, fileNameResult,\
     baseFolder, baseFolder_, fileName, fileName1 = sffU.names_of_files_and_patches(name_file, N_test, fileIndex)

mat1 = scipy.io.loadmat(baseFolder+fileNameTest+ ".mat")
mat1_ = scipy.io.loadmat(baseFolder+fileNameTest_cat+ ".mat")
for i in range(6):
    fileName_train = fileNameTrain + str(i+1)+ name_file+str(fileIndex)+".mat" #1200
    fileName_train_ = fileNameTrain_cat + str(i + 1) + name_file + str(fileIndex) + ".mat"
    mat = scipy.io.loadmat(baseFolder+fileName_train)
    mat_ = scipy.io.loadmat(baseFolder + fileName_train_)
    x_train, y_train, x_test, y_test, INPUT_SHAPE = sffU.load_from_mat_train_and_test(mat, mat1,mat_, mat1_, i,\
                                                                                        pl_norm, pl, pl_real_teor)

    cnn_s = segnet(INPUT_SHAPE, y_train.shape[2],  pl_norm)#axis_direction,
    cnn_s.summary()

    if (pl_save == 1):
        s = time.time()
        history = cnn_s.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
                            shuffle=True)
        e = time.time()
        print(e - s)
        # Сохранение нейронной сети
        #sffU.save_weights_of_cnn(cnn_s,fileName, fileName1, i)
        if (1==0):
            scipy.io.savemat(baseFolder + "test.mat",{'y_predict': y_predict,
                                                                 'y_predict1': y_predict})

    else:
        # Загружаем нейронную сеть
        #cnn_u=sffU.load_weights_of_cnn(fileName, fileName1, i)
        if (1==0):
            mesh_load_mat = scipy.io.loadmat(baseFolder + "test.mat")
            y_predict = np.array(mesh_load_mat['y_predict'])
            y_predict1 = np.array(mesh_load_mat['y_predict1'])

    y_predict = cnn_s.predict(x_test, batch_size=batch_size)
    #y_predict1 = cnn_s.predict(x_train, batch_size=batch_size)
    accuracy,y_pred_f, y_true_f = sffU.dice_coef2(y_test, y_predict,i,fileNameResult,fileIndex)
    if (pl_norm == 1):
        #Прорисовка результата сегментации в
        sffU.plot_results_image_labes(y_pred_f,INPUT_SHAPE, N_test, fileIndex, fileNameResult1, i)
    g = 0
g1 = 0