"""Вспомогательные функции для обучения и тестировния сети U-net в проектах 2020 года
"""

# Импорт библеотеки по работе с матрицами
import numpy as np
# Импорт библеотеки keras для нейросетей
from keras.models import Model
from keras import backend as K
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Convolution2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.convolutional import MaxPooling2D
# Импорт библеотеки для прорисовки результатов
import matplotlib.pyplot as plt
import cv2
# Импорт библеотеки для работы с .mat файлами
import scipy.io



def name_of_results(pl_sphere_cyl):
    """Функция для сохранения имени результатов этапов метода"""
    if (pl_sphere_cyl[0]==1)&(pl_sphere_cyl[1]==1):
        name_file='Model_Obr_ol_rus_1_t'
    elif (pl_sphere_cyl[0]==1)&(pl_sphere_cyl[1]==2):
        name_file='Model_Obr_ol_rus_5_t'
    elif  (pl_sphere_cyl[0]==1)&(pl_sphere_cyl[1]==3):
        name_file='Model_Obr_ol_rus_4_t'
    elif  (pl_sphere_cyl[0]==1)&(pl_sphere_cyl[1]==4):
        name_file='Model_Obr_ol_rus_2_t'
    elif (pl_sphere_cyl[0]==2)&(pl_sphere_cyl[1]==1):
        name_file='Model_Obr_ol_rus_1'
    elif  (pl_sphere_cyl[0]==2)&(pl_sphere_cyl[1]==2):
        name_file='Model_Obr_ol_rus_5'
    elif (pl_sphere_cyl[0]==2)&(pl_sphere_cyl[1]==3):
        name_file='Model_Obr_ol_rus_4'
    elif (pl_sphere_cyl[0]==2)&(pl_sphere_cyl[1]==4):
        name_file='Model_Obr_ol_rus_2'
    elif  (pl_sphere_cyl[0]==3)&(pl_sphere_cyl[1]==2):
        name_file='Model_Sector_3_blades_t'
    elif  (pl_sphere_cyl[0]==3)&(pl_sphere_cyl[1]==1):
        name_file='Model_turbine_blades_t'
    elif  (pl_sphere_cyl[0]==4)&(pl_sphere_cyl[1]==1):
        name_file='Model_turbine_blades'
    elif  (pl_sphere_cyl[0]==4)&(pl_sphere_cyl[1]==2):
        name_file='Model_Sector_3_blades'
    elif  (pl_sphere_cyl[0]==5)&(pl_sphere_cyl[1]==1):
        name_file='Three_spheres_radius64148'
    elif  (pl_sphere_cyl[0]==6)&(pl_sphere_cyl[1]==1):
        name_file='Sphere_plane'
    return name_file


def names_of_files_and_patches(name_file, N_test, fileIndex):
    """Функция для формирования массива названия путей и имен файлов в программе"""
    fileNameTest = "Data_for_semantic_segmentation_test_" + name_file + str(N_test)
    fileNameTrain = "Data_for_semantic_segmentation_train_"
    fileNameResult1 = "D:\\Vadim\\Semantic_segmentation_images_2020\\result_" + name_file + "U_net"
    fileNameTemplate = "Semantic_segmentation_U_net_" + name_file
    fileNameResult = "D:\\Vadim\\Семантическая сегментация нейросети\\2020\\U_net\\results\\" + name_file + "_pred_Unet_"
    baseFolder = "D:\\Vadim\\Семантическая сегментация нейросети\\2020\\Подготовка данных для сегментации_2020\\"
    baseFolder_ = "D:\\Vadim\\Семантическая сегментация нейросети\\2020\\U_net\\results\\"
    fileName = baseFolder_ + fileNameTemplate + str(fileIndex) + ".json"
    fileName1 = baseFolder_ + fileNameTemplate + str(fileIndex) + ".h5"
    return fileNameTest, fileNameTrain, fileNameResult1, fileNameTemplate, fileNameResult, baseFolder, baseFolder_,\
            fileName, fileName1


def make_data_for_train_and_predict(N, N_test,path_train, path_train_label, path_test, imgtype,height,width,ch):
    """Подготавливает данные для сегментации"""
    from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
    x_train = np.zeros((N, height, width, ch))
    y_train1 = np.zeros((N, height, width, ch))
    y_train = np.zeros((N, height, width, 1))
    x_test = np.zeros((N_test, height, width, ch))

    k = np.array([[[0.2989, 0.587, 0.114]]])
    for i in range(N):
        im_ = load_img(path_train+"/"+str(i)+"."+imgtype)
        im = img_to_array(im_)
        im_tr_= load_img(path_train_label+"/"+str(i)+"."+imgtype)
        im_tr = img_to_array(im_tr_)
        if (i==0):
            L= im.shape[0]
            H = im.shape[1]
            ch1=im_tr.shape[2]
        if (abs(height-L)>0)and(abs(width-H)>0):
            x_train[i, :, :, :] = cv2.resize(im, (height, width))
            y_train1[i, :, :, :] = cv2.resize(im_tr, (height, width))
        else:
            x_train[i, :, :, :] = im
            y_train1[i, :, :, :] = im_tr
        # Преобразование в черно-белый цвет
        if (ch1==3):
            y_train[0, :, :, 0] = np.round(np.sum(y_train1[0, :, :, :] * k, axis=2)).astype(np.uint8)

    for i in range(N_test):
        im_ = load_img(path_test + "/" + str(i) + "." + imgtype)
        im = img_to_array(im_)
        if (i==0):
            L= im.shape[0]
            H = im.shape[1]
        if (abs(height-L)>0)and(abs(width-H)>0):
            x_test[i, :, :, :] = cv2.resize(im, (height, width))
        else:
            x_test[i, :, :, :] = im
        # Нормализация
        x_train /= 255
        y_train /= 255
        x_test /= 255
        return x_train,y_train,x_test, im

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef1(y_true, y_pred,i,fileNameResult,fileIndex):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    coef = (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)
    Name=fileNameResult+str(i+1)+str(fileIndex) + '.mat'
    scipy.io.savemat(Name, dict(
        [('y_true'+str(i+1), y_true), ('y_pred'+str(i+1), y_pred), ('y_true_f'+str(i+1), y_true_f), ('y_pred_f'+str(i+1), y_pred_f),
            ('intersection'+str(i+1), intersection), ('coef'+str(i+1), coef)]))
    return coef


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def load_from_mat_train_and_test(mat, mat1, i, pl_norm, pl, pl_real_teor):
    """Функция для формирования тестовой и обучающей выборок после загрузки, а так же входного формата"""
    if (i==0):
        x_train  = np.array(mat['x_train1'])
        y_train1 = np.array(mat['y_train1'])
        if (pl_real_teor==0):
            x_test = np.array(mat['x_test1'])
            y_test1 = np.array(mat['y_test1'])
        else:
            x_test = np.array(mat1['x_test1'])
            y_test1 = np.array(mat1['y_test1'])

    elif (i==1):
        x_train = np.array(mat['x_train2'])
        y_train1 = np.array(mat['y_train2'])
        if (pl_real_teor == 0):
            x_test = np.array(mat['x_test2'])
            y_test1 = np.array(mat['y_test2'])
        else:
            x_test = np.array(mat1['x_test2'])
            y_test1 = np.array(mat1['y_test2'])
    elif (i==2):
        x_train = np.array(mat['x_train3'])
        y_train1 = np.array(mat['y_train3'])
        if (pl_real_teor == 0):
            x_test = np.array(mat['x_test3'])
            y_test1 = np.array(mat['y_test3'])
        else:
            x_test = np.array(mat1['x_test3'])
            y_test1 = np.array(mat1['y_test3'])
    elif (i == 3):
        x_train = np.array(mat['x_train4'])
        y_train1 = np.array(mat['y_train4'])
        if (pl_real_teor == 0):
            x_test = np.array(mat['x_test4'])
            y_test1 = np.array(mat['y_test4'])
        else:
            x_test = np.array(mat1['x_test4'])
            y_test1 = np.array(mat1['y_test4'])
    elif (i == 4):
        x_train = np.array(mat['x_train5'])
        y_train1 = np.array(mat['y_train5'])
        if (pl_real_teor == 0):
            x_test = np.array(mat['x_test5'])
            y_test1 = np.array(mat['y_test5'])
        else:
            x_test = np.array(mat1['x_test5'])
            y_test1 = np.array(mat1['y_test5'])
    elif (i == 5):
        x_train = np.array(mat['x_train6'])
        y_train1 = np.array(mat['y_train6'])
        if (pl_real_teor == 0):
            x_test = np.array(mat['x_test6'])
            y_test1 = np.array(mat['y_test6'])
        else:
            x_test = np.array(mat1['x_test6'])
            y_test1 = np.array(mat1['y_test6'])

    N = x_train.shape[0]
    N_test = x_test.shape[0]
    height = x_train.shape[1]
    width = x_train.shape[2]
    y_train = np.zeros((N, height, width, 1))
    y_test = np.zeros((N_test, height, width, 1))
    for ij in range(N):
        y_train[ij, :, :, 0] = y_train1[ij, :, :]
    for ij in range(N_test):
        y_test[ij, :, :, 0] = y_test1[ij, :, :]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    x_train /= 255
    x_test /= 255
    if (pl_norm == 1):
        y_train /= 255
        y_test /= 255

    L = height  # im_rshp.shape[0]
    H = width  # im_rshp.shape[1]
    ch = 3  # im_rshp.shape[2]

    if (pl == 1):
        fig = plt.figure(figsize=(8, 3))
        ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
        plt.imshow(y_train[0, :, :, :])

        ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
        plt.imshow(x_train[0, :, :, :])
        plt.show()

    INPUT_SHAPE = [L, H, ch]
    return x_train, y_train, x_test, y_test, INPUT_SHAPE

def plot_results_image(y_predict, N_test, fileIndex, fileNameResult1, i):
    """Функция для прорисовки решений после сегментации"""
    for ij in range(0, N_test, 5):
        y_predict_img = (y_predict[ij, :, :, 0] * 255.).astype(np.uint8)
        S = fileNameResult1 + '_' + str(i + 1) + '_' + str(ij+1) + '_' + str(fileIndex) + '.png'
        cv2.imwrite(S, y_predict_img)

def save_weights_of_cnn(cnn_u, fileName,fileName1, i):
    """Функция сохранения нейросети"""
    print("Сохраняем сеть")
    # Сохраняем сеть для последующего использования
    # Генерируем описание модели в формате json
    if (1 == 1):
        model_json = cnn_u.to_json()
        json_file = open(fileName + str(i + 1), "w")
        # Записываем архитектуру сети в файл
        json_file.write(model_json)
        json_file.close()
        # Записываем данные о весах в файл
        cnn_u.save_weights(fileName1 + str(i + 1))
    print("Сохранение сети завершено")

def load_weights_of_cnn(fileName, fileName1, i):
    """Функция загрузки нейросети"""
    from keras.models import model_from_json

    print("Загружаю сеть из файлов")
    # Загружаем данные об архитектуре сети
    json_file = open(fileName + str(i + 1), "r")
    loaded_model_json = json_file.read()
    json_file.close()
    # Создаем модель
    cnn_u = model_from_json(loaded_model_json)
    # Загружаем сохраненные веса в модель
    cnn_u.load_weights(fileName1 + str(i + 1))
    print("Загрузка сети завершена")
    # Компилируем загруженную модель
    cnn_u.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])
    return cnn_u

def unet4(INPUT_SHAPE, axis_direction,pl_norm):  # (img_channels, image_rows, image_cols):
    # th - (channels, height, width), tf - (height, width, channels)
    if (1 == 1):
        inputs = Input(shape=INPUT_SHAPE)
        # INPUT_SHAPE1=(800, 800,3)
        # inputs = Input(shape=INPUT_SHAPE1)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=axis_direction)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=axis_direction)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=axis_direction)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=axis_direction)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        if (pl_norm == 1):
            conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
            model = Model(inputs=[inputs], outputs=[conv10])
            # model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])
            model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        else:
            conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)  # num_classes,
            model = Model(inputs=[inputs], outputs=[conv10])
            model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

