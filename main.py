# This is a sample Python script.
import random
from math import floor

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from numpy import array, int32
from copy import copy, deepcopy
import matplotlib.pyplot as plt


def test():
    a = np.array([[1, 1], [2, 2], [3, 3]])
    np.insert(a, [1, 1], [5, 2], axis=0)
    print(a)


def read_image_pgm():
    # Use a breakpoint in the code line below to debug your script.*
    file_name = input('Enter the image name : ')
    with open(file_name, 'r') as f:
        file_lines = f.readlines()
    f.close()
    words = file_lines[2].split()
    columns = int(words[0])
    lines = int(words[1])
    print("type image", file_lines[0])
    print("comment image", file_lines[1])
    print("columns image", columns)
    print("lines image", lines)
    print("max value image", file_lines[3])
    matrix = [[0 for x in range(columns)] for y in range(lines)]
    for i in range(4, len(file_lines)):
        ll = file_lines[i].split()
        for j in range(0, len(ll)):
            matrix[i - 4][j] = int(ll[j])

    # print(array(matrix))
    # print(array(matrix).shape)
    return array(matrix)


def write_image_pgm(img, filename, maxVal=255, magicNum='P2'):
    img = int32(img).tolist()
    image = open(filename + ".pgm", 'w')
    file = open(filename + ".txt", "w+")
    content = str(img)
    file.write(content)
    file.close()
    width = 0
    height = 0
    for row in img:
        height = height + 1
        width = len(row)
    image.write(magicNum + '\n')
    image.write(str(width) + ' ' + str(height) + '\n')
    image.write(str(maxVal) + '\n')
    for i in range(height):
        count = 1
        for j in range(width):
            image.write(str(img[i][j]) + ' ')
            if count >= 17:
                # No line should contain gt 70 chars (17*4=68)
                # Max three chars for pixel plus one space
                count = 1
                image.write('\n')
            else:
                count = count + 1
        image.write('\n')
    image.close()


# la moyenne d’une image en niveaux de gris
def average_gris(matrice):
    somme = 0
    for i in range(0, matrice.shape[0]):
        for j in range(0, matrice.shape[1]):
            somme += matrice[i][j]

    return somme / (matrice.shape[0] * matrice.shape[1])


# l’écart type d’une image en niveaux de gris
def standard_deviation_gris(matrix):
    somme = 0
    moy = average_gris(matrix)
    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            somme += (matrix[i][j] - moy) ** 2
    return np.sqrt(somme / (matrix.shape[0] * matrix.shape[1]))


# l’histogramme de niveaux de gris
def histogram(img):
    arr = np.zeros(256)
    for el in img:
        for num in el:
            arr[num] += 1
    return arr


# cumulated histogram
def histogram_cumulated(img):
    arr = histogram(img)
    arr_cumulated = np.zeros(256)
    amount = 0
    for i, el in enumerate(arr):
        amount += el
        arr_cumulated[i] = amount
    return arr_cumulated


# cumulated probability
def pc(img):
    arr = np.zeros(256)
    hc = histogram_cumulated(img)
    resolution = img.shape[0] * img.shape[1]
    for index, el in enumerate(hc):
        arr[index] = el / resolution
    return arr


def get_n1(img):
    arr_pc = pc(img)
    arr_n1 = np.zeros(256)
    for i, el in enumerate(arr_pc):
        arr_n1[i] = floor(255 * el)
    return arr_n1


def equalizer(img):
    histogram_arr = histogram(img)
    n1 = get_n1(img)
    arr_eq = np.zeros(256)
    k = 0
    for i, el in enumerate(histogram_arr):
        while k < 256 and n1[k] == i:
            arr_eq[i] += histogram_arr[k]
            k += 1
    return arr_eq


def equalize_image(img):
    lines = img.shape[0]
    columns = img.shape[1]
    n1 = get_n1(img)
    for i in range(lines):
        for j in range(columns):
            index = img[i][j]
            img[i][j] = n1[index]
    write_image_pgm(img, "newImage")
    return img
    # Press the green button in the gutter to run the script.


# creating function that add noise to the given image
def noisify_image(img):
    lines = img.shape[0]
    columns = img.shape[1]
    for i in range(lines):
        for j in range(columns):
            value = random.randint(0, 20)
            if value == 0:
                img[i][j] = 0
            if value == 20:
                img[i][j] = 255
            print(value)
    write_image_pgm(img, "noisyImage")
    return img


def calculate_average_filter(index1, index2, img, mask_size):
    mask_limit = mask_size // 2
    res = 0
    mask_value = 1 / mask_size
    for i in range(index1 - mask_limit, index1 + mask_limit):
        for j in range(index2 - mask_limit, index2 + mask_limit):
            res += img[i][j] * mask_value
    return res


# applying average filter to an image
def average_filter(img, mask_size):
    lines = img.shape[0]
    columns = img.shape[1]
    mask_limit = mask_size // 2
    matrix_result = deepcopy(img)

    for i in range(mask_limit, mask_size - mask_limit):
        for j in range(mask_limit, mask_size - mask_limit):
            matrix_result[i][j] = calculate_average_filter(i, j, img, mask_size)
    write_image_pgm(img, "FilteredImage")
    return matrix_result


if __name__ == '__main__':
    # print_hi('PyCharm')
    image_matrix = read_image_pgm()
    # write_image_pgm(image_matrix, 'result')
    # print('La moyenne : ', average_gris(image_matrix))
    # print('L’écart type : ', standard_deviation_gris(image_matrix))
    # print('Histogramme : ', histogram(image_matrix))
    # print('Histogramme cumulé : ', histogram_cumulated(image_matrix))
    # print('pc : ', pc(image_matrix))
    # print('n1 : ', get_n1(image_matrix))
    # print('histo egalisé : ', equalize_image(image_matrix))
    img_noisy = noisify_image(image_matrix)
    #average_filter(img_noisy, 3)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
