# coding:utf-8
import cv2
from scipy import misc, ndimage
import numpy as np
import math
import os

class Table_structure_recognition():
    def __init__(self):
        self.dir = np.zeros((8, 2))
        self.dir[0][0] = 1
        self.dir[1][0] = -1
        self.dir[2][1] = 1
        self.dir[3][1] = -1
        self.dir[4][0] = 1
        self.dir[4][1] = 1
        self.dir[5][0] = 1
        self.dir[5][1] = -1
        self.dir[6][0] = -1
        self.dir[6][1] = 1
        self.dir[7][0] = -1
        self.dir[7][1] = -1
        self.seg_imgs=[]

    def __call__(self, binary, img):
        self.img = img
        self.binary = binary

        # 自适应获取核值 识别横线
        h, w = binary.shape
        # print(rows,cols)
        scale = 30
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // scale, 1))
        # print(kernel.shape)
        eroded = cv2.erode(binary, kernel, iterations=1)

        new_scale = 25
        new_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // new_scale, 1))
        dilated_col = cv2.dilate(eroded, new_kernel, iterations=1)

        # cv2.imshow("", dilated_col)
        # cv2.waitKey()
        # cv2.imwrite("w.jpg",dilated_col)

        # 识别竖线
        scale = 30
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // scale))
        eroded = cv2.erode(binary, kernel, iterations=1)
        new_scale = 25
        new_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // new_scale))
        dilated_row = cv2.dilate(eroded, new_kernel, iterations=1)
        # cv2.imshow("excel_vertical_line", dilated_row)
        # cv2.waitKey(0)
        # cv2.imwrite("h.jpg",dilated_row)


        #提取框架特征
        self.frame = cv2.bitwise_or(dilated_col,dilated_row)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        self.frame = cv2.dilate(self.frame, kernel, iterations=1)
       # cv2.imwrite("frame.png", self.frame)


        # 提取交点特征
        self.bitwise_and = cv2.bitwise_and(dilated_col, dilated_row)
        # cv2.imshow("excel_bitwise_and", bitwise_and)
        # cv2.waitKey(0)

        # 识别黑白图中的白色交叉点，将横纵坐标取出
        ys, xs = np.where(self.bitwise_and > 0)

        # 缩点，细化交点特征
        for i in range(len(xs)):
            if (self.bitwise_and[ys[i]][xs[i]]):
                self.shrink_point(ys[i], xs[i])
                self.bitwise_and[ys[i]][xs[i]] = 255

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        # self.bitwise_and=cv2.dilate(self.bitwise_and,kernel)
        # cv2.imwrite("bitwise_and.jpg",self.bitwise_and)
        #cv2.imwrite("bitwise_and.jpg", self.bitwise_and)
        # cv2.imshow("test",self.bitwise_and)
        # cv2.waitKey(0)

        # 提取横纵坐标特征
        ys, xs = np.where(self.bitwise_and > 0)
        # 纵坐标
        self.y_point_arr = []
        # 横坐标
        self.x_point_arr = []

        # 对X和Y坐标排序和判重
        i = 0
        sort_x_point = np.sort(xs)
        for i in range(len(sort_x_point) - 1):
            if sort_x_point[i + 1] - sort_x_point[i] > 10:
                self.x_point_arr.append(sort_x_point[i])
        self.x_point_arr.append(sort_x_point[i])

        i = 0
        sort_y_point = np.sort(ys)
        # print(np.sort(ys))
        for i in range(len(sort_y_point) - 1):
            if sort_y_point[i + 1] - sort_y_point[i] > 10:
                self.y_point_arr.append(sort_y_point[i])
        self.y_point_arr.append(sort_y_point[i])

        # print('y_point_arr', y_point_arr)
        # print('x_point_arr', x_point_arr)

        # 对表格交点横竖坐标特征进行细化
        num = 0
        num_y = np.zeros(len(self.y_point_arr))
        num_x = np.zeros(len(self.x_point_arr))
        for i in range(len(self.y_point_arr) - 1):
            for j in range(len(self.x_point_arr) - 1):
                if self.judge_ver_exist(self.y_point_arr[i], self.x_point_arr[j]) == 0:
                    continue
                aim_i = 0
                aim_j = 0
                for ti in range(i + 1, len(self.y_point_arr)):
                    for tj in range(j + 1, len(self.x_point_arr)):
                        if self.judge_ver_exist(self.y_point_arr[ti], self.x_point_arr[tj]) == 0:
                            continue
                        if (self.judge_horizontal_line(self.y_point_arr[i], self.x_point_arr[j],
                                                       self.x_point_arr[tj]) and
                                self.judge_horizontal_line(self.y_point_arr[ti], self.x_point_arr[j],
                                                           self.x_point_arr[tj]) and
                                self.judge_vertical_line(self.x_point_arr[j], self.y_point_arr[i],
                                                         self.y_point_arr[ti]) and
                                self.judge_vertical_line(self.x_point_arr[tj], self.y_point_arr[i],
                                                         self.y_point_arr[ti]) and
                                (self.y_point_arr[ti] - self.y_point_arr[i]) > 10 and
                                (self.x_point_arr[tj] - self.x_point_arr[j]) > 10):
                            aim_i = ti
                            aim_j = tj
                            break
                    if aim_i:
                        break
                if aim_i == 0:
                    continue
                num_y[i] += 1
                num_y[aim_i] +=1
                num_x[j] += 1
                num_x[aim_j] += 1
                num += 1
        tmp_y=[]
        tmp_x=[]
        for i in range(len(self.y_point_arr)):
            if num_y[i]:
                tmp_y.append(self.y_point_arr[i])
        for i in range(len(self.x_point_arr)):
            if num_x[i]:
                tmp_x.append(self.x_point_arr[i])
        self.y_point_arr = tmp_y
        self.x_point_arr = tmp_x
        print('x_point_arr:', self.x_point_arr)
        print('y_point_arr:', self.y_point_arr)
        print("横坐标数量：",len(self.x_point_arr))
        print('纵坐标数量：',len(self.y_point_arr))
        print("单元格数量：",num)

        #提取单元格特征
        self.is_table_vertex = [[0] * len(self.x_point_arr) for _ in range(len(self.y_point_arr))]
        self.table_vertex_horizontal_num = [[0] * len(self.x_point_arr) for _ in range(len(self.y_point_arr))]
        self.table_vertex_vertical_num = [[0] * len(self.x_point_arr) for _ in range(len(self.y_point_arr))]
        os.chdir("seg_images")
        for i in range(len(self.y_point_arr) - 1):
            for j in range(len(self.x_point_arr) - 1):
                if self.judge_ver_exist(self.y_point_arr[i], self.x_point_arr[j]) == 0:
                    continue
                aim_i = 0
                aim_j = 0
                for ti in range(i + 1, len(self.y_point_arr)):
                    for tj in range(j + 1, len(self.x_point_arr)):
                        # if i == 0 and j == 0\
                        #     and ti == 1 and tj==1:
                        #     print(self.judge_ver_exist(self.y_point_arr[i], self.x_point_arr[j]))
                        #     print(self.judge_ver_exist(self.y_point_arr[ti], self.x_point_arr[tj]))
                        #     print(self.judge_horizontal_line(self.y_point_arr[i], self.x_point_arr[j], self.x_point_arr[tj]))
                        if self.judge_ver_exist(self.y_point_arr[ti], self.x_point_arr[tj]) == 0:
                            continue
                        if (self.judge_horizontal_line(self.y_point_arr[i], self.x_point_arr[j], self.x_point_arr[tj]) and
                            self.judge_horizontal_line(self.y_point_arr[ti], self.x_point_arr[j], self.x_point_arr[tj]) and
                            self.judge_vertical_line(self.x_point_arr[j], self.y_point_arr[i], self.y_point_arr[ti]) and
                            self.judge_vertical_line(self.x_point_arr[tj], self.y_point_arr[i], self.y_point_arr[ti]) and
                                (self.y_point_arr[ti] - self.y_point_arr[i]) > 10 and
                                (self.x_point_arr[tj] - self.x_point_arr[j]) > 10):
                            aim_i = ti
                            aim_j = tj
                            break
                    if aim_i:
                        break
                if aim_i == 0:
                    continue
                self.is_table_vertex[i][j] = 1
                self.table_vertex_vertical_num[i][j] = aim_i - i
                self.table_vertex_horizontal_num[i][j] = aim_j - j
                # print(y_point_arr[i],x_point_arr[j])
                # print(y_point_arr[aim_i],x_point_arr[aim_j])
                # 在分割时，第一个参数为y坐标，第二个参数为x坐标
                cell = self.img[self.y_point_arr[i]:self.y_point_arr[aim_i], self.x_point_arr[j]:self.x_point_arr[aim_j]]
                self.seg_imgs.append(cell)
                # cv2.imshow("seg",cell)
                # cv2.imshow("seg_2", cell_2)
                # cv2.waitKey(0)
                # cv2.imwrite(str(len(self.seg_imgs)) + ".png", cell)
        os.chdir("..")
        return self.seg_imgs, self.x_point_arr,  self.y_point_arr, self. is_table_vertex,\
               self.table_vertex_horizontal_num,self.table_vertex_vertical_num


    def shrink_point(self, y, x):
        self.bitwise_and[y][x] = 0
        for i in range(8):
            tmp_y = int(self.dir[i][0]) + y
            tmp_x = int(self.dir[i][1]) + x
            if (tmp_y < self.bitwise_and.shape[0] and
                tmp_x < self.bitwise_and.shape[1] and
                    tmp_y  and tmp_x and
                self.bitwise_and[tmp_y][tmp_x]):
                self.shrink_point(tmp_y, tmp_x)

    def judge_ver_exist(self, y, x):
        if self.frame[y][x]:
            return 1
        for i in range(8):
            tmp_y = int(self.dir[i][0]) + y
            tmp_x = int(self.dir[i][1]) + x
            if self.frame[tmp_y][tmp_x]:
                return 1
        return 0

    def judge_vertical_line(self, x, y_min, y_max):
        for i in range(30):
            if (self.judge_ver_exist(y_min + (y_max - y_min) // 30 * i, x)==0):
                return 0
        return 1

    def judge_horizontal_line(self, y, x_min, x_max):
        for i in range(30):
            if (self.judge_ver_exist(y, x_min + (x_max - x_min) // 30 * i)==0):
                return 0
        return 1

