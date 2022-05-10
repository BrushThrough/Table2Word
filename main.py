# coding:utf-8
from table_recognition.photo_init import Photo_init
from table_recognition.table_structure_recognition import Table_structure_recognition
import argparse
import cv2
import paddlehub as hub
from word_restore import Word_Maker
import os
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir",default="./mini_test_imgs")
    parser.add_argument("--save_dir", default="./word_result")
    args = parser.parse_args()

    file_list = os.listdir(args.test_dir)

    for file in file_list:
        print("正在识别图片：",file)
        img_path = os.path.join(args.test_dir,file)

        img = cv2.imread(img_path)
        photo_init = Photo_init()
        img_init ,img= photo_init(img)
        # cv2.imwrite("test.png", img_init)

        table_recognition = Table_structure_recognition()
        seg_imgs, x_point_arr, y_point_arr, is_table_vertex, \
        table_vertex_horizontal_num, table_vertex_vertical_num=table_recognition(img_init, img)


        ocr = hub.Module(directory="chinese_ocr_db_crnn_mobile")
        db = hub.Module(directory='chinese_text_detection_db_mobile')
        ocr.set_text_detector_module(db)

        ocr_results = ocr.recognize_text(
            images=seg_imgs,  # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
            use_gpu=False,  # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
            output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
            visualization=False,  # 是否将识别结果保存为图片文件；
            box_thresh=0.5,  # 检测文本框置信度的阈值；
            text_thresh=0.5,  # 识别中文文本置信度的阈值；
        )

        file_name = os.path.splitext(file)[0]
        save_dir = os.path.join(args.save_dir,file_name)
        generator = Word_Maker()
        generator(x_point_arr,y_point_arr,is_table_vertex,table_vertex_horizontal_num,
                  table_vertex_vertical_num,ocr_results=ocr_results,dir=save_dir)
        print(file," 识别完成",end="\n----------------\n\n")