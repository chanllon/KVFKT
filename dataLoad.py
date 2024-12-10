import numpy as np
from utils import getLogger
import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import time
import matplotlib.pyplot as plt


class DataLoader():
    def __init__(self, n_questions, seq_len, separate_char):
        self.separate_char = separate_char
        self.seq_len = seq_len
        self.n_questions = n_questions

    # data format:
    '''
    2
    2,6
    0,1
    2022-09-07,2022-09-10
    '''

    def load_data(self, path):

        '''
        先对原数据进行处理，在进行加载操作
        Args:
            path:  数据路径

        Returns:
        '''
        # newPath = self.before_loadData(path);

        q_data = []
        qa_data = []
        t_data = []
        with open(path, 'r') as f:
            for line_index, line in enumerate(f):
                line = line.strip()
                # skip the number of sequence
                if line_index % 4 == 0:
                    continue
                # handle the question_line
                elif line_index % 4 == 1:
                    q_tag_list = line.split(self.separate_char)

                # handle the answer_line
                elif line_index % 4 == 2:
                    a_tag_list = line.split(self.separate_char)

                # handle the time line
                elif line_index % 4 == 3:
                    t_tag_list = line.split(self.separate_char)

                    # find the number of split for this sequence
                    n_split = len(q_tag_list) // self.seq_len
                    if len(q_tag_list) % self.seq_len != 0:
                        n_split += 1

                    for k in range(n_split):
                        # temporary container for each sequence
                        q_container = list()
                        qa_container = list()
                        t_container = list()

                        start_idx = k * self.seq_len
                        end_idx = min((k + 1) * self.seq_len, len(a_tag_list))

                        for i in range(start_idx, end_idx):
                            q_value = int(q_tag_list[i])
                            a_value = int(a_tag_list[i])
                            t_value = int(t_tag_list[i])

                            qa_value = int(q_value + a_value * self.n_questions)
                            q_container.append(q_value)
                            qa_container.append(qa_value)
                            t_container.append(t_value)
                        q_data.append(q_container)
                        qa_data.append(qa_container)
                        t_data.append(t_container)

        # convert it to numpy array
        # if there is no value, fill it with zero
        q_data_array = np.zeros((len(q_data), self.seq_len))
        qa_data_array = np.zeros((len(qa_data), self.seq_len))
        t_data_array = np.zeros((len(t_data), self.seq_len))
        for i in range(len(q_data)):
            _q_data = q_data[i]
            _qa_data = qa_data[i]
            _t_data = t_data[i]
            q_data_array[i, :len(_q_data)] = _q_data
            qa_data_array[i, :len(_qa_data)] = _qa_data
            t_data_array[i, :len(_t_data)] = _t_data

        print(qa_data_array)

        return q_data_array, qa_data_array, t_data_array

    def transDateFormat(self, curTime):
        '''
        Args:
            curTime: 当前时间  "2019-10-11 13:33:00.000"
        Returns: 转换后时间戳 12151533213
        '''
        curTime = curTime.split(".")[0]
        timeArrayCur = time.strptime(curTime, "%Y-%m-%d %H:%M:%S")
        timeStampCur = int(time.mktime(timeArrayCur))
        return timeStampCur

    def before_loadData(self):
        '''
        对原先数据格式进行预处理
        Args:
            path:

        Returns:

        '''
        # train_data_path = os.path.join(args.data_dir, args.data_name + '_train{}.csv'.format(i))
        # 进行相应的预处理，处理成对应文件的格式
        # newPath = "./data/student_answer_datail.csv"
        # train_task = pd.read_csv("D:\\AAAAA暨大研究生\\智慧教育\\预选数据集\\train_task_3_4.csv")
        # answer_metadata_task = pd.read_csv("D:\\AAAAA暨大研究生\\智慧教育\\预选数据集\\answer_metadata_task_3_4.csv")
        # student_answer_detail = pd.merge(train_task, answer_metadata_task, on='AnswerId', how='inner')
        # student_answer_detail.sort_values(by='DateAnswered').sort_values(by='UserId').to_csv(newPath, index=0)
        student_answer_detail = pd.read_csv("./data/student_answer_datail.csv")
        # 将时间格式转化为时间戳
        student_answer_detail['newDateAnswered'] = student_answer_detail['DateAnswered'].apply(
            lambda x: self.transDateFormat(x))
        # 得到UserId集合
        student_num = student_answer_detail['UserId'].unique()
        print(student_num)
        target = []
        for i in student_num:
            # 拿到学生i对应题目的做对与做错情况
            tempDf = student_answer_detail[student_answer_detail.UserId == i].loc[:,
                     ['QuestionId', 'IsCorrect', 'newDateAnswered']]
            # 对学生i的做题顺序进行排序
            tempDf = tempDf.sort_values(by='newDateAnswered')
            tempDf = pd.DataFrame(tempDf.values.T, index=tempDf.columns, columns=tempDf.index)
            if (tempDf.shape[1] == 0):
                continue
            '''
                这里要将数据每一份分为150个
            '''
            excise_num_sum = tempDf.shape[1] - 1
            per_excise_num = 50                                   #seq_len
            excise_portion = excise_num_sum // per_excise_num
            problem_id = tempDf.values[0].tolist()
            correctness = tempDf.values[1].tolist()
            answer_date = tempDf.values[2].tolist()
            for j in range(excise_portion - 1):
                target.append([per_excise_num])
                target.append(problem_id[j * per_excise_num: (j + 1) * per_excise_num])
                target.append(correctness[j * per_excise_num: (j + 1) * per_excise_num])
                target.append(answer_date[j * per_excise_num: (j + 1) * per_excise_num])
            print("已经完成了第{}个了.....".format(i))
        # 用该方法可以  但是会卡顿一会
        with open("./data/oper_ok.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in target:
                writer.writerow(row)


if __name__ == '__main__':
    dataLoader = DataLoader(50, 50, ',')
    dataLoader.before_loadData()
    print("  ")
