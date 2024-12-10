import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow_core.contrib import layers
from utils import getLogger
import time

# set logger

logger = getLogger('zijideming')

'''
每个记忆单元  ：

    先获取注意力权重 --》 根据知识点遗忘矩阵对用户的知识状态进行遗忘操作  ---》 读过程  ---》根据用户对问题的答题情况进行写过程
'''


class MemoryHeadGroup():
    def __init__(self, memory_size, memory_state_dim, is_write, forget_matrix, name="DKVMN-HEAD" , forget_cycle = 6000):
        self.name = name
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.is_write = is_write
        self.forget_matrix = forget_matrix
        self.forget_cycle = forget_cycle

    # 计算当前的题目与学生概念储备的相关性  （查询向量与key矩阵的操作）
    # key_memory_state_dim 为表示每个问题的知识点数量
    # 题目 embedded_query_vector= [0.2,0.5,0.6,0.1,0.1,0.8]   shape=1*key_memory_state_dim
    # batch_size : 送入的每批学生数量
    def correlation_weight(self, embedded_query_vector, key_memory_matrix):
        """
                Given a batch of queries, calculate the similarity between the query and
                each key-memory slot via inner dot product. Then, calculate the weighting
                of each memory slot by softmax function.

                Parameters:
                    - embedded_query_vector (k): Shape (batch_size, key_memory_state_dim)      提取出的问题的嵌入向量
                    - key_memory_matrix (D_k): Shape (memory_size, key_memory_state_dim)       问题与知识点的相关性矩阵
                Result:
                    - correlation_weight (w): Shape (batch_size, memory_size)                  题目应该在每个知识点上投入的权重

                用于计算 查询与习题相关性   （对应论文中的获取注意力权重， 也即读过程）
        """
        embedding_result = tf.matmul(
            embedded_query_vector, tf.transpose(key_memory_matrix)
        )
        correlation_weight = tf.nn.softmax(embedding_result)
        return correlation_weight

    def read(self, value_memory_matrix, correlation_weight):
        value_memory_matrix_reshaped = tf.reshape(value_memory_matrix, [-1,
                                                                        self.memory_state_dim])  # 将数组里面的数据拉直之后进行reshape， 平均分为若干组，每组self.mem_state_dim    个
        correlation_weight_reshaped = tf.reshape(correlation_weight, [-1, 1])  # 将矩阵里面的数据进行拉直  平均分为若干组，每组里面一个

        _read_result = tf.multiply(value_memory_matrix_reshaped, correlation_weight_reshaped)  # row-wise multiplication
        read_result = tf.reshape(_read_result, [-1, self.memory_size, self.memory_state_dim])
        read_result = tf.reduce_sum(read_result, axis=1, keepdims=False)  # 沿着x轴 对矩阵每一行进行求值
        return read_result  # 得到相当于公式中的rt

    # 将nowTime 与 知识点遗忘矩阵进行时间间隔计算，更新学生的知识状态的掌握
    def writeBeforeRead(self, nowTime_matrix, value_memory_matrix):
        '''
         具体操作
        :param nowTime_matrix:    现在做题时间 shape(1 , batch_size)
        :param forget_memory_matrix:  Shape(batch_size,memory_size)
        :param value_memory_matrix:   Shape (batch_size, memory_size, value_memory_state_dim)
        :return:
        '''
        _batch_size = nowTime_matrix.shape[0]
        nowTime_matrix = tf.reshape(nowTime_matrix, [_batch_size, 1])
        nowTime_matrix = tf.tile(nowTime_matrix, [1, self.memory_size])  # ===> batch_size * memory_size
        time_interval_matrix = tf.subtract(nowTime_matrix, self.forget_matrix)
        temp_transfor_forget_matrix = tf.to_float(time_interval_matrix)
        # 计算出知识点的保留率
        retention_rate_forget_matrix = self.forget_fun(temp_transfor_forget_matrix)

        # 将知识点矩阵扩展为与value_memory_matrix维度相同的样子
        retention_rate_forget_matrix = tf.reshape(retention_rate_forget_matrix, [_batch_size, self.memory_size, 1])
        retention_rate_forget_matrix = tf.tile(retention_rate_forget_matrix, [1, 1, self.memory_state_dim])

        # dot
        #temp_value_memory_matrix = tf.multiply(value_memory_matrix, retention_rate_forget_matrix)
        #new_value_memory_matrix = tf.nn.softmax(temp_value_memory_matrix)
        new_value_memory_matrix = tf.multiply(value_memory_matrix, retention_rate_forget_matrix)
        self.forget_matrix = nowTime_matrix
        return new_value_memory_matrix

    def forget_fun(self, time):
        '''
        根据遗忘间隔计算知识的保留率 (遗忘间隔以s为单位)
        :param time:  batch_size * memory_size
        :return:
        '''
        # https://zhuanlan.zhihu.com/p/445497724
        # Retention = -FI / ln(1 - FI)    遗忘指数与保留率之间的关系  × （不采用）
        # FI =  R = exp(-t / S)  遗忘指数  ==》 采用指数式遗忘
        S = self.forget_cycle  # 一小时为间隔设置遗忘周期  (也即假设1小时不记忆知识点  就会遗忘光)
        tempMatrix = tf.constant([[-1]], dtype=float)
        tempMatrix = tf.tile(tempMatrix, [time.shape[0], time.shape[1]])
        time = tf.multiply(time, tempMatrix)  # -t
        tempMatrix = tf.constant([[S]], dtype=float)
        tempMatrix = tf.tile(tempMatrix, [time.shape[0], time.shape[1]])
        time = time / tempMatrix  # -t/S
        time = tf.exp(time)  # FI =  R = exp(-t / S)  遗忘指数
        return time

    def writeAfterRead(self, value_memory_matrix, correlation_weight, embedded_content_vector, reuse=False):
        '''
            基于相关性权重矩阵和学生答复 更新学生知识状态矩阵
        :param value_memory_matrix:  学生知识状态矩阵  Shape (batch_size, memory_size, value_memory_state_dim)
        :param correlation_weight:  知识点相关性权重  Shape (batch_size, memory_size)
        :param embedded_content_vector: 学生真实作答情况  Shape (batch_size, value_memory_state_dim)
        :param reuse:  声明在训练过程中权重是否再次使用
        :return:  new_value_memory_matrix: Shape (batch_size, memory_size, value_memory_state_dim)
        '''
        assert self.is_write  # 如果不可写则，，，，报错

        # 擦除向量
        erase_signal = layers.fully_connected(
            inputs=embedded_content_vector,
            num_outputs=self.memory_state_dim,
            scope=self.name + '/EraseOperation',  # variable_scope的可选范围。
            reuse=reuse,  # 是否应重用图层及其变量。必须给出能够重用层范围的能力。
            activation_fn=tf.sigmoid
        )

        # 附赠向量: Shape (batch_size, value_memory_state_dim)
        add_signal = layers.fully_connected(
            inputs=embedded_content_vector,
            num_outputs=self.memory_state_dim,
            scope=self.name + '/AddOperation',
            reuse=reuse,
            activation_fn=tf.tanh
        )

        # reshape from (batch_size, value_memory_state_dim) to (batch_size, 1, value_memory_state_dim)
        erase_reshaped = tf.reshape(erase_signal, [-1, 1, self.memory_state_dim])
        # reshape from (batch_size, value_memory_state_dim) to (batch_size, 1, value_memory_state_dim)
        add_reshaped = tf.reshape(add_signal, [-1, 1, self.memory_state_dim])
        # reshape from (batch_size, memory_size) to (batch_size, memory_size, 1)
        cw_reshaped = tf.reshape(correlation_weight, [-1, self.memory_size, 1])

        # erase_mul/add_mul: Shape (batch_size, memory_size, value_memory_state_dim)
        erase_mul = tf.multiply(erase_reshaped, cw_reshaped)
        add_mul = tf.multiply(add_reshaped, cw_reshaped)

        # Update value memory
        new_value_memory_matrix = value_memory_matrix * (1 - erase_mul)  # erase memory
        new_value_memory_matrix += add_mul

        return new_value_memory_matrix


class DKVMN():
    def __init__(self, memory_size, key_memory_state_dim, value_memory_state_dim, forget_memory_state_dim,
                 init_key_memory=None, init_value_memory=None, init_forget_memory=None, name="DKVMN" , forget_cycle = 60000):
        '''

        :param memory_size:    知识点个数
        :param key_memory_state_dim:      key矩阵中习题集合向量维度
        :param value_memory_state_dim:   学生的每个知识点向量维度
        :param forget_memory_state_dim:   遗忘元个数 (暂未用)
        :param init_key_memory:          初始化key矩阵
        :param init_value_memory:        初始化学生知识点矩阵(batch_size,[知识点矩阵])
        :param init_forget_memory:       初始化学生与知识点对应的遗忘矩阵 (batch_size, memory_size)
        :param name:
        :param forget_cycle:     遗忘周期
        '''
        self.name = name
        self.memory_size = memory_size
        self.forget_memory_state_dim = forget_memory_state_dim
        self.key_memory_state_dim = key_memory_state_dim
        self.value_memory_state_dim = value_memory_state_dim

        self.key_memory_matrix = init_key_memory
        self.value_memory_matrix = init_value_memory
        self.forget_memory_matrix = init_forget_memory

        # 分别初始化key head与value head  其中key_head没有相关读取功能
        self.key_head = MemoryHeadGroup(
            self.memory_size, self.key_memory_state_dim, forget_matrix=self.forget_memory_matrix,
            name=self.name + '-KeyHead', is_write=False , forget_cycle = forget_cycle
        )
        self.value_head = MemoryHeadGroup(
            self.memory_size, self.value_memory_state_dim, forget_matrix=self.forget_memory_matrix,
            name=self.name + '-ValueHead', is_write=True , forget_cycle = forget_cycle
        )

    def attention(self, embedded_query_vector):
        correlation_weight = self.key_head.correlation_weight(
            embedded_query_vector=embedded_query_vector,
            key_memory_matrix=self.key_memory_matrix
        )
        return correlation_weight

    def read(self, correlation_weight):
        read_content = self.value_head.read(
            value_memory_matrix=self.value_memory_matrix,
            correlation_weight=correlation_weight
        )
        return read_content

    def writeBeforeRead(self, nowTime_vector):
        self.value_memory_matrix = self.value_head.writeBeforeRead(
            nowTime_matrix=nowTime_vector,
            value_memory_matrix=self.value_memory_matrix
        )
        return self.value_memory_matrix

    def writeAfterRead(self, correlation_weight, embedded_result_vector, reuse):
        self.value_memory_matrix = self.value_head.writeAfterRead(
            value_memory_matrix=self.value_memory_matrix,
            correlation_weight=correlation_weight,
            embedded_content_vector=embedded_result_vector,
            reuse=reuse
        )
        return self.value_memory_matrix


if __name__ == '__main__':
    memory = DKVMN(
        memory_size=5,
        key_memory_state_dim=5,
        value_memory_state_dim=5,
        forget_memory_state_dim=5,
        init_key_memory=np.random.random((5, 5)).astype(np.float32),
        init_forget_memory=np.random.random((5, 5)).astype(np.float32),
        init_value_memory=np.random.random((5, 5, 5)).astype(np.float32)
    )
    testTensor = tf.constant([[500],
                              [100],
                              [100],
                              [100],
                              [100]])
    print(testTensor.shape)
    split1 = tf.split(testTensor, num_or_size_splits=1, axis=1)
    transform1 = tf.squeeze(split1[0], 1)
    print(transform1.shape)
    mm = memory.writeBeforeRead(nowTime_vector=transform1)
    xiangguan = memory.attention(tf.constant([
        [1, 3, 5, 4, 0],
        [1, 3, 5, 4, 0],
        [1, 3, 5, 4, 0],
        [1, 3, 5, 4, 0],
        [1, 3, 5, 4, 0],
    ], dtype=float))
    read_resu = memory.read(xiangguan)
    print("the read_result = ")
    print(read_resu)
    write_after = memory.writeAfterRead(xiangguan, tf.constant([
        [1, 0, 1, 1, 0],
        [1, 0, 1, 1, 0],
        [1, 0, 1, 1, 0],
        [1, 0, 1, 1, 0],
        [1, 0, 1, 1, 0],
    ], dtype=float), False)
    # tf.constant([[1,0,2,1,0,1]],dtype=tf.float64)
    print(read_resu)
    logging.info("Initializting q and qa")
    '''
        输入数据一定要是 float32类型的
    '''
    StringTime = "2019-10-11 13:33:00"
    timeArray = time.strptime(StringTime, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    print(timeStamp)
