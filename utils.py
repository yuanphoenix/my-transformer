from typing import List

import paddle
from paddle import Tensor


def convert():
    chinese = ['你好吗', "我爱你", "中国是个伟大的国家"]
    english = ['how are you', 'i love you', 'china is a great country']
    word_list = []
    for item in chinese:
        for word in item:
            # 中文一个字一个字的加入list
            word_list.append(word)
    for item in english:
        word_list.extend(item.split())
    word_list = list(set(word_list))
    word_list.insert(0, 0)
    word2id = {item: index for index, item in enumerate(word_list)}
    id2word = {index: item for index, item in enumerate(word_list)}
    return word2id, id2word


def convert_list_to_tensor(str_list: List[str]) -> (Tensor, Tensor):
    """

    :param str_list:
    :return: 原始的id矩阵；处理好了的掩码矩阵
    """
    max_length = 0
    if str_list[0][0].isalpha():
        for item in str_list:
            ll = item.split(' ')
            max_length = len(ll) if len(ll) > max_length else max_length
    else:
        max_length = len(max(str_list, key=len))
    max_length += 2
    word2id, id2word, = convert()
    result = []
    padding_metric = []
    pad = -1
    mask_seq_seq = []
    for sentence in str_list:
        ids = [0, ]  # 开始的标志
        padding_mask = []
        if sentence[0].isalpha():
            # 英文
            word_list = sentence.split(' ')
            for word in word_list:
                ids.append(word2id[word])
        else:
            for word in sentence:
                ids.append(word2id[word])
        padding_mask.extend([1] * len(ids))
        ids.append(0)  # 结束的标志
        pad_nums = max_length - len(ids)
        ids.extend([pad] * pad_nums)
        padding_mask.extend([0] * (len(ids) - len(padding_mask) - 1))
        result.append(ids)
        count = padding_mask.count(1)
        mask_seq_seq.append([[padding_mask] * count])
        padding_metric.append(padding_mask)

    return paddle.to_tensor(result), paddle.to_tensor(padding_metric), paddle.to_tensor(mask_seq_seq)

if __name__ == '__main__':
    result, padding_metric, mask_seq_seq, = convert_list_to_tensor(["i love you", "china i"])
    result.reshape(2,-1)
