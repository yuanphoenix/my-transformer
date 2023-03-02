import paddle

if __name__ == '__main__':
    a = paddle.to_tensor([[[1, 1, 0], [1, 1, 0], [0, 0, 0]], [[1, 1, 0], [1, 1, 0], [0, 0, 0]]])
    print(a)
    print(paddle.count_nonzero(a, axis=-1))
