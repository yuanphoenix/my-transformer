import paddle
import paddle.nn as nn

# 不知道这个有没有用。。
nn.initializer.set_global_initializer(nn.initializer.Uniform(), nn.initializer.Constant())

from EncoderDecoder import EncoderDecoder
from utils import convert_list_to_tensor


def train():
    english = ['i love you', 'china is a great country', 'i love china', 'china is a country']
    chinese = ['我爱你', '中国是一个伟大的国家', '我爱中国', '中国是一个国家']
    input_ids, _, input_metric = convert_list_to_tensor(english)
    encod_ids, _, encod_metric = convert_list_to_tensor(chinese, endlish=False)
    input_ids = input_ids[:, 1:]
    true_labels = encod_ids[:, 1:]
    encod_ids = encod_ids[:, :-1]
    transformer = EncoderDecoder(vocab_size=26, d_model=512)
    adamw = paddle.optimizer.AdamW(learning_rate=0.001, parameters=transformer.parameters())
    for epoch in range(700):
        output_dict = transformer(input_ids, encod_ids, true_labels, src_mask=input_metric, tgt_mask=encod_metric)
        loss = output_dict['loss']
        print(f"第{epoch + 1}次训练,loss是{loss.item()},logits是{paddle.tolist(output_dict['index'])}")

        adamw.clear_gradients()
        loss.backward()
        adamw.step()
    evaluate(transformer)


@paddle.no_grad()
def evaluate(model: EncoderDecoder, MAX_LENGTH=6):
    model.eval()
    str_list = ['china']
    enput_ids, _, enput_mask = convert_list_to_tensor(str_list)
    enput_ids = enput_ids[:, 1:]

    de_ids = [[0]]
    de_ids = paddle.to_tensor(de_ids)
    for i in range(MAX_LENGTH):
        tgt_mask = paddle.ones([i + 1, i + 1]).unsqueeze(0)

        output_dict = model(enput_ids, de_ids, src_mask=enput_mask, tgt_mask=tgt_mask)

        result = output_dict['index']
        # temp = result[:, -1].item()
        # if temp == 0:
        #     print("结束了")
        #     return
        g = result[:, -1].unsqueeze(0)
        de_ids = paddle.concat((de_ids, g), axis=1)
        print(paddle.tolist(de_ids))

    # input = [1, 2, 3, 4, 5, 6, 8, 0]
    # input = paddle.to_tensor(input).unsqueeze(0)
    # de_input = paddle.to_tensor([0]).unsqueeze(0)
    # for i in range(MAX_LENGTH):
    #     output_dict = model(input, de_input)
    #     result = output_dict['index']
    #     temp = result[:, -1].item()
    #     if temp == 0:
    #         print("结束了")
    #         return
    #     g = result[:, -1].unsqueeze(0)
    #     de_input = paddle.concat((de_input, g), axis=1)
    #     print(paddle.tolist(de_input))


if __name__ == '__main__':
    train()
    # vocab_size = 11
    # original = [0, 1, 2, 3, 4, 5, 6, 8, 0]
    # encode_input = original[1:]
    # decode_input = original[0:-1]
    # encode_input = paddle.to_tensor(encode_input).unsqueeze(0)
    # decode_input = paddle.to_tensor(decode_input).unsqueeze(0)
    #
    # transformer = EncoderDecoder(vocab_size=vocab_size, d_model=512)
    # adamw = paddle.optimizer.AdamW(learning_rate=0.001, parameters=transformer.parameters())
    # for epoch in range(400):
    #     output_dict = transformer(encode_input, label=decode_input, true_label=encode_input)
    #     loss = output_dict['loss']
    #     print(f"第{epoch + 1}次训练，logits是{paddle.tolist(output_dict['index'])},loss是{loss.item()}")
    #     adamw.clear_gradients()
    #     loss.backward()
    #     adamw.step()
    # evaluate(transformer)
