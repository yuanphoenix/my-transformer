import paddle

from EncoderDecoder import EncoderDecoder
from utils import convert, convert_list_to_tensor, PAD_ID

# ---------------- 玩具训练集 ----------------
ENGLISH = ['i love you', 'china is a great country', 'i love china', 'china is a country']
CHINESE = ['我爱你', '中国是一个伟大的国家', '我爱中国', '中国是一个国家']

VOCAB_SIZE = 26          # 与 utils.convert() 的词表大小一致
D_MODEL = 512
MODEL_PATH = 'transformer.pdparams'
EPOCHS = 700
LOG_INTERVAL = 50


def build_data():
    """把训练集转成 (encoder输入, decoder输入, 标签, src_mask, tgt_mask)。

    decoder输入 = 中文去掉末尾结束符; 标签 = 中文去掉开头起始符。二者错一位,
    正是"用上一个词预测下一个词"的自回归目标。
    """
    input_ids, _, input_metric = convert_list_to_tensor(ENGLISH)
    encod_ids, _, encod_metric = convert_list_to_tensor(CHINESE, endlish=False)
    input_ids = input_ids[:, 1:]        # 去掉起始符
    true_labels = encod_ids[:, 1:]      # 标签 = 去掉起始符
    encod_ids = encod_ids[:, :-1]       # decoder输入 = 去掉结束符
    return input_ids, encod_ids, true_labels, input_metric, encod_metric


def save_model(model, path=MODEL_PATH):
    """只存权重(state_dict), 不存整个对象——跨版本更稳、文件更小。"""
    paddle.save(model.state_dict(), path)
    print(f'模型权重已保存到 {path}')


def load_model(path=MODEL_PATH, vocab_size=VOCAB_SIZE, d_model=D_MODEL):
    """新建同结构模型, 再把权重灌回去。"""
    model = EncoderDecoder(vocab_size=vocab_size, d_model=d_model)
    model.set_state_dict(paddle.load(path))
    model.eval()
    print(f'已从 {path} 加载模型权重')
    return model


def train():
    input_ids, encod_ids, true_labels, input_metric, encod_metric = build_data()
    model = EncoderDecoder(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
    adamw = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())

    for epoch in range(EPOCHS):
        out = model(input_ids, encod_ids, true_labels,
                    src_mask=input_metric, tgt_mask=encod_metric)
        loss = out['loss']
        adamw.clear_gradients()
        loss.backward()
        adamw.step()

        if (epoch + 1) % LOG_INTERVAL == 0 or epoch == 0:
            print(f'epoch {epoch + 1}/{EPOCHS}  loss={float(loss):.4f}')

    save_model(model)
    return model


@paddle.no_grad()
def evaluate(model=None, path=MODEL_PATH, max_len=12):
    """评估分两部分:
    1) teacher-forcing: 用标准decoder输入, 算 loss 和非padding位的token准确率(量化指标)
    2) 自回归翻译: 从起始符逐字生成, 解码回中文, 与参考答案对照(看真实输出)
    """
    if model is None:
        model = load_model(path)
    model.eval()

    # ---- 1) teacher-forcing 指标 ----
    input_ids, encod_ids, true_labels, input_metric, encod_metric = build_data()
    out = model(input_ids, encod_ids, true_labels,
                src_mask=input_metric, tgt_mask=encod_metric)
    loss = float(out['loss'])
    pred = out['index']                                  # [B, L]
    non_pad = true_labels != PAD_ID                      # 只统计真实词位置
    correct = int(((pred == true_labels) & non_pad).sum())
    total = int(non_pad.sum())
    acc = correct / total * 100
    print(f'\n[teacher-forcing] loss={loss:.4f}  token准确率={acc:.1f}% ({correct}/{total})')

    # ---- 2) 自回归翻译对照 ----
    _, id2word = convert()
    print('\n[自回归翻译] 模型真正输出的中文:')
    for en, ref in zip(ENGLISH, CHINESE):
        hyp = translate(model, en, id2word, max_len)
        flag = 'OK' if hyp == ref else 'XX'
        print(f'  {en:28s} -> {hyp:12s} 参考: {ref}  [{flag}]')


@paddle.no_grad()
def translate(model, sentence, id2word, max_len=12):
    """对单句英文做自回归解码, 返回解码出的中文字符串。"""
    src_ids, _, src_mask = convert_list_to_tensor([sentence])
    src_ids = src_ids[:, 1:]
    ys = paddle.to_tensor([[0]])                         # 起始符 BOS
    for _ in range(max_len):
        tgt_mask = paddle.ones([1, ys.shape[1], ys.shape[1]])
        out = model(src_ids, ys, src_mask=src_mask, tgt_mask=tgt_mask)
        next_id = int(out['index'][0, -1])
        if next_id == 0:                                 # 生成结束符则停止
            break
        ys = paddle.concat([ys, paddle.to_tensor([[next_id]])], axis=1)
    return decode(ys[0].tolist(), id2word)


def decode(ids, id2word):
    """把id序列拼回中文: 跳过起始符0, 遇padding停止, 中文按字直接拼接。"""
    chars = []
    for i in ids:
        if i == 0:            # 起始符
            continue
        if i == PAD_ID:       # padding
            break
        chars.append(str(id2word[i]))
    return ''.join(chars)


if __name__ == '__main__':
    trained = train()        # 训练 + 保存权重
    evaluate(trained)        # 用刚训好的模型评估
    # 也可以之后单独加载评估(不重新训练):
    # evaluate()             # 默认从 MODEL_PATH 加载
