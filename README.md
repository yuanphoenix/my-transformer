## Transformer

### 关于如何测试
训练过程搞明白了，但是编码器解码器架构如何进行推理没有想通。根据李宏毅说的，解码器是一个字一个字往外崩的，并且
还有开始标记，终止标记。这些虽然相比于Transformer模型来说是细微末节，但是恰恰是这些细节才是模型的临门一脚。
Transformer 在训练的时候确实是并行的，但是在验证的时候确实串行的。