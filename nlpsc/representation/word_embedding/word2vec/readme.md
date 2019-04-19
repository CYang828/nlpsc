安装gensim: pip install --upgrade gensim
python word2vec_train.py wiki-zh_cn_segmented word2vec.model word2vec.bin


试试看：
>
Python 3.6.2 (default, Aug  7 2017, 18:50:00)
[GCC 4.2.1 Compatible Apple LLVM 8.1.0 (clang-802.0.42)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import gensim
>>> model = gensim.models.KeyedVectors.load_word2vec_format("/Users/zhazhang/Downloads/zhwiki/word2vec.bin")
>>> len(model['男人'])
300
>>> model['男人']
array([-1.57072365e-01, -5.53397499e-02, -5.00365980e-02,  2.37463728e-01,
        .
        .
        .
        .
        .
        .
       -3.87294620e-01,  1.78412586e-01, -6.70763791e-01, -1.12617269e-01],
      dtype=float32)
>>> model.similarity('男人','女人')
0.7983566649991888
>>> model.n_similarity(["女人", "皇帝"], ["男人", "皇后"])
0.7118446848260049
>>> model.most_similar("女人")
[('男人', 0.7983566522598267), ('女孩', 0.6287996768951416), ('荡妇', 0.6176502108573914), ('最痛', 0.6117048859596252), ('中年男人', 0.5965931415557861), ('少妇', 0.5842644572257996), ('爱情观', 0.5830596089363098), ('不哭', 0.5812167525291443), ('会爱上', 0.5771925449371338), ('男孩', 0.5767508149147034)]
>>> model.most_similar("健身")
[('健体', 0.7552504539489746), ('健身房', 0.7460017800331116), ('桑拿室', 0.7268406748771667), ('桑拿浴', 0.7190618515014648), ('桌球室', 0.7161827683448792), ('按摩室', 0.7140408754348755), ('乒乓球室', 0.7108071446418762), ('阅读室', 0.7044647336006165), ('体操房', 0.6930304169654846), ('舞室', 0.6772329211235046)]
继续训练：
>>> model = gensim.models.Word2Vec.load('word2vec.model')
>>> model.similarity("你好", "阳光")
0.232829629224467
>>> sentences = [['小剑', '你好', '啊'], ['小剑', '阳光', '啊']]
>>> model.train(sentences, total_words=model.corpus_count,epochs=model.iter)
30
>>> model.similarity("你好", "阳光")
0.2556345407150138
注意：这里只对从model.save()恢复的模型有效，从model.save_word2vec_format()恢复过来的模型只能用于查询。