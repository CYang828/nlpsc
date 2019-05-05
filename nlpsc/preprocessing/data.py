import re


punctuation = re.compile(r"[-~!@#$%^&*()_+`=\[\]\\\{\}\"|;':,./<>?·！@#￥%……&*（）——+【】、；‘：“”，。、《》？「『」』]")
wrap = re.compile(r"(\n+)")


def literal_clean(literal):
    return wrap.sub('\n', punctuation.sub('', literal).replace(' ', ''))


if __name__ == '__main__':
    print(literal_clean('[说明：由学生身边熟悉的事物引入新课，容易激发学生的好奇心和求知欲，'
                        '同时又容易使学生产生亲切感，从而带着良好的学习状态进入新课的学习。'))
