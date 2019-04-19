import re


punctuation = re.compile(r"[-~!@#$%^&*()_+`=\[\]\\\{\}\"|;':,./<>?·！@#￥%……&*（）——+【】、；‘：“”，。、《》？「『」』]")


def literal_clean(literal):
    return punctuation.sub('', literal)


if __name__ == '__main__':
    print(literal_clean('[说明：由学生身边熟悉的事物引入新课，容易激发学生的好奇心和求知欲，'
                        '同时又容易使学生产生亲切感，从而带着良好的学习状态进入新课的学习。'))
