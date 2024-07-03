import pickle

def get_vocab(corpus1, corpus2):
    """
    获取两个语料库中的词汇表。

    参数:
    corpus1: 第一个语料库，格式为嵌套列表。
    corpus2: 第二个语料库，格式为嵌套列表。

    返回:
    word_vocab: 包含两个语料库中所有独特词汇的集合。
    """
    word_vocab = set()  # 初始化一个空的集合来存储词汇
    for corpus in [corpus1, corpus2]:  # 遍历两个语料库
        for i in range(len(corpus)):  # 遍历语料库中的每个元素
            word_vocab.update(corpus[i][1][0])  # 更新词汇集合，添加第一部分的词
            word_vocab.update(corpus[i][1][1])  # 更新词汇集合，添加第二部分的词
            word_vocab.update(corpus[i][2][0])  # 更新词汇集合，添加第三部分的词
            word_vocab.update(corpus[i][3])  # 更新词汇集合，添加第四部分的词
    print(len(word_vocab))  # 输出词汇集合的大小
    return word_vocab  # 返回词汇集合

def load_pickle(filename):
    """
    从pickle文件中加载数据。

    参数:
    filename: 要加载的pickle文件的路径。

    返回:
    data: 从pickle文件中加载的数据。
    """
    with open(filename, 'rb') as f:  # 以二进制读模式打开文件
        data = pickle.load(f)  # 使用pickle加载文件中的数据
    return data  # 返回加载的数据

def vocab_processing(filepath1, filepath2, save_path):
    """
    处理词汇表，移除总词汇表中包含在指定词汇表中的词。

    参数:
    filepath1: 包含排除词汇的文件路径。
    filepath2: 包含总词汇表的文件路径。
    save_path: 处理后词汇表的保存路径。
    """
    with open(filepath1, 'r') as f:  # 打开包含排除词汇的文件
        total_data1 = set(eval(f.read()))  # 读取并解析文件内容为集合
    with open(filepath2, 'r') as f:  # 打开包含总词汇表的文件
        total_data2 = eval(f.read())  # 读取并解析文件内容

    word_set = get_vocab(total_data2, total_data2)  # 获取总词汇表中的所有词汇

    excluded_words = total_data1.intersection(word_set)  # 获取需要排除的词汇
    word_set = word_set - excluded_words  # 从总词汇表中移除排除词汇

    print(len(total_data1))  # 输出排除词汇的数量
    print(len(word_set))  # 输出处理后的词汇表的数量

    with open(save_path, 'w') as f:  # 打开保存路径文件
        f.write(str(word_set))  # 将处理后的词汇表写入文件

if __name__ == "__main__":
    """
    主函数，处理不同类型的词汇表文件，并保存结果。
    """
    python_hnn = './data/python_hnn_data_teacher.txt'  # Python HNN数据文件路径
    python_staqc = './data/staqc/python_staqc_data.txt'  # Python STAQ数据文件路径
    python_word_dict = './data/word_dict/python_word_vocab_dict.txt'  # Python词汇表文件路径

    sql_hnn = './data/sql_hnn_data_teacher.txt'  # SQL HNN数据文件路径
    sql_staqc = './data/staqc/sql_staqc_data.txt'  # SQL STAQ数据文件路径
    sql_word_dict = './data/word_dict/sql_word_vocab_dict.txt'  # SQL词汇表文件路径

    new_sql_staqc = './ulabel_data/staqc/sql_staqc_unlabled_data.txt'  # 新SQL STAQ数据文件路径
    new_sql_large = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'  # 大SQL数据文件路径
    large_word_dict_sql = './ulabel_data/sql_word_dict.txt'  # 大SQL词汇表文件保存路径

    vocab_processing(sql_word_dict, new_sql_large, large_word_dict_sql)  # 处理并保存词汇表
