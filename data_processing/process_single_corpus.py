import pickle  # 用于序列化和反序列化Python对象
from collections import Counter  # 用于计数可哈希对象

def load_pickle(filename):
    """
    从pickle文件中加载数据。

    参数:
    filename: str，pickle文件的路径。

    返回:
    data: 反序列化后的数据。
    """
    with open(filename, 'rb') as f:  # 以二进制读取模式打开文件
        data = pickle.load(f, encoding='iso-8859-1')  # 使用ISO-8859-1编码反序列化对象
    return data  # 返回反序列化后的数据

def split_data(total_data, qids):
    """
    将数据根据qid的数量分为单个和多个两类。

    参数:
    total_data: list，总的数据列表。
    qids: list，数据中包含的qid列表。

    返回:
    total_data_single: list，包含单个qid的数据。
    total_data_multiple: list，包含多个qid的数据。
    """
    result = Counter(qids)  # 计算每个qid出现的次数
    total_data_single = []  # 初始化单个qid数据列表
    total_data_multiple = []  # 初始化多个qid数据列表
    
    for data in total_data:  # 遍历总数据
        if result[data[0][0]] == 1:  # 如果qid出现次数为1
            total_data_single.append(data)  # 加入单个qid数据列表
        else:
            total_data_multiple.append(data)  # 否则加入多个qid数据列表
    
    return total_data_single, total_data_multiple  # 返回分类后的数据

def data_staqc_processing(filepath, save_single_path, save_multiple_path):
    """
    处理staqc数据，将其分为单个和多个，并保存到指定路径。

    参数:
    filepath: str，要处理的staqc数据文件路径。
    save_single_path: str，保存单个qid数据的文件路径。
    save_multiple_path: str，保存多个qid数据的文件路径。
    """
    with open(filepath, 'r') as f:  # 打开并读取文件内容
        total_data = eval(f.read())  # 使用eval将字符串转换为Python对象
    
    qids = [data[0][0] for data in total_data]  # 获取所有数据的qid
    total_data_single, total_data_multiple = split_data(total_data, qids)  # 将数据分为单个和多个
    
    with open(save_single_path, "w") as f:  # 打开保存单个qid数据的文件
        f.write(str(total_data_single))  # 将单个qid数据写入文件
    
    with open(save_multiple_path, "w") as f:  # 打开保存多个qid数据的文件
        f.write(str(total_data_multiple))  # 将多个qid数据写入文件

def data_large_processing(filepath, save_single_path, save_multiple_path):
    """
    处理大数据集，将其分为单个和多个，并保存为pickle文件。

    参数:
    filepath: str，要处理的pickle数据文件路径。
    save_single_path: str，保存单个qid数据的pickle文件路径。
    save_multiple_path: str，保存多个qid数据的pickle文件路径。
    """
    total_data = load_pickle(filepath)  # 加载pickle文件
    qids = [data[0][0] for data in total_data]  # 获取所有数据的qid
    total_data_single, total_data_multiple = split_data(total_data, qids)  # 将数据分为单个和多个
    
    with open(save_single_path, 'wb') as f:  # 打开保存单个qid数据的pickle文件
        pickle.dump(total_data_single, f)  # 将单个qid数据序列化并写入文件
    
    with open(save_multiple_path, 'wb') as f:  # 打开保存多个qid数据的pickle文件
        pickle.dump(total_data_multiple, f)  # 将多个qid数据序列化并写入文件

def single_unlabeled_to_labeled(input_path, output_path):
    """
    将单个未标记数据转换为带有标签的数据。

    参数:
    input_path: str，输入的pickle数据文件路径。
    output_path: str，输出的带标签数据文件路径。
    """
    total_data = load_pickle(input_path)  # 加载pickle文件
    labels = [[data[0], 1] for data in total_data]  # 为每个数据项添加标签1
    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1]))  # 根据数据项和标签对数据进行排序
    
    with open(output_path, "w") as f:  # 打开保存带标签数据的文件
        f.write(str(total_data_sort))  # 将排序后的数据写入文件

if __name__ == "__main__":
    """
    主函数入口，处理不同类型的数据文件并保存结果。
    """
    # 定义staqc数据的路径和保存路径
    staqc_python_path = './ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_single_save = './ulabel_data/staqc/single/python_staqc_single.txt'
    staqc_python_multiple_save = './ulabel_data/staqc/multiple/python_staqc_multiple.txt'
    data_staqc_processing(staqc_python_path, staqc_python_single_save, staqc_python_multiple_save)  # 处理staqc Python数据

    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_single_save = './ulabel_data/staqc/single/sql_staqc_single.txt'
    staqc_sql_multiple_save = './ulabel_data/staqc/multiple/sql_staqc_multiple.txt'
    data_staqc_processing(staqc_sql_path, staqc_sql_single_save, staqc_sql_multiple_save)  # 处理staqc SQL数据

    # 定义大数据集的路径和保存路径
    large_python_path = './ulabel_data/python_codedb_qid2index_blocks_unlabeled.pickle'
    large_python_single_save = './ulabel_data/large_corpus/single/python_large_single.pickle'
    large_python_multiple_save = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    data_large_processing(large_python_path, large_python_single_save, large_python_multiple_save)  # 处理大数据集Python数据

    large_sql_path = './ulabel_data/sql_codedb_qid2index_blocks_unlabeled.pickle'
    large_sql_single_save = './ulabel_data/large_corpus/single/sql_large_single.pickle'
    large_sql_multiple_save = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    data_large_processing(large_sql_path, large_sql_single_save, large_sql_multiple_save)  # 处理大数据集SQL数据

    # 定义单个未标记数据转换为带标签数据的保存路径
    large_sql_single_label_save = './ulabel_data/large_corpus/single/sql_large_single_label.txt'
    large_python_single_label_save = './ulabel_data/large_corpus/single/python_large_single_label.txt'
    single_unlabeled_to_labeled(large_sql_single_save, large_sql_single_label_save)  # 将SQL单个未标记数据转换为带标签数据
    single_unlabeled_to_labeled(large_python_single_save, large_python_single_label_save)  # 将Python单个未标记数据转换为带标签数据
