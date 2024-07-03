import pickle
import multiprocessing
from python_structured import *
from sqlang_structured import *


def multipro_python_query(data_list):
    """
    多进程处理Python查询数据的函数
    :param data_list: 包含待处理数据的列表
    :return: 处理后的数据列表
    """
    return [python_query_parse(line) for line in data_list]

def multipro_python_code(data_list):
    """
    多进程处理Python代码数据的函数
    :param data_list: 包含待处理数据的列表
    :return: 处理后的数据列表
    """
    return [python_code_parse(line) for line in data_list]

def multipro_python_context(data_list):
    """
    多进程处理Python上下文数据的函数
    :param data_list: 包含待处理数据的列表
    :return: 处理后的数据列表
    """
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(python_context_parse(line))
    return result
def multipro_sqlang_query(data_list):
    """
    多进程处理SQL查询数据的函数
    :param data_list: 包含待处理数据的列表
    :return: 处理后的数据列表
    """
    return [sqlang_query_parse(line) for line in data_list]


def multipro_sqlang_code(data_list):
    """
    多进程处理SQL代码数据的函数
    :param data_list: 包含待处理数据的列表
    :return: 处理后的数据列表
    """
    return [sqlang_code_parse(line) for line in data_list]


def multipro_sqlang_context(data_list):
    """
    多进程处理SQL上下文数据的函数
    :param data_list: 包含待处理数据的列表
    :return: 处理后的数据列表
    """
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(sqlang_context_parse(line))
    return result


def parse(data_list, split_num, context_func, query_func, code_func):
    """
    解析数据的函数，使用多进程来提高效率
    :param data_list: 包含待处理数据的列表
    :param split_num: 每个子列表的长度
    :param context_func: 处理上下文数据的函数
    :param query_func: 处理查询数据的函数
    :param code_func: 处理代码数据的函数
    :return: 上下文数据列表、查询数据列表、代码数据列表
    """
    pool = multiprocessing.Pool()  # 创建进程池
    split_list = [data_list[i:i + split_num] for i in range(0, len(data_list), split_num)]  # 将数据按块分割
    results = pool.map(context_func, split_list)  # 处理上下文数据
    context_data = [item for sublist in results for item in sublist]  # 合并结果
    print(f'context条数：{len(context_data)}')

    results = pool.map(query_func, split_list)  # 处理查询数据
    query_data = [item for sublist in results for item in sublist]  # 合并结果
    print(f'query条数：{len(query_data)}')

    results = pool.map(code_func, split_list)  # 处理代码数据
    code_data = [item for sublist in results for item in sublist]  # 合并结果
    print(f'code条数：{len(code_data)}')

    pool.close()  # 关闭进程池
    pool.join()  # 等待所有进程结束

    return context_data, query_data, code_data


def main(lang_type, split_num, source_path, save_path, context_func, query_func, code_func):
    """
    主函数，处理数据并保存结果
    :param lang_type: 语言类型
    :param split_num: 每个子列表的长度
    :param source_path: 源数据文件路径
    :param save_path: 处理后数据保存路径
    :param context_func: 处理上下文数据的函数
    :param query_func: 处理查询数据的函数
    :param code_func: 处理代码数据的函数
    """
    with open(source_path, 'rb') as f:
        corpus_lis = pickle.load(f)

    context_data, query_data, code_data = parse(corpus_lis, split_num, context_func, query_func, code_func)
    qids = [item[0] for item in corpus_lis]

    total_data = [[qids[i], context_data[i], code_data[i], query_data[i]] for i in range(len(qids))]

    with open(save_path, 'wb') as f:
        pickle.dump(total_data, f)

if __name__ == '__main__':
    # STaQC Python数据路径
    staqc_python_path = '.ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.pkl'

    # STaQC SQL数据路径
    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = './ulabel_data/staqc/sql_staqc_unlabled_data.pkl'

    # 处理STaQC数据
    main(python_type, split_num, staqc_python_path, staqc_python_save, multipro_python_context, multipro_python_query, multipro_python_code) # type: ignore
    main(sqlang_type, split_num, staqc_sql_path, staqc_sql_save, multipro_sqlang_context, multipro_sqlang_query, multipro_sqlang_code) # type: ignore

    # 大规模Python数据路径
    large_python_path = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.pkl'

    # 大规模SQL数据路径
    large_sql_path = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.pkl'

    # 处理大规模数据
    main(python_type, split_num, large_python_path, large_python_save, multipro_python_context, multipro_python_query, multipro_python_code) # type: ignore
    main(sqlang_type, split_num, large_sql_path, large_sql_save, multipro_sqlang_context, multipro_sqlang_query, multipro_sqlang_code) # type: ignore
