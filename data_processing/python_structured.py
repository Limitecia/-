# -*- coding: utf-8 -*-
import re
import ast
import sys
import token
import tokenize
from nltk import wordpunct_tokenize
from io import StringIO
import inflection
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

wnler = WordNetLemmatizer()

# 正则表达式匹配变量赋值语句
PATTERN_VAR_EQUAL = re.compile(r"(\s*[_a-zA-Z][_a-zA-Z0-9]*\s*)(,\s*[_a-zA-Z][_a-zA-Z0-9]*\s*)*=")
# 正则表达式匹配for循环中的变量
PATTERN_VAR_FOR = re.compile(r"for\s+[_a-zA-Z][_a-zA-Z0-9]*\s*(,\s*[_a-zA-Z][_a-zA-Z0-9]*)*\s+in")

def repair_program_io(code):
    """
    修复IPython和Python交互会话中的输入输出代码块

    参数:
        code (str): 原始代码字符串

    返回:
        repaired_code (str): 修复后的代码字符串
        code_list (list): 修复后代码的列表形式
    """
    # 匹配IPython输入输出模式的正则表达式
    pattern_case1_in = re.compile(r"In ?\[\d+]: ?")
    pattern_case1_out = re.compile(r"Out ?\[\d+]: ?")
    pattern_case1_cont = re.compile(r"( )+\.+: ?")

    # 匹配Python交互模式的正则表达式
    pattern_case2_in = re.compile(r">>> ?")
    pattern_case2_cont = re.compile(r"\.\.\. ?")

    # 所有模式的集合
    patterns = [pattern_case1_in, pattern_case1_out, pattern_case1_cont, pattern_case2_in, pattern_case2_cont]

    # 将代码按行分割
    lines = code.split("\n")
    lines_flags = [0 for _ in range(len(lines))]
    code_list = []

    # 标记每行代码匹配的模式
    for line_idx in range(len(lines)):
        line = lines[line_idx]
        for pattern_idx in range(len(patterns)):
            if re.match(patterns[pattern_idx], line):
                lines_flags[line_idx] = pattern_idx + 1
                break
    lines_flags_string = "".join(map(str, lines_flags))

    bool_repaired = False

    # 如果所有行都不需要修复
    if lines_flags.count(0) == len(lines_flags):
        repaired_code = code
        code_list = [code]
        bool_repaired = True
    # 修复代码块
    elif re.match(re.compile(r"(0*1+3*2*0*)+"), lines_flags_string) or re.match(re.compile(r"(0*4+5*0*)+"), lines_flags_string):
        repaired_code = ""
        pre_idx = 0
        sub_block = ""
        if lines_flags[0] == 0:
            flag = 0
            while flag == 0:
                repaired_code += lines[pre_idx] + "\n"
                pre_idx += 1
                flag = lines_flags[pre_idx]
            sub_block = repaired_code
            code_list.append(sub_block.strip())
            sub_block = ""

        for idx in range(pre_idx, len(lines_flags)):
            if lines_flags[idx] != 0:
                repaired_code += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"
                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] == 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"
            else:
                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] != 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += lines[idx] + "\n"

        if len(sub_block.strip()):
            code_list.append(sub_block.strip())

        if len(repaired_code.strip()) != 0:
            bool_repaired = True

    # 如果代码仍未修复，则尝试删除0标记的行
    if not bool_repaired:
        repaired_code = ""
        sub_block = ""
        bool_after_Out = False
        for idx in range(len(lines_flags)):
            if lines_flags[idx] != 0:
                if lines_flags[idx] == 2:
                    bool_after_Out = True
                else:
                    bool_after_Out = False
                repaired_code += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"
                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] == 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"
            else:
                if not bool_after_Out:
                    repaired_code += lines[idx] + "\n"
                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] != 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += lines[idx] + "\n"

    return repaired_code, code_list

def get_vars(ast_root):
    """
    获取抽象语法树中的变量名称

    参数:
        ast_root (AST): 抽象语法树的根节点

    返回:
        var_names (list): 变量名称的列表
    """
    return sorted(
        {node.id for node in ast.walk(ast_root) if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)}
    )

def get_vars_heuristics(code):
    """
    使用启发式方法从代码中提取变量名

    参数:
        code (str): 原始代码字符串

    返回:
        varnames (set): 变量名称的集合
    """
    varnames = set()
    code_lines = [line for line in code.split("\n") if len(line.strip())]

    start = 0
    end = len(code_lines) - 1
    bool_success = False

    while not bool_success:
        try:
            root = ast.parse("\n".join(code_lines[start:end]))
        except:
            end -= 1
        else:
            bool_success = True

    varnames = varnames.union(set(get_vars(root)))

    for line in code_lines[end:]:
        line = line.strip()
        try:
            root = ast.parse(line)
        except:
            pattern_var_equal_matched = re.match(PATTERN_VAR_EQUAL, line)
            if pattern_var_equal_matched:
                match = pattern_var_equal_matched.group()[:-1]
                varnames = varnames.union(set([var.strip() for var in match.split(",")]))

            pattern_var_for_matched = re.search(PATTERN_VAR_FOR, line)
            if pattern_var_for_matched:
                match = pattern_var_for_matched.group()[3:-2]
                varnames = varnames.union(set([var.strip() for var in match.split(",")]))
        else:
            varnames = varnames.union(get_vars(root))

    return varnames

def PythonParser(code):
    """
    解析Python代码，提取变量和标记化代码

    参数:
        code (str): 原始Python代码字符串

    返回:
        tokenized_code (list): 标记化后的代码列表
        bool_failed_var (bool): 是否变量提取失败
        bool_failed_token (bool): 是否标记化失败
    """
    bool_failed_var = False
    bool_failed_token = False

    try:
        root = ast.parse(code)
        varnames = set(get_vars(root))
    except:
        repaired_code, _ = repair_program_io(code)
        try:
            root = ast.parse(repaired_code)
            varnames = set(get_vars(root))
        except:
            bool_failed_var = True
            varnames = get_vars_heuristics(code)

    tokenized_code = []

    def first_trial(_code):
        """
        检查代码是否能被tokenize成功

        参数:
            _code (str): 待检查的代码

        返回:
            (bool): 是否成功
        """
        if len(_code) == 0:
            return True
        try:
            g = tokenize.generate_tokens(StringIO(_code).readline)
            term = next(g)
        except:
            return False
        else:
            return True

    bool_first_success = first_trial(code)
    while not bool_first_success:
        code = code[1:]
        bool_first_success = first_trial(code)
    g = tokenize.generate_tokens(StringIO(code).readline)
    term = next(g)

    bool_finished = False
    while not bool_finished:
        term_type = term[0]
        lineno = term[2][0] - 1
        posno = term[3][1] - 1
        if token.tok_name[term_type] in {"NUMBER", "STRING", "NEWLINE"}:
            tokenized_code.append(token.tok_name[term_type])
        elif not token.tok_name[term_type] in {"COMMENT", "ENDMARKER"} and len(term[1].strip()):
            candidate = term[1].strip()
            if candidate not in varnames:
                tokenized_code.append(candidate)
            else:
                tokenized_code.append("VAR")

        # 获取下一个标记
        bool_success_next = False
        while not bool_success_next:
            try:
                term = next(g)
            except StopIteration:
                bool_finished = True
                break
            except:
                bool_failed_token = True
                code_lines = code.split("\n")
                if lineno > len(code_lines) - 1:
                    print(sys.exc_info())
                else:
                    failed_code_line = code_lines[lineno]
                    if posno < len(failed_code_line) - 1:
                        failed_code_line = failed_code_line[posno:]
                        tokenized_failed_code_line = wordpunct_tokenize(failed_code_line)
                        tokenized_code += tokenized_failed_code_line
                    if lineno < len(code_lines) - 1:
                        code = "\n".join(code_lines[lineno + 1:])
                        g = tokenize.generate_tokens(StringIO(code).readline)
                    else:
                        bool_finished = True
                        break
            else:
                bool_success_next = True

    return tokenized_code, bool_failed_var, bool_failed_token



#############################################################################
# 缩略词处理

def revert_abbrev(line):
    """
    将常见的英文缩略词恢复为完整形式。

    参数：
        line (str): 输入的字符串，可能包含缩略词。

    返回：
        str: 将缩略词恢复为完整形式的字符串。
    """
    # 定义正则表达式模式，用于匹配缩略词
    pat_is = re.compile("(it|he|she|that|this|there|here)(\"s)", re.I)
    pat_s1 = re.compile("(?<=[a-zA-Z])\"s")
    pat_s2 = re.compile("(?<=s)\"s?")
    pat_not = re.compile("(?<=[a-zA-Z])n\"t")
    pat_would = re.compile("(?<=[a-zA-Z])\"d")
    pat_will = re.compile("(?<=[a-zA-Z])\"ll")
    pat_am = re.compile("(?<=[I|i])\"m")
    pat_are = re.compile("(?<=[a-zA-Z])\"re")
    pat_ve = re.compile("(?<=[a-zA-Z])\"ve")

    # 替换缩略词为完整形式
    line = pat_is.sub(r"\1 is", line)
    line = pat_s1.sub("", line)
    line = pat_s2.sub("", line)
    line = pat_not.sub(" not", line)
    line = pat_would.sub(" would", line)
    line = pat_will.sub(" will", line)
    line = pat_am.sub(" am", line)
    line = pat_are.sub(" are", line)
    line = pat_ve.sub(" have", line)

    return line


# 获取词性
def get_wordpos(tag):
    """
    根据词性标签返回对应的WordNet词性。

    参数：
        tag (str): 词性标签。

    返回：
        str: WordNet词性标签，可能为形容词、动词、名词或副词。
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# ---------------------子函数1：句子的去冗--------------------
def process_nl_line(line):
    """
    预处理句子，去除冗余信息，标准化格式。

    参数：
        line (str): 输入的句子。

    返回：
        str: 经过预处理的句子。
    """
    line = revert_abbrev(line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = line.replace('\n', ' ')
    line = re.sub(' +', ' ', line)
    line = line.strip()
    line = inflection.underscore(line)

    # 去除括号里的内容
    space = re.compile(r"\([^(|^)]+\)")
    line = re.sub(space, '', line)
    line = line.strip()
    return line


# ---------------------子函数1：句子的分词--------------------
def process_sent_word(line):
    """
    对句子进行分词、词性标注和词形还原。

    参数：
        line (str): 输入的句子。

    返回：
        list: 经过处理的词列表。
    """
    # 找单词
    line = re.findall(r"\w+|[^\s\w]", line)
    line = ' '.join(line)
    
    # 替换不同的内容为标记
    decimal = re.compile(r"\d+(\.\d+)+")
    line = re.sub(decimal, 'TAGINT', line)
    string = re.compile(r'\"[^\"]+\"')
    line = re.sub(string, 'TAGSTR', line)
    decimal = re.compile(r"0[xX][A-Fa-f0-9]+")
    line = re.sub(decimal, 'TAGINT', line)
    number = re.compile(r"\s?\d+\s?")
    line = re.sub(number, ' TAGINT ', line)
    other = re.compile(r"(?<![A-Z|a-z_])\d+[A-Za-z]+")
    line = re.sub(other, 'TAGOER', line)
    
    cut_words = line.split(' ')
    cut_words = [x.lower() for x in cut_words]
    word_tags = pos_tag(cut_words)
    tags_dict = dict(word_tags)
    word_list = []
    
    for word in cut_words:
        word_pos = get_wordpos(tags_dict[word])
        if word_pos in ['a', 'v', 'n', 'r']:
            word = wnler.lemmatize(word, pos=word_pos)
        word = wordnet.morphy(word) if wordnet.morphy(word) else word
        word_list.append(word)
    
    return word_list


#############################################################################

def filter_all_invachar(line):
    """
    过滤掉所有非常用符号，仅保留字母、数字、常用标点符号。

    参数：
        line (str): 输入的字符串。

    返回：
        str: 过滤后的字符串。
    """
    assert isinstance(line, object)
    line = re.sub('[^(0-9|a-zA-Z\-_\'\")\n]+', ' ', line)
    line = re.sub('-+', '-', line)
    line = re.sub('_+', '_', line)
    line = line.replace('|', ' ').replace('¦', ' ')
    return line


def filter_part_invachar(line):
    """
    过滤掉部分非常用符号，仅保留字母、数字、常用标点符号。

    参数：
        line (str): 输入的字符串。

    返回：
        str: 过滤后的字符串。
    """
    line = re.sub('[^(0-9|a-zA-Z\-_\'\")\n]+', ' ', line)
    line = re.sub('-+', '-', line)
    line = re.sub('_+', '_', line)
    line = line.replace('|', ' ').replace('¦', ' ')
    return line


########################主函数：代码的tokens#################################
def python_code_parse(line):
    """
    将Python代码解析为token列表。

    参数：
        line (str): 输入的Python代码。

    返回：
        list: 解析后的token列表。如果解析失败，返回'-1000'。
    """
    line = filter_part_invachar(line)
    line = re.sub('\.+', '.', line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = re.sub('>>+', '', line)
    line = re.sub(' +', ' ', line)
    line = line.strip('\n').strip()
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)

    try:
        typedCode, failed_var, failed_token = PythonParser(line)
        typedCode = inflection.underscore(' '.join(typedCode)).split(' ')
        cut_tokens = [re.sub("\s+", " ", x.strip()) for x in typedCode]
        token_list = [x.lower() for x in cut_tokens]
        token_list = [x.strip() for x in token_list if x.strip() != '']
        return token_list
    except:
        return '-1000'


########################主函数：代码的tokens#################################


#######################主函数：句子的tokens##################################
def python_query_parse(line):
    """
    将自然语言查询解析为token列表。

    参数：
        line (str): 输入的查询语句。

    返回：
        list: 解析后的token列表。
    """
    line = filter_all_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)
    
    for i in range(0, len(word_list)):
        if re.findall('[()]', word_list[i]):
            word_list[i] = ''
    word_list = [x.strip() for x in word_list if x.strip() != '']
    return word_list


def python_context_parse(line):
    """
    将自然语言上下文解析为token列表。

    参数：
        line (str): 输入的上下文语句。

    返回：
        list: 解析后的token列表。
    """
    line = filter_part_invachar(line)
    line = process_nl_line(line)
    print(line)
    word_list = process_sent_word(line)
    word_list = [x.strip() for x in word_list if x.strip() != '']
    return word_list


#######################主函数：句子的tokens##################################

if __name__ == '__main__':
    """
    主函数入口，测试各个解析函数的功能。
    """
    # 测试 python_query_parse 函数
    # 测试用例1：调整LibreOffice Calc中行高和列宽的查询
    print(python_query_parse("change row_height and column_width in libreoffice calc use python tagint"))
    
    # 测试用例2：在Python中给datetime.time增加N秒的标准方法查询
    print(python_query_parse('What is the standard way to add N seconds to datetime.time in Python?'))
    
    # 测试用例3：将SQL中的INT转换为VARCHAR的查询
    print(python_query_parse("Convert INT to VARCHAR SQL 11?"))
    
    # 测试用例4：构建包含特定模式的字典的查询
    print(python_query_parse('python construct a dictionary {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 0, 2], 3: [0, 0, 3], ...,999: [9, 9, 9]}'))

    # 测试 python_context_parse 函数
    # 测试用例1：计算平方和直到用户指定的总和达到SQL查询
    print(python_context_parse('How to calculateAnd the value of the sum of squares defined as \n 1^2 + 2^2 + 3^2 + ... +n2 until a user specified sum has been reached sql()'))
    
    # 测试用例2：在SQL中显示包含特定信息的记录的查询
    print(python_context_parse('how do i display records (containing specific) information in sql() 11?'))
    
    # 测试用例3：将SQL中的INT转换为VARCHAR的查询
    print(python_context_parse('Convert INT to VARCHAR SQL 11?'))

    # 测试 python_code_parse 函数
    # 测试用例1：C#代码片段
    print(python_code_parse('if(dr.HasRows)\n{\n // ....\n}\nelse\n{\n MessageBox.Show("ReservationAnd Number Does Not Exist","Error", MessageBoxButtons.OK, MessageBoxIcon.Asterisk);\n}'))
    
    # 测试用例2：Python代码片段，包含错误的变量名
    print(python_code_parse('root -> 0.0 \n while root_ * root < n: \n root = root + 1 \n print(root * root)'))
    
    # 测试用例3：Python代码片段，包含变量赋值和循环
    print(python_code_parse('root = 0.0 \n while root * root < n: \n print(root * root) \n root = root + 1'))
    
    # 测试用例4：Python代码片段，包含循环和条件语句
    print(python_code_parse('n = 1 \n while n <= 100: \n n = n + 1 \n if n > 10: \n  break print(n)'))
    
    # 测试用例5：包含复杂的嵌套函数定义的Python代码片段
    print(python_code_parse("diayong(2) def sina_download(url, output_dir='.', merge=True, info_only=False, **kwargs):\n    if 'news.sina.com.cn/zxt' in url:\n        sina_zxt(url, output_dir=output_dir, merge=merge, info_only=info_only, **kwargs)\n  return\n\n    vid = match1(url, r'vid=(\\d+)')\n    if vid is None:\n        video_page = get_content(url)\n        vid = hd_vid = match1(video_page, r'hd_vid\\s*:\\s*\\'([^\\']+)\\'')\n  if hd_vid == '0':\n            vids = match1(video_page, r'[^\\w]vid\\s*:\\s*\\'([^\\']+)\\'').split('|')\n            vid = vids[-1]\n\n    if vid is None:\n        vid = match1(video_page, r'vid:\"?(\\d+)\"?')\n    if vid:\n   sina_download_by_vid(vid, output_dir=output_dir, merge=merge, info_only=info_only)\n    else:\n        vkey = match1(video_page, r'vkey\\s*:\\s*\"([^\"]+)\"')\n        if vkey is None:\n            vid = match1(url, r'#(\\d+)')\n            sina_download_by_vid(vid, output_dir=output_dir, merge=merge, info_only=info_only)\n            return\n        title = match1(video_page, r'title\\s*:\\s*\"([^\"]+)\"')\n        sina_download_by_vkey(vkey, title=title, output_dir=output_dir, merge=merge, info_only=info_only)"))

    # 测试用例6：Python字典遍历代码片段
    print(python_code_parse("d = {'x': 1, 'y': 2, 'z': 3} \n for key in d: \n  print (key, 'corresponds to', d[key])"))
    
    # 测试用例7：注释行的代码片段
    print(python_code_parse('  #       page  hour  count\n # 0     3727441     1   2003\n # 1     3727441     2    654\n # 2     3727441     3   5434\n # 3     3727458     1    326\n # 4     3727458     2   2348\n # 5     3727458     3   4040\n # 6   3727458_1     4    374\n # 7   3727458_1     5   2917\n # 8   3727458_1     6   3937\n # 9     3735634     1   1957\n # 10    3735634     2   2398\n # 11    3735634     3   2812\n # 12    3768433     1    499\n # 13    3768433     2   4924\n # 14    3768433     3   5460\n # 15  3768433_1     4   1710\n # 16  3768433_1     5   3877\n # 17  3768433_1     6   1912\n # 18  3768433_2     7   1367\n # 19  3768433_2     8   1626\n # 20  3768433_2     9   4750\n'))
