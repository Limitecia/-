# -*- coding: utf-8 -*-
import re
import sqlparse  # 0.4.2版本的sqlparse库，用于解析和格式化SQL语句

# 骆驼命名法
import inflection

# 词性还原
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
wnler = WordNetLemmatizer()

# 词干提取
from nltk.corpus import wordnet

#############################################################################
# 定义不同的标记类型常量
OTHER = 0
FUNCTION = 1
BLANK = 2
KEYWORD = 3
INTERNAL = 4

TABLE = 5
COLUMN = 6
INTEGER = 7
FLOAT = 8
HEX = 9
STRING = 10
WILDCARD = 11

SUBQUERY = 12

DUD = 13

# 将标记类型常量映射到对应的字符串
ttypes = {0: "OTHER", 1: "FUNCTION", 2: "BLANK", 3: "KEYWORD", 4: "INTERNAL", 5: "TABLE", 6: "COLUMN", 7: "INTEGER",
          8: "FLOAT", 9: "HEX", 10: "STRING", 11: "WILDCARD", 12: "SUBQUERY", 13: "DUD"}

# 正则表达式扫描器，定义了各种模式及其处理方法
scanner = re.Scanner([
    (r"\[[^\]]*\]", lambda scanner, token: token),
    (r"\+", lambda scanner, token: "REGPLU"),
    (r"\*", lambda scanner, token: "REGAST"),
    (r"%", lambda scanner, token: "REGCOL"),
    (r"\^", lambda scanner, token: "REGSTA"),
    (r"\$", lambda scanner, token: "REGEND"),
    (r"\?", lambda scanner, token: "REGQUE"),
    (r"[\.~``;_a-zA-Z0-9\s=:\{\}\-\\]+", lambda scanner, token: "REFRE"),
    (r'.', lambda scanner, token: None),
])

def tokenizeRegex(s):
    """
    使用预定义的正则表达式扫描器对字符串进行标记化处理。

    参数:
    s: str，待处理的字符串。

    返回:
    results: list，标记化后的结果。
    """
    results = scanner.scan(s)[0]
    return results

class SqlangParser():
    @staticmethod
    def sanitizeSql(sql):
        """
        对SQL语句进行预处理，标准化其格式。

        参数:
        sql: str，待处理的SQL语句。

        返回:
        s: str，处理后的SQL语句。
        """
        s = sql.strip().lower()  # 去除首尾空格并转换为小写
        if not s.endswith(";"):  # 确保语句以分号结尾
            s += ';'
        s = re.sub(r'\(', r' ( ', s)  # 在括号两侧添加空格
        s = re.sub(r'\)', r' ) ', s)
        words = ['index', 'table', 'day', 'year', 'user', 'text']
        for word in words:  # 对特定关键字进行替换
            s = re.sub(r'([^\w])' + word + '$', r'\1' + word + '1', s)
            s = re.sub(r'([^\w])' + word + r'([^\w])', r'\1' + word + '1' + r'\2', s)
        s = s.replace('#', '')  # 移除井号
        return s

    def parseStrings(self, tok):
        """
        递归处理SQL标记，解析其中的字符串。

        参数:
        tok: sqlparse.sql.TokenList或sqlparse.sql.Token，待处理的SQL标记。
        """
        if isinstance(tok, sqlparse.sql.TokenList):
            for c in tok.tokens:
                self.parseStrings(c)
        elif tok.ttype == STRING:
            if self.regex:
                tok.value = ' '.join(tokenizeRegex(tok.value))
            else:
                tok.value = "CODSTR"

    def renameIdentifiers(self, tok):
        """
        重命名SQL标识符（表名和列名）。

        参数:
        tok: sqlparse.sql.TokenList或sqlparse.sql.Token，待处理的SQL标记。
        """
        if isinstance(tok, sqlparse.sql.TokenList):
            for c in tok.tokens:
                self.renameIdentifiers(c)
        elif tok.ttype == COLUMN:
            if str(tok) not in self.idMap["COLUMN"]:
                colname = "col" + str(self.idCount["COLUMN"])
                self.idMap["COLUMN"][str(tok)] = colname
                self.idMapInv[colname] = str(tok)
                self.idCount["COLUMN"] += 1
            tok.value = self.idMap["COLUMN"][str(tok)]
        elif tok.ttype == TABLE:
            if str(tok) not in self.idMap["TABLE"]:
                tabname = "tab" + str(self.idCount["TABLE"])
                self.idMap["TABLE"][str(tok)] = tabname
                self.idMapInv[tabname] = str(tok)
                self.idCount["TABLE"] += 1
            tok.value = self.idMap["TABLE"][str(tok)]

        elif tok.ttype == FLOAT:
            tok.value = "CODFLO"
        elif tok.ttype == INTEGER:
            tok.value = "CODINT"
        elif tok.ttype == HEX:
            tok.value = "CODHEX"

    def __hash__(self):
        """
        计算SQL标记列表的哈希值。

        返回:
        hash值: int，SQL标记列表的哈希值。
        """
        return hash(tuple([str(x) for x in self.tokensWithBlanks]))

    def __init__(self, sql, regex=False, rename=True):
        """
        初始化SqlangParser对象，解析并处理SQL语句。

        参数:
        sql: str，待处理的SQL语句。
        regex: bool，是否使用正则表达式解析字符串。
        rename: bool，是否重命名标识符。
        """
        self.sql = SqlangParser.sanitizeSql(sql)  # 预处理SQL语句

        self.idMap = {"COLUMN": {}, "TABLE": {}}  # 初始化标识符映射
        self.idMapInv = {}
        self.idCount = {"COLUMN": 0, "TABLE": 0}
        self.regex = regex  # 设置是否使用正则表达式解析字符串

        self.parseTreeSentinel = False
        self.tableStack = []

        self.parse = sqlparse.parse(self.sql)  # 解析SQL语句
        self.parse = [self.parse[0]]

        self.removeWhitespaces(self.parse[0])  # 移除空白符
        self.identifyLiterals(self.parse[0])  # 标识字面量
        self.parse[0].ptype = SUBQUERY
        self.identifySubQueries(self.parse[0])  # 标识子查询
        self.identifyFunctions(self.parse[0])  # 标识函数
        self.identifyTables(self.parse[0])  # 标识表

        self.parseStrings(self.parse[0])  # 解析字符串

        if rename:
            self.renameIdentifiers(self.parse[0])  # 重命名标识符

        self.tokens = SqlangParser.getTokens(self.parse)  # 获取标记

    @staticmethod
    def getTokens(parse):
        """
        获取SQL解析树中的所有标记。

        参数:
        parse: list，SQL解析树。

        返回:
        flatParse: list，扁平化的SQL标记列表。
        """
        flatParse = []
        for expr in parse:
            for token in expr.flatten():
                if token.ttype == STRING:
                    flatParse.extend(str(token).split(' '))
                else:
                    flatParse.append(str(token))
        return flatParse

    def removeWhitespaces(self, tok):
        """
        递归移除SQL标记中的空白符。

        参数:
        tok: sqlparse.sql.TokenList或sqlparse.sql.Token，待处理的SQL标记。
        """
        if isinstance(tok, sqlparse.sql.TokenList):
            tmpChildren = []
            for c in tok.tokens:
                if not c.is_whitespace:
                    tmpChildren.append(c)

            tok.tokens = tmpChildren
            for c in tok.tokens:
                self.removeWhitespaces(c)

    def identifySubQueries(self, tokenList):
        """
        递归标识SQL中的子查询。

        参数:
        tokenList: sqlparse.sql.TokenList，待处理的SQL标记列表。

        返回:
        isSubQuery: bool，是否为子查询。
        """
        isSubQuery = False

        for tok in tokenList.tokens:
            if isinstance(tok, sqlparse.sql.TokenList):
                subQuery = self.identifySubQueries(tok)
                if subQuery and isinstance(tok, sqlparse.sql.Parenthesis):
                    tok.ttype = SUBQUERY
            elif str(tok) == "select":
                isSubQuery = True
        return isSubQuery

    def identifyLiterals(self, tokenList):
        """
        标识SQL标记中的字面量类型。

        参数:
        tokenList: sqlparse.sql.TokenList，待处理的SQL标记列表。
        """
        blankTokens = [sqlparse.tokens.Name, sqlparse.tokens.Name.Placeholder]
        blankTokenTypes = [sqlparse.sql.Identifier]

        for tok in tokenList.tokens:
            if isinstance(tok, sqlparse.sql.TokenList):
                tok.ptype = INTERNAL
                self.identifyLiterals(tok)
            elif (tok.ttype == sqlparse.tokens.Keyword or str(tok) == "select"):
                tok.ttype = KEYWORD
            elif (tok.ttype == sqlparse.tokens.Number.Integer or tok.ttype == sqlparse.tokens.Literal.Number.Integer):
                tok.ttype = INTEGER
            elif (tok.ttype == sqlparse.tokens.Number.Hexadecimal or tok.ttype == sqlparse.tokens.Literal.Number.Hexadecimal):
                tok.ttype = HEX
            elif (tok.ttype == sqlparse.tokens.Number.Float or tok.ttype == sqlparse.tokens.Literal.Number.Float):
                tok.ttype = FLOAT
            elif (tok.ttype == sqlparse.tokens.String.Symbol or tok.ttype == sqlparse.tokens.String.Single or tok.ttype == sqlparse.tokens.Literal.String.Single or tok.ttype == sqlparse.tokens.Literal.String.Symbol):
                tok.ttype = STRING
            elif (tok.ttype == sqlparse.tokens.Wildcard):
                tok.ttype = WILDCARD
            elif (tok.ttype in blankTokens or isinstance(tok, blankTokenTypes[0])):
                tok.ttype = COLUMN

    def identifyFunctions(self, tokenList):
        """
        标识SQL标记中的函数。

        参数:
        tokenList: sqlparse.sql.TokenList，待处理的SQL标记列表。
        """
        for tok in tokenList.tokens:
            if (isinstance(tok, sqlparse.sql.Function)):
                self.parseTreeSentinel = True
            elif (isinstance(tok, sqlparse.sql.Parenthesis)):
                self.parseTreeSentinel = False
            if self.parseTreeSentinel:
                tok.ttype = FUNCTION
            if isinstance(tok, sqlparse.sql.TokenList):
                self.identifyFunctions(tok)

    def identifyTables(self, tokenList):
        """
        标识SQL标记中的表。

        参数:
        tokenList: sqlparse.sql.TokenList，待处理的SQL标记列表。
        """
        if tokenList.ptype == SUBQUERY:
            self.tableStack.append(False)

        for i in range(len(tokenList.tokens)):
            prevtok = tokenList.tokens[i - 1]
            tok = tokenList.tokens[i]

            if (str(tok) == "." and tok.ttype == sqlparse.tokens.Punctuation and prevtok.ttype == COLUMN):
                prevtok.ttype = TABLE

            elif (str(tok) == "from" and tok.ttype == sqlparse.tokens.Keyword):
                self.tableStack[-1] = True

            elif ((str(tok) == "where" or str(tok) == "on" or str(tok) == "group" or str(tok) == "order" or str(tok) == "union") and tok.ttype == sqlparse.tokens.Keyword):
                self.tableStack[-1] = False

            if isinstance(tok, sqlparse.sql.TokenList):
                self.identifyTables(tok)

            elif (tok.ttype == COLUMN):
                if self.tableStack[-1]:
                    tok.ttype = TABLE

        if tokenList.ptype == SUBQUERY:
            self.tableStack.pop()

    def __str__(self):
        """
        返回解析后的SQL标记字符串表示。
        """
        return ' '.join([str(tok) for tok in self.tokens])

    def parseSql(self):
        """
        返回解析后的SQL标记列表。

        返回:
        list，解析后的SQL标记列表。
        """
        return [str(tok) for tok in self.tokens]
#############################################################################

#############################################################################
# 缩略词处理
def revert_abbrev(line):
    """
    处理句子中的缩略词，将其还原为完整形式。

    参数:
    line: str，待处理的句子。

    返回:
    line: str，处理后的句子。
    """
    pat_is = re.compile("(it|he|she|that|this|there|here)(\"s)", re.I)
    # 's
    pat_s1 = re.compile("(?<=[a-zA-Z])\"s")
    # s
    pat_s2 = re.compile("(?<=s)\"s?")
    # not
    pat_not = re.compile("(?<=[a-zA-Z])n\"t")
    # would
    pat_would = re.compile("(?<=[a-zA-Z])\"d")
    # will
    pat_will = re.compile("(?<=[a-zA-Z])\"ll")
    # am
    pat_am = re.compile("(?<=[I|i])\"m")
    # are
    pat_are = re.compile("(?<=[a-zA-Z])\"re")
    # have
    pat_ve = re.compile("(?<=[a-zA-Z])\"ve")

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
    根据词性标注返回对应的词性。

    参数:
    tag: str，词性标注。

    返回:
    str，对应的词性。
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

# 句子的去冗
def process_nl_line(line):
    """
    对句子进行去冗处理。

    参数:
    line: str，待处理的句子。

    返回:
    line: str，处理后的句子。
    """
    # 句子预处理
    line = revert_abbrev(line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = line.replace('\n', ' ')
    line = line.replace('\t', ' ')
    line = re.sub(' +', ' ', line)
    line = line.strip()
    # 骆驼命名转下划线
    line = inflection.underscore(line)

    # 去除括号里内容
    space = re.compile(r"\([^\(|^\)]+\)")  # 后缀匹配
    line = re.sub(space, '', line)
    # 去除末尾.和空格
    line = line.strip()
    return line

# 句子的分词
def process_sent_word(line):
    """
    对句子进行分词处理，并替换特定模式。

    参数:
    line: str，待处理的句子。

    返回:
    word_list: list，处理后的词列表。
    """
    # 找单词
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)

    # 替换小数
    decimal = re.compile(r"\d+(\.\d+)+")
    line = re.sub(decimal, 'TAGINT', line)
    # 替换字符串
    string = re.compile(r'\"[^\"]+\"')
    line = re.sub(string, 'TAGSTR', line)
    # 替换十六进制
    decimal = re.compile(r"0[xX][A-Fa-f0-9]+")
    line = re.sub(decimal, 'TAGINT', line)
    # 替换数字
    number = re.compile(r"\s?\d+\s?")
    line = re.sub(number, ' TAGINT ', line)
    # 替换字符
    other = re.compile(r"(?<![A-Z|a-z|_|])\d+[A-Za-z]+")  # 后缀匹配
    line = re.sub(other, 'TAGOER', line)
    cut_words = line.split(' ')
    # 全部小写化
    cut_words = [x.lower() for x in cut_words]
    # 词性标注
    word_tags = pos_tag(cut_words)
    tags_dict = dict(word_tags)
    word_list = []
    for word in cut_words:
        word_pos = get_wordpos(tags_dict[word])
        if word_pos in ['a', 'v', 'n', 'r']:
            # 词性还原
            word = wnler.lemmatize(word, pos=word_pos)
        # 词干提取（效果最好）
        word = wordnet.morphy(word) if wordnet.morphy(word) else word
        word_list.append(word)
    return word_list



def filter_all_invachar(line):
    """
    去除字符串中的所有非常用符号，防止解析错误。

    Args:
        line (str): 需要处理的字符串。

    Returns:
        str: 处理后的字符串。
    """
    # 去除所有非常用符号，只保留数字、字母、常见符号和换行符
    line = re.sub('[^(0-9|a-z|A-Z|\-|_|\'|\"|\-|\(|\)|\n)]+', ' ', line)
    # 将连续的中横线替换为单个中横线
    line = re.sub('-+', '-', line)
    # 将连续的下划线替换为单个下划线
    line = re.sub('_+', '_', line)
    # 去除竖线和特殊符号
    line = line.replace('|', ' ').replace('¦', ' ')
    return line

def filter_part_invachar(line):
    """
    去除字符串中的部分非常用符号，防止解析错误。

    Args:
        line (str): 需要处理的字符串。

    Returns:
        str: 处理后的字符串。
    """
    # 去除大部分非常用符号，只保留常见符号、数字、字母和换行符
    line = re.sub('[^(0-9|a-z|A-Z|\-|#|/|_|,|\'|=|>|<|\"|\-|\\|\(|\)|\?|\.|\*|\+|\[|\]|\^|\{|\}|\n)]+', ' ', line)
    # 将连续的中横线替换为单个中横线
    line = re.sub('-+', '-', line)
    # 将连续的下划线替换为单个下划线
    line = re.sub('_+', '_', line)
    # 去除竖线和特殊符号
    line = line.replace('|', ' ').replace('¦', ' ')
    return line

def sqlang_code_parse(line):
    """
    解析SQL代码，将其转换为token列表。

    Args:
        line (str): 需要解析的SQL代码。

    Returns:
        list: 解析后的token列表。
    """
    # 过滤非常用符号
    line = filter_part_invachar(line)
    # 将连续的点替换为单个点
    line = re.sub('\.+', '.', line)
    # 将连续的制表符替换为单个制表符
    line = re.sub('\t+', '\t', line)
    # 将连续的换行符替换为单个换行符
    line = re.sub('\n+', '\n', line)
    # 将连续的空格替换为单个空格
    line = re.sub(' +', ' ', line)
    # 去除多余符号
    line = re.sub('>>+', '', line)
    # 替换小数
    line = re.sub(r"\d+(\.\d+)+", 'number', line)
    # 去除首尾的换行符和空格
    line = line.strip('\n').strip()
    # 使用正则表达式匹配所有的单词和符号
    line = re.findall(r"[\w]+|[^\s\w]", line)
    # 将列表转换为字符串
    line = ' '.join(line)

    try:
        # 解析SQL代码
        query = SqlangParser(line, regex=True)
        typedCode = query.parseSql()
        typedCode = typedCode[:-1]
        # 将驼峰命名法转换为下划线命名法
        typedCode = inflection.underscore(' '.join(typedCode)).split(' ')
        # 去除多余的空格
        cut_tokens = [re.sub("\s+", " ", x.strip()) for x in typedCode]
        # 将所有token转换为小写
        token_list = [x.lower() for x in cut_tokens]
        # 去除列表中的空字符串
        token_list = [x.strip() for x in token_list if x.strip() != '']
        # 返回token列表
        return token_list
    except:
        # 如果解析失败，返回错误码
        return '-1000'

def sqlang_query_parse(line):
    """
    解析SQL查询，将其转换为token列表。

    Args:
        line (str): 需要解析的SQL查询。

    Returns:
        list: 解析后的token列表。
    """
    # 过滤所有非常用符号
    line = filter_all_invachar(line)
    # 处理自然语言行
    line = process_nl_line(line)
    # 处理句子中的单词
    word_list = process_sent_word(line)
    # 去除单词中的括号
    for i in range(0, len(word_list)):
        if re.findall('[\(\)]', word_list[i]):
            word_list[i] = ''
    # 去除列表中的空字符串
    word_list = [x.strip() for x in word_list if x.strip() != '']
    return word_list

def sqlang_context_parse(line):
    """
    解析SQL上下文，将其转换为token列表。

    Args:
        line (str): 需要解析的SQL上下文。

    Returns:
        list: 解析后的token列表。
    """
    # 过滤部分非常用符号
    line = filter_part_invachar(line)
    # 处理自然语言行
    line = process_nl_line(line)
    # 处理句子中的单词
    word_list = process_sent_word(line)
    # 去除列表中的空字符串
    word_list = [x.strip() for x in word_list if x.strip() != '']
    return word_list

if __name__ == '__main__':
    # 测试sqlang_code_parse函数
    print(sqlang_code_parse('""geometry": {"type": "Polygon" , 111.676,"coordinates": [[[6.69245274714546, 51.1326962505233], [6.69242714158622, 51.1326908883821], [6.69242919794447, 51.1326955158344], [6.69244041615532, 51.1326998744549], [6.69244125953742, 51.1327001609189], [6.69245274714546, 51.1326962505233]]]} How to 123 create a (SQL  Server function) to "join" multiple rows from a subquery into a single delimited field?'))
    # 测试sqlang_query_parse函数
    print(sqlang_query_parse("change row_height and column_width in libreoffice calc use python tagint"))
    print(sqlang_query_parse('MySQL Administrator Backups: "Compatibility Mode", What Exactly is this doing?'))
    # 测试sqlang_code_parse函数
    print(sqlang_code_parse('>UPDATE Table1 \n SET Table1.col1 = Table2.col1 \n Table1.col2 = Table2.col2 FROM \n Table2 WHERE \n Table1.id =  Table2.id'))
    print(sqlang_code_parse("SELECT\n@supplyFee:= 0\n@demandFee := 0\n@charedFee := 0\n"))
    print(sqlang_code_parse('@prev_sn := SerialNumber,\n@prev_toner := Remain_Toner_Black\n'))
    print(sqlang_code_parse(' ;WITH QtyCTE AS (\n  SELECT  [Category] = c.category_name\n          , [RootID] = c.category_id\n          , [ChildID] = c.category_id\n  FROM    Categories c\n  UNION ALL \n  SELECT  cte.Category\n          , cte.RootID\n          , c.category_id\n  FROM    QtyCTE cte\n          INNER JOIN Categories c ON c.father_id = cte.ChildID\n)\nSELECT  cte.RootID\n        , cte.Category\n        , COUNT(s.sales_id)\nFROM    QtyCTE cte\n        INNER JOIN Sales s ON s.category_id = cte.ChildID\nGROUP BY cte.RootID, cte.Category\nORDER BY cte.RootID\n'))
    print(sqlang_code_parse("DECLARE @Table TABLE (ID INT, Code NVARCHAR(50), RequiredID INT);\n\nINSERT INTO @Table (ID, Code, RequiredID)   VALUES\n    (1, 'Physics', NULL),\n    (2, 'Advanced Physics', 1),\n    (3, 'Nuke', 2),\n    (4, 'Health', NULL);    \n\nDECLARE @DefaultSeed TABLE (ID INT, Code NVARCHAR(50), RequiredID INT);\n\nWITH hierarchy \nAS (\n    --anchor\n    SELECT  t.ID , t.Code , t.RequiredID\n    FROM @Table AS t\n    WHERE t.RequiredID IS NULL\n\n    UNION ALL   \n\n    --recursive\n    SELECT  t.ID \n          , t.Code \n          , h.ID        \n    FROM hierarchy AS h\n        JOIN @Table AS t \n            ON t.RequiredID = h.ID\n    )\n\nINSERT INTO @DefaultSeed (ID, Code, RequiredID)\nSELECT  ID \n        , Code \n        , RequiredID\nFROM hierarchy\nOPTION (MAXRECURSION 10)\n\n\nDECLARE @NewSeed TABLE (ID INT IDENTITY(10, 1), Code NVARCHAR(50), RequiredID INT)\n\nDeclare @MapIds Table (aOldID int,aNewID int)\n\n;MERGE INTO @NewSeed AS TargetTable\nUsing @DefaultSeed as Source on 1=0\nWHEN NOT MATCHED then\n Insert (Code,RequiredID)\n Values\n (Source.Code,Source.RequiredID)\nOUTPUT Source.ID ,inserted.ID into @MapIds;\n\n\nUpdate @NewSeed Set RequiredID=aNewID\nfrom @MapIds\nWhere RequiredID=aOldID\n\n\n/*\n--@NewSeed should read like the following...\n[ID]  [Code]           [RequiredID]\n10....Physics..........NULL\n11....Health...........NULL\n12....AdvancedPhysics..10\n13....Nuke.............12\n*/\n\nSELECT *\nFROM @NewSeed\n"))



