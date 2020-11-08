from pygments.lexers.dotnet import CSharpLexer
import re
import itertools

# Consider using ANTLR for better parsing
lexer = CSharpLexer()


def split_identifier(identifier):
    pattern = re.compile(r'[a-zA-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))')
    res = [pattern.findall(subotken) for subotken in identifier.split("_")]
    res = list(itertools.chain.from_iterable(res))
    res = list(map(lambda x: x.lower(), res))
    res = " _ ".join(res).split()
    return res


def group_tokens(tokens, token_str):
    tokens = list(filter(lambda t: str(t[0]) != "Token.Text", tokens))
    tokens_new = list()
    flag = False
    agg_str = ""
    token_obj = None
    for t, s in tokens:
        str_t = str(t)
        if str_t == token_str:
            if flag is False:
                flag = True
                token_obj = t
            agg_str += s
            continue
        elif flag is True:
            flag = False
            tokens_new.append((token_obj, agg_str))
            agg_str = ""
        tokens_new.append((t, s))
    if flag is True:
        tokens_new.append((token_obj, agg_str))
    return tokens_new


def tokenize_code(code, group_punc=False):
    # print("Code: {}".format(code))
    # print()
    tokens = list(lexer.get_tokens(code))
    if group_punc is True:
        tokens = group_tokens(tokens, "Token.Punctuation")
    tokens = group_tokens(tokens, "Token.Literal.Number")
    if len(tokens) == 0:
        return "", "", None
    #     if len(tokens) == 0: # or (len(tokens) == 1 and str(tokens[0][0]) == "Token.Name.Attribute"):
    #         return None, None, None
    tokens_, seperated = zip(*tokens)
    tokens = list(filter(lambda x: (not str(x[0]).startswith("Token.Text")) and x[1] != "", tokens))
    seperated = " ".join(seperated)
    tokens_ = " ".join(map(lambda x: str(x), tokens_))
    normalized_line = list()
    for i in range(len(tokens)):
        token = tokens[i][0]
        lexema = tokens[i][1]
        if token[0] == 'Comment':
            break
        if token[0] == 'Punctuation' and i + 1 < len(tokens) and tokens[i + 1][0][
            0] == 'Punctuation' and lexema == '/' and tokens[i + 1][1] == '*':
            break
        if token[0] == 'Name':
            normalized_line += split_identifier(lexema)
        elif len(token) == 2 and token[0] == 'Literal' and token[1] == 'String':
            normalized_line.append('STR')
        elif len(token) == 2 and token[0] == 'Literal' and token[1] == 'Number':
            if int(lexema) in [0,1,2]:
                normalized_line.append(lexema)
            else:
                normalized_line.append('NUM')
        else:
            normalized_line.append(lexema)

    normalized_line = " ".join(" ".join(normalized_line).strip().split())
    return normalized_line, seperated, tokens_


def tokenize_code_for_dataset(code, group_punc=False):
    tokens = list(lexer.get_tokens(code))
    if group_punc is True:
        tokens = group_tokens(tokens, "Token.Punctuation")
    tokens = group_tokens(tokens, "Token.Literal.Number")
    tokens = list(filter(lambda x: (not str(x[0]).startswith("Token.Comment")) and (not str(x[0]).startswith("Token.Text")) and x[1] != "", tokens))
    if len(tokens) == 0:
        return [], []
    tokens_, seperated = zip(*tokens)
    tokens_ = list(map(lambda x: str(x), tokens_))
    if len(seperated) != len(tokens_):
        print(code)
        print(seperated)
        print(tokens_)
    return seperated, tokens_


if __name__ == '__main__':
    code = 'using System.Linq; public class QueryExpressionTest { public static void Main() { var expr1 = new int[] { 1, 2, 3, 4, 5 }; var query2 = from int namespace in expr1 select namespace; var query25 = from i in expr1 let namespace = expr1 select i; } }"; var tree = compilation.SyntaxTrees[0]; var semanticModel = compilation.GetSemanticModel(tree); var queryExpr = tree.GetCompilationUnitRoot().DescendantNodes().OfType<QueryExpressionSyntax>().Where(x => x.ToFullString() == "from i in expr1 let ").Single(); var symbolInfo = semanticModel.GetSemanticInfoSummary(queryExpr); Assert.Null(symbolInfo.Symbol); } [WorkItem(542496, "DevDiv")] [Fact] public void QueryExpressionInFieldInitReferencingAnotherFieldWithInteractiveParseOption() {'
    print(tokenize_code("QueryExpressionTest > 5"))