from Bio import pairwise2

GAP = '<->'
CHANGE_START = '<%>'
CHANGE_END = '</%>'


def align(before, after, gap=GAP):
    if len(before) == 0:
        return [gap] * len(after), after
    if len(after) == 0:
        return before, [gap] * len(before)
    alignments = pairwise2.align.globalxs(before, after, -.5, -.1, gap_char=[gap], penalize_end_gaps=False, one_alignment_only=True)
    shortest = alignments[0]
    # for al in alignments:
    #     if len(al[0]) < len(shortest[0]):
    #         shortest = al
    return shortest[0], shortest[1]


def add_diff_symbol(before, after):
    symbols = list()
    for i in range(len(before)):
        if before[i] == after[i]:
            symbols.append("=")
        elif before[i] == GAP:
            symbols.append("+")
        elif after[i] == GAP:
            symbols.append("-")
        else:
            symbols.append("*")
    return symbols


def fix_stars(symbols):
    def aux(generator):
        found = False
        for i in generator:
            if symbols[i] == '*':
                found = True
            elif symbols[i] == '=':
                found = False
            elif found is True:
                symbols[i] = '*'

    aux(range(len(symbols)))
    aux(reversed(range(len(symbols))))
    return symbols


def group_diff(after, before, symbols):
    combined = list()
    symbols = fix_stars(symbols)
    after_word = list()
    before_word = list()
    res = list()
    current_s = symbols[0]
    for i, s in enumerate(symbols):
        if s == current_s:
            after_word.append(after[i])
            before_word.append(before[i])
        else:
            if current_s == '=':
                combined.append(" ".join(before_word))
            else:
                if current_s == '+':
                    res_str = "<+> {}".format(" ".join(after_word))
                elif current_s == '-':
                    res_str = "<-> {}".format(" ".join(before_word))
                elif current_s == '*':
                    res_str = "<*> {} -> {}".format(" ".join(filter(lambda x: x != GAP, before_word)), " ".join(filter(lambda x: x != GAP, after_word)))
                res.append(res_str)
                combined += [CHANGE_START] + [res_str] + [CHANGE_END]
            current_s = s
            after_word = [after[i]]
            before_word = [before[i]]
    if current_s == '=':
        combined.append(" ".join(before_word))
    else:
        if current_s == '+':
            res_str = "<+> {}".format(" ".join(after_word))
        elif current_s == '-':
            res_str = "<-> {}".format(" ".join(before_word))
        elif current_s == '*':
            res_str = "<*> {} -> {}".format(" ".join(filter(lambda x: x != GAP, before_word)), " ".join(filter(lambda x: x != GAP, after_word)))
        res.append(res_str)
        combined += [CHANGE_START] + [res_str] + [CHANGE_END]
    return res, combined


def create_diff_strs(before_str, after_str):
    before = before_str.strip().split()
    after = after_str.strip().split()
    before, after = align(before, after)
    if len(before) == 0:
        return "", ""
    symbols = add_diff_symbol(before, after)
    res, combined = group_diff(after, before, symbols)
    res = " | ".join(res)
    combined = " ".join(combined)
    return res, combined


def align_lines(before_lines, after_lines):
    return align(before_lines, after_lines, gap='')


def removes_only(before_str, after_str):
    before = before_str.strip().split()
    after = after_str.strip().split()
    before, after = align(before, after)
    if len(before) == 0:
        return True
    symbols = add_diff_symbol(before, after)
    if '*' in symbols or '+' in symbols:
        return False
    return True


if __name__ == '__main__':
    before = 'return task . from _ result ( new dictionary < string , string > ( ) ) ;'
    after = 'return new dictionary < string , string > ( ) ;'
    # print(align_lines(before.strip().split(), after.strip().split()))
    print(removes_only(before, after))
