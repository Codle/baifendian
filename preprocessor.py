""" 数据预处理
"""

import argparse
import os
from typing import List
import pandas as pd
from lxml.etree import parse as xml_parse

parse = argparse.ArgumentParser()
parse.add_argument('--data_path', default='./data', help='data directionary')
args = parse.parse_args()


def get_two_pair(l1: List, l2: List) -> List:
    """ 获取成对数据
    """
    pair = []
    for q1 in l1:
        for q2 in l2:
            pair.append((q1, q2, 0))
    for q1 in l1:
        for q2 in l1:
            pair.append((q1, q2, 1))

    return pair


def main() -> None:
    doc = xml_parse(os.path.join(args.data_path, 'train_set.xml'))
    questions = doc.findall('Questions')
    res = []
    for question in questions:
        equ_questions = question.find('EquivalenceQuestions')
        neq_questions = question.find('NotEquivalenceQuestions')

        equ_list = [q.text for q in equ_questions.findall('question')]
        neq_list = [q.text for q in neq_questions.findall('question')]
        res += get_two_pair(equ_list, neq_list)
    new_df = pd.DataFrame(res)
    new_df.columns = ['question1', 'question2', 'label']
    new_df.to_csv(os.path.join(args.data_path, 'train_set.csv'),
                  index=False, sep='\t')


if __name__ == "__main__":
    main()
