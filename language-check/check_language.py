#!/usr/bin/env python3

import argparse
import json
import language_check

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()

tool = language_check.LanguageTool('en-US')
with open(args.input) as f:
    dataset = json.load(f)

bad_ques_cnt = 0

rules_to_ignore = ['ASK_THE_QUESTION', 'CURRENCY_SPACE', 'IN_WHO', 'BOTH_AS_WELL_AS', 'ALLOW_TO', 'A_PLURAL', 'EN_COMPOUNDS', 'WHO_NOUN', 'COMMA_MONTH_DATE', 'A_RB_NN', 'CD_NN', 'DT_DT', 'IN_A_X_MANNER', 'ABOUT_ITS_NN', 'SOME_NN_VBP', 'SENTENCE_FRAGMENT', 'SOME_OF_THE', 'THE_SUPERLATIVE', 'IS_COMPRISED_OF', 'ITS_JJ_NNSNN', 'THIS_NNS', 'ON_BEHAVE', 'IF_IS', 'ORIGINALLY_DISCOVERED', 'EN_A_VS_AN', 'APOS_ARE', 'SENT_START_CONJUNCTIVE_LINKING_ADVERB_COMMA', 'ON_GOING', 'PERIOD_OF_TIME', 'COMP_THAN', 'APART_A_PART', 'ONE_ORE', 'LARGE_NUMBER_OF', 'SENTENCE_WHITESPACE', 'ENGLISH_WORD_REPEAT_RULE', 'ALL_OF_THE', 'COMMA_PARENTHESIS_WHITESPACE', 'HOLLOW_TUBE', 'CLOSE_SCRUTINY', 'EN_QUOTES', 'RATHER_THEN', 'MASS_AGREEMENT', 'WHOS_NN', 'PHRASE_REPETITION', 'MANY_NN', 'AFFORD_VB', 'NEEDNT_TO_DO_AND_DONT_NEED_DO', 'FROM_FORM', 'NOW', 'PROGRESSIVE_VERBS', 'OUT_OF_PLACE', 'BEEN_PART_AGREEMENT', 'THEIR_IS', 'EN_UNPAIRED_BRACKETS', 'UPPERCASE_SENTENCE_START', 'MUCH_COUNTABLE', 'DOES_NP_VBZ', 'MORFOLOGIK_RULE_EN_US', 'USE_TO_VERB', 'HE_VERB_AGR', 'ADOPT_TO', 'HAVE_PART_AGREEMENT', 'WHOS', 'FEWER_LESS', 'NUMEROUS_DIFFERENT', 'IN_REGARD_TO', 'IT_IS', 'ADVERB_WORD_ORDER', 'GENERAL_XX', 'FOR_ITS_NN', 'MOST_SOME_OF_NNS', 'APOSTROPHE_PLURAL', 'POSSESSIVE_APOSTROPHE', 'WHETHER', 'FROM_WHENCE', 'WHERE_AS', 'WHITESPACE_RULE', 'DID_PAST', 'DID_BASEFORM', 'AFFECT_EFFECT', 'A_INFINITVE']
with open(args.output, 'w') as f:
    articles = []
    for article in dataset['data']:
        for paragraph in article['paragraphs']:
            filtered_qas = []
            for qa in paragraph['qas']:
                question = qa['question']
                matches = tool.check(question)
                error_cnt = 0
                for match in matches:
                    if match.ruleId not in rules_to_ignore:
                        print(match)
                        error_cnt = error_cnt + 1
                if error_cnt == 0:
                    articles.append(article)
                else:
                    print(question)
                    print(paragraph)
                    bad_ques_cnt = bad_ques_cnt + 1
    

    dataset['data'] = articles
    f.write(json.dumps(dataset))
    print(bad_ques_cnt)






