import numpy as np
import json
import argparse
import os
import random
import re
import string
import sys
from nltk.tree import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir')
    ARGS = parser.parse_args()
    return ARGS

# auxiliary verbs, from https://en.wikipedia.org/wiki/Auxiliary_verb
AUX_V = ['am', 'is', 'are', 'can', 'could', 'dare', 'do', 'does', 'did', 'have', 'had', 'may', 'might', 'must', 'need', 'shall', 'should', 'will', 'would']
AUX_V_REGEX = '('+'|'.join(['('+AUX_V[i]+')' for i in range(len(AUX_V))])+')'
AUX_V_BE = ['am', 'is', 'are']
AUX_V_BE_REGEX = '('+'|'.join(['('+AUX_V_BE[i]+')' for i in range(len(AUX_V_BE))])+')'
AUX_V_DOES = ['can', 'could', 'dare', 'does', 'did', 'have', 'had', 'may', 'might', 'must', 'need', 'shall', 'should', 'will', 'would']
AUX_V_DOES_REGEX = '('+'|'.join(['('+AUX_V_DOES[i]+')' for i in range(len(AUX_V_DOES))])+')'
AUX_V_DOESONLY = ['does', 'did']
AUX_V_DOESONLY_REGEX = '('+'|'.join(['('+AUX_V_DOESONLY[i]+')' for i in range(len(AUX_V_DOESONLY))])+')'
AUX_V_DO_REGEX = '(do) '


# global variables
QUESTION_TYPES = ['__+', \
'(when '+AUX_V_REGEX+'.*)|(when\?)', \
'(where '+AUX_V_REGEX+'.*)|(where\?)', \
'what ', \
'which ', \
'(whom '+AUX_V_REGEX+'.*)|(who '+AUX_V_REGEX+'.*)|(who\?)|(whom\?)', \
'why ', \
'(how many)|(how much)', \
'(\Ahow [^(many)(much)])|(\W+how [^(many)(much)])', \
'(name)|(choose)|(identify)', \
'(\A'+AUX_V_REGEX+' )|(([!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]){1} '+AUX_V_REGEX+' )'
]


# SAMPLE_TYPE:
# -1: don't sample randomly, sample by question type
# 0: sample the inverse of all the question types
# not -1 or 0: sample by question type
SAMPLE_TYPE = -1
# used when SAMPLE_TYPE == -1
QUESTION_TYPE = 3


# for parsing sentences using the stanford core nlp package with a python wrapper
# from https://github.com/dasmith/stanford-corenlp-python
# this is effective after launching the server by in parallel doing
# python corenlp.py

# import importdir
# importdir.do("/home/anglil/csehomedir/projects/dqa/stanford-corenlp-python", globals())

sys.path.insert(0, '/home/anglil/csehomedir/projects/dqa/dqa-eccv16/stanford-corenlp-python')
import jsonrpc
from simplejson import loads
server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(), jsonrpc.TransportTcpIp(addr=("127.0.0.1", 8080)))


# turn qa_pairs into hypotheses, test
def qa2hypo_test(args):
    root_dir = args.root_dir
    qa_path = os.path.join(root_dir, 'qa_pairs.json')
    qa_res_path = os.path.join(root_dir, 'qa_res.json')

    print "Loading json files ..."
    qa_pairs = json.load(open(qa_path, 'rb'))
    qa_pairs_list = qa_pairs['qa_pairs']

    # number of samples and the types of questions to sample
    k = SAMPLE_TYPE
    
    # execute the sampling (for the purpose of examining the result)
    q_type = QUESTION_TYPES[QUESTION_TYPE]
    qa_pairs_list = sample_qa(qa_pairs_list, k, q_type) # set the case lower in the function for questions
    
    # result file
    res = []

    ctr = 0
    for item in qa_pairs_list:
        question = item['question']
        ans = item['ans']
        question = question.lower()
        ans = ans.lower().strip('.')

        # determine the question type:
        if k != -1:
            q_type = get_question_type(question)

        if re.search('what '+AUX_V_DOESONLY_REGEX, question) or re.search('what '+AUX_V_DO_REGEX, question):

            print 'Question:', question
            print
            print 'Answer:', ans

            # test_patterns([q_type], question)
            sent = rule_based_transform(question, ans, q_type)
          
            print 'Result:', sent
            res.append({'Question':question, 'Answer':ans, 'Result':sent})

            ctr += 1
            print "--------------------------------------"
    
    print ctr
    print "Dumping json files ..."
    json.dump(res, open(qa_res_path, 'wb'))


# turn qa_pairs into hypotheses
def qa2hypo(question, answer):
    question = question.lower()
    answer = answer.lower().strip('.')

    # determine the question type:
    q_type = get_question_type(question)

    sent = rule_based_transform(question, answer, q_type)
    return sent


# determine the question type
def get_question_type(question):
    for q_type in QUESTION_TYPES:
        if re.search(q_type, question):
            return q_type
    return 'none of these'

# rule based qa2hypo transformation
def rule_based_transform(question, ans, q_type):
    if q_type == QUESTION_TYPES[0]:
        s, e = test_pattern(q_type, question)
        hypo = replace(question, s, e, ans)
    else:
        if q_type == QUESTION_TYPES[1]:
            s, e = test_pattern('when', question)
            if re.search('when '+AUX_V_DOES_REGEX, question):
                s2, e2 = test_pattern('when '+AUX_V_DOES_REGEX, question)
                hypo = replace(question, s2, e2, '')
                hypo = strip_nonalnum_re(hypo)+' in '+ans
            elif re.search('when '+AUX_V_DO_REGEX, question):
                s3, e3 = test_pattern('when '+AUX_V_DO_REGEX, question)
                hypo = replace(question, s3, e3, '')
                hypo = strip_nonalnum_re(hypo)+' in '+ans
            else:
                hypo = replace(question, s, e, ans)

        elif q_type == QUESTION_TYPES[2]:
            s, e = test_pattern('where', question)
            if re.search('where '+AUX_V_DOES_REGEX, question):
                s2, e2 = test_pattern('where '+AUX_V_DOES_REGEX, question)
                hypo = replace(question, s2, e2, '')
                hypo = strip_nonalnum_re(hypo)+' at '+ans
            elif re.search('where '+AUX_V_DO_REGEX, question):
                s3, e3 = test_pattern('where '+AUX_V_DO_REGEX, question)
                hypo = replace(question, s3, e3, '')
                hypo = strip_nonalnum_re(hypo)+' at '+ans
            else:
                hypo = replace(question, s, e, ans)

        elif q_type == QUESTION_TYPES[3]:
            
            if re.search('what '+AUX_V_DOESONLY_REGEX, question):
                s_aux, e_aux, s_vp, e_vp, first_VP=find_np_pos(question, ans, 'what '+AUX_V_DOESONLY_REGEX, node_type='VP')
                hypo = replace(question, e_vp, e_vp, ' '+ans)
                print 'hypo:', hypo
                hypo = replace(hypo, s_aux, e_aux, '')
                hypo = strip_nonalnum_re(hypo)
                
                # hypo = strip_nonalnum_re(hypo)+' '+ans
            elif re.search('what '+AUX_V_DO_REGEX, question):
                s_aux, e_aux, s_vp, e_vp, first_VP=find_np_pos(question, ans, 'what '+AUX_V_DO_REGEX, node_type='VP')
                hypo = replace(question, e_vp, e_vp, ' '+ans)
                print 'hypo:', hypo
                hypo = replace(hypo, s_aux, e_aux, '')
                hypo = strip_nonalnum_re(hypo)
                
                # hypo = strip_nonalnum_re(hypo)+' '+ans
            else:
                s, e = test_pattern('what', question)
                hypo = replace(question, s, e, ans)

        elif q_type == QUESTION_TYPES[4]:
            s, e = test_pattern('which', question)
            hypo = replace(question, s, e, ans)

        elif q_type == QUESTION_TYPES[5]:
            s, e = test_pattern('(who)|(whom)', question)
            hypo = replace(question, s, e, ans)

        elif q_type == QUESTION_TYPES[6]:
            s, e = test_pattern('why', question)
            hypo = strip_question_mark(question)+', '+ans
            if not re.search('because', ans, re.IGNORECASE):
                hypo = strip_question_mark(question)+', because '+ans

        elif q_type == QUESTION_TYPES[7]:
            s, e = test_pattern('(how many)|(how much)', question)
            hypo = replace(question, s, e, ans)

        elif q_type == QUESTION_TYPES[8]:
            s, e = test_pattern('(\Ahow )|(\W+how )', question)
            hypo = replace(question, s, e, ' '+ans+' is how ')

        elif q_type == QUESTION_TYPES[9]:
            s, e = test_pattern('(name)|(choose)|(identify)', question)
            hypo = replace(question, s, e, ans+' is')

        # if starting with aux_v, exchange the Verb and Noun
        # if it is an or question, choose the one that heuristically matches the answer
        elif q_type == QUESTION_TYPES[10]:
            # if re.search('(yes, )|(no, )', ans):
            #     s, e = test_pattern('(yes, )|(no, )', ans)
            #     hypo = replace(ans, s, e, '')
            # elif ' or ' in question:
            #     hypo = ans
            # elif re.search('yes\W??', ans):
            #     s, e = test_pattern('(\A'+AUX_V_REGEX+' )|(([!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]){1} '+AUX_V_REGEX+' )', question)
            #     hypo = replace(question, s, e, "")
            # elif re.search('no\W??', ans):
            #     s, e = test_pattern('(\A'+AUX_V_REGEX+' )|(([!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]){1} '+AUX_V_REGEX+' )', question)
            #     hypo = "not "+replace(question, s, e, "")

            if re.search('\A(((yes)\W+)|((yes)$))', ans):
                s_aux, e_aux, s_np, e_np, first_NP = find_np_pos(question, ans, q_type)
                hypo = replace(question, s_aux, e_np, first_NP + ' ' + question[s_aux:e_aux-1] + ' ')
            
            elif re.search('\A(((no)\W+)|((no)$))', ans):
                s_aux, e_aux, s_np, e_np, first_NP = find_np_pos(question, ans, q_type)
                hypo = replace(question, s_aux, e_np, first_NP + ' ' + question[s_aux:e_aux] + 'not ')

            elif re.search(' or ', question):
                s_aux, e_aux, s_np, e_np, first_NP = find_np_pos(question, ans, q_type)
                hypo = replace(question, s_aux, e_np, first_NP + ' ' + question[s_aux:e_aux-1] + ' ')
                s_candidate, e_candidate, candidate = find_or_pos(hypo, ans, q_type)
                hypo = replace(hypo, s_candidate, e_candidate, candidate)

            else:
                s, e = test_pattern('(\A'+AUX_V_REGEX+' )|(([!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]){1} '+AUX_V_REGEX+' )', question)
                hypo = replace(question, s, e, "")
                hypo = strip_nonalnum_re(hypo)+' '+ans

        else:
            hypo = strip_nonalnum_re(question)+' '+ans

    hypo = strip_question_mark(hypo)
    return hypo


# find the positions of the NPs or VPs around 'or'
def find_or_pos(question, ans, q_type):

    sent_parse = loads(server.parse(question))
    parse_tree = sent_parse['sentences'][0]['parsetree']
    tree = ParentedTree.fromstring(parse_tree)
    # print the tree
    tree.pretty_print()

    or_node = None
    for subtree in tree.subtrees(filter=lambda x: x.label() == 'CC'):
        # print "or position:", subtree.leaves()
        or_node = subtree
        break

    # left_siblings = []
    # l = or_node.left_sibling()
    # while l:
    #     left_siblings.append(l)
    #     l = l.left_sibling()

    # right_siblings = []
    # r = or_node.right_sibling()
    # while r:
    #     right_siblings.append(r)
    #     r = r.right_sibling()

    # print left_siblings
    # print right_siblings

    or_parent = or_node.parent()
    candidates_tok = or_parent.leaves()
    candidates_len = len(' '.join(candidates_tok))

    candidates_list = []
    item = ''
    for tok in candidates_tok:
        if (tok != ',') and (tok != 'or'):
            item = item + ' ' + tok
        else:
            candidates_list.append(item)
            item = ''
    candidates_list.append(item)

    candidate_chosen = candidates_list[0]
    for candidate in candidates_list:
        if ans in candidate:
            candidate_chosen = candidate
            break
    print "candidates_list:", candidates_list
    print "candidate_chosen:", candidate_chosen

    s0, e0 = test_pattern(candidates_list[0].strip(), question)
    s1, e1 = test_pattern(candidates_list[-1].strip(), question)

    return s0, e1, candidate_chosen



# find the positions of the aux_v and the first noun
def find_np_pos(question, ans, q_type, node_type='NP'):
    s_aux, e_aux = test_pattern(q_type, question)
    print '  %2d : %2d = "%s"' % (s_aux, e_aux-1, question[s_aux:e_aux])
                
    question_partial = question[s_aux:]
    if node_type=='VP':
        question_partial = question[e_aux:]
    sent_parse = loads(server.parse(question_partial))
    parse_tree = sent_parse['sentences'][0]['parsetree']
    print "Parse Tree:", parse_tree
    tree = Tree.fromstring(parse_tree)

    first_NP = None
    for subtree in tree.subtrees(filter=lambda x: x.label() == node_type):
        # print(subtree.leaves())
        first_NP = ' '.join(subtree.leaves())
        break
        # if re.search('(\A'+AUX_V_BE_REGEX+' )|(([!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]){1} '+AUX_V_BE_REGEX+' )', question):
        #     for subsubtree in subtree.subtrees(filter=lambda y: y.label() == 'NP'):
        #         first_NP = ' '.join(subsubtree.leaves())
        #         break
        # else:
        #     first_NP = ' '.join(subtree.leaves())
        #     break

    print node_type+':', first_NP
    print

    first_NP_len = 0
    if first_NP:
        first_NP_len = len(first_NP)
        s_np, e_np = test_pattern((first_NP.split(' '))[0], question)
    else:
        s_np = len(question)-1
    # s_np = e_aux+1
    e_np = s_np + first_NP_len

    return s_aux, e_aux, s_np, e_np, first_NP


# strip the question mark
def strip_question_mark(sent):
    if sent.endswith('?') or sent.endswith(':'):
        return sent[:-1]
    else:
        return sent

# strip any non alnum characters in the end
def strip_nonalnum_re(sent):
    return re.sub(r"^\W+|\W+$", "", sent)

# replace 
def replace(text, start, end, replacement):
    text_left = text[:start]
    text_right = text[end:]
    return text_left+replacement+text_right

# sample sentences
def sample_qa(qa_pairs_list, k, q_type):
    l = range(len(qa_pairs_list))
    l_sampled = []

    # random sampling
    if k != -1 and k != 0:
        l_sampled = random.sample(l, k)

    # inverse sampling
    elif k == 0:
        return sample_qa_inverse(qa_pairs_list)

    # sample by question type (k == -1)
    else:
        for num in l:
            q = qa_pairs_list[num]['question'].lower() # use the lower case for all
            # --- regex ---
            if re.search(q_type, q):
                l_sampled.append(num)

    return [qa_pairs_list[i] for i in l_sampled]

# sample sentences -- the inverse set; this is a helper to sample_qa
def sample_qa_inverse(qa_pairs_list):
    l = range(len(qa_pairs_list))
    l_sampled = []

    for num in l:
        q = qa_pairs_list[num]['question'].lower() # use the lower case for all
        flag = 0
        for q_type in QUESTION_TYPES:
            # --- regex ---
            if re.search(q_type, q) != None:
                flag = 1
                break
        if flag == 0:
            l_sampled.append(num)

    return [qa_pairs_list[i] for i in l_sampled]


# for print purpose
def test_patterns(patterns, text):
    """Given source text and a list of patterns, look for
    matches for each pattern within the text and print
    them to stdout.
    """
    # Show the character positions and input text
    # print
    # print ''.join(str(i/10 or ' ') for i in range(len(text)))
    # print ''.join(str(i%10) for i in range(len(text)))
    # print text

    # Look for each pattern in the text and print the results
    for pattern in patterns:
        print
        print 'Matching "%s"' % pattern
        # --- regex ---
        for match in re.finditer(pattern, text):
            s = match.start()
            e = match.end()
            print '  %2d : %2d = "%s"' % \
                (s, e-1, text[s:e])
            # print '    Groups:', match.groups()
            # if match.groupdict():
            #     print '    Named groups:', match.groupdict()
            print
    return

# for return purpose
def test_pattern(pattern, text):
    match = re.search(pattern, text)
    pos = len(text)-1
    if not match:
        return pos, pos
    s = match.start()
    e = match.end()
    # print '  %2d : %2d = "%s"' % (s, e-1, text[s:e])
    return s, e

if __name__ == "__main__":
    # question = "Does frog fly?"
    # answer = "no"
    # sent = qa2hypo(question, answer)
    # print(sent)

    qa2hypo_test(get_args())

