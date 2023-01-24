import numpy as np
from tools.features import norm_1_sig, movmean
import matplotlib.pyplot as plt
import re

def query_search(X, K1, K2, K3, query, win_size):
# Parse text - ---------------------------------------------
# Idea 2 - Check if keywords with space or brackets
# indicating window -> Now using regex!!!
# Idea 3(Now in version 4 of the app) - include followed by / next to / after / before(
# for now...only using 1 time and
# not multiple instances of that operator)
#--------------------------------------------------------
    # 1 - check if multi_part search or single part search:
    search_levels = np.shape(query)[0]

    # if (search_levels > 1):
    #     #do multilevel search
    #     [scores, signal_groups] = multi_level_search(query)
    # else:
    #do singlelevel search
    scores, signal_groups = single_level_search(X, K1, K2, K3, query[0], win_size)

    return scores


def single_level_search(X, K1, K2, K3, query, win_size):
    # 2 - check if single, or multiple signal
    signal_groups = re.split("s\d:", query)[1:]
    signal_indexs = re.findall("s(\d):", query)
    # 3 - apply regex to query, separating brackets and single, now also taking care of multisignals

    ##### If there are multiple signals being queried
    if (np.ndim(X) > 1):
        # if("followed by" in signal_groups[1]):
        #     # If there is a "followed by" statement for multiple signals, a different approach should be made
        #     group1 = signal_groups[1].split("followed by")
        #     group2 = signal_groups[3]
        #     # scores = norm_1_sig(multi_query_search_followedby(X, K1, K2, K3, [group1[1], group2], sig_indexs, win_size))
        # # If normal interaction between signals, than it is the normal addition of scores
        # else:
        scores = multi_query_search(X, K1, K2, K3, signal_groups, signal_indexs, win_size)
    else:
        #### Only 1 signal being queried
        # 1 - Check the presence of sequence operator
        query = signal_groups[0]
        re_followed_by = re.search("\(.+\) followed by \(.+\)", query)
        # check scores for followed by
        if(re_followed_by is None):
            score_fb = 0
        else:
            followed_by_match = re_followed_by.group()
            re_remaining = " ".join([" ".join(list(filter(None, s_i.split(" ")))) for s_i in re.split("\(.+\) followed by \(.+\)", query)])
            query = "".join(re_remaining)
            score_fb = norm_1_sig(followed_by_search(X, K1, K2, K3, followed_by_match, win_size))

        scores = uni_query_search(X, K1, K2, K3, query, win_size)
        scores = scores + score_fb

    scores = norm_1_sig(scores)

    return scores, signal_groups


def uni_query_search(X, K1, K2, K3, query_group, win_size):
    scores = np.zeros(len(X) - win_size+1)
    # keywords
    re_bracket_group = re.findall("\[.+?\]", query_group)
    if(len(re_bracket_group)>0):
        # 2 - calculate scores for each bracket group and single keyword
        # 2.1 - start with brackets
        for bracket_i in re_bracket_group:
            # calculate score and add to the previous array
            scores = scores + grouped_followed_by(X, K2, K3, bracket_i[1:-1], win_size)

    # 2.2 - calculate scores for each remaining singular keyword or all keywords in case the split does not return anything
    merged_groups = " ".join(list(filter(None, [s_i.strip() for s_i in re.split("\[.+?\]", query_group)])))
    splitted_keywords = merged_groups.split(" ")
    for keyword_i in splitted_keywords:
        keyword_score = single_keyword_score_estimation(K1, keyword_i, win_size)
        scores = scores + keyword_score

    return scores

def single_keyword_score_estimation(KEY_VECTOR1, keyword, win_size):
    #check for operators on keyword
    if("!" in keyword):
        keyword = str(keyword).lower()
        keyword = keyword[1:]
        single_keyword_score = 1 - KEY_VECTOR1[keyword]
    elif(keyword == ""):
        #if empty, it does not contribute to the scoring function
        single_keyword_score = 0
    else:
        keyword = str(keyword).lower()
        single_keyword_score = KEY_VECTOR1[keyword]

    # a = np.floor(win_size/2).astype(int)
    # single_keyword_score = single_keyword_score[a:-a-1]

    return single_keyword_score

def single_keyword_score_estimation_multi(KEY_VECTOR1, keyword, index_i, win_size):
    #check for operators on keyword
    if("!" in keyword):
        keyword = str(keyword).lower()
        keyword = keyword[2:]
        key_vector = KEY_VECTOR1[keyword]
        single_keyword_score = 1 - key_vector[index_i, :]
    else:
        keyword = str(keyword).lower()
        key_vector = KEY_VECTOR1[keyword]
        single_keyword_score = key_vector[index_i, :]

    a = np.floor(win_size/2).astype(int)
    single_keyword_score = single_keyword_score[a:-a-1]

    return single_keyword_score

def followed_by_search(X, K1, K2, K3, query, win_size):
    queries = query.split("followed by")
    query1 = queries[0][1:-2]
    query2 = queries[1][2:-1]

    score_pre = uni_query_search(X, K1, K2, K3, str(query1), win_size)
    score_pos = uni_query_search(X, K1, K2, K3, str(query2), win_size)
    scores = np.zeros(len(X)-win_size+1)
    scores[:len(score_pre)-win_size] = score_pre[:-win_size] + score_pos[win_size:]

    return scores


def grouped_followed_by(X, K2, K3, window_query, win_size):
    keywords = window_query.split(" ")
    # extract keywords for each signal, subwindowed and normalized
    norm_mean_scores = grouped_followed_by_extract_score(X, K2, K3, keywords, win_size)

    return norm_mean_scores

def grouped_followed_by_multi(X, K2, K3, window_query, index_i, win_size):
    keywords = window_query.split(" ")
    #extract keywords for each signal, subwindowed and normalized
    norm_mean_scores = grouped_followed_by_extract_score_multi(X, K2, K3, keywords, index_i, win_size)

    return norm_mean_scores


def grouped_followed_by_extract_score_multi(X, KEY_VECTOR2, KEY_VECTOR3, keywords, index_i, win_size):
    if (len(keywords) < 3):
        key_vector = KEY_VECTOR2
    elif len(keywords) > 2:
        key_vector = KEY_VECTOR3

    win_sub_size = np.floor(win_size / len(keywords)).astype(int)
    scores = np.zeros(len(keywords), np.shape(X)[1] - win_size)

    for i in range(len(keywords)):
        # check for operators on keyword
        if ("!" in keywords[i]):
            keyword = str(keywords[i]).lower()
            keyword = keyword[1:]
            search_op = 1
        else:
            keyword = str(keywords[i]).lower()
            search_op = 2

        a = (i - 1) * win_sub_size + 1
        b = (np.shape(X)[1] - win_size) + ((i - 1) * win_sub_size)

        #get vector of values associated with the keyword written for a specific window
        # Added a movmean to each keyword when used, based on the win_sub_size.
        if (keyword == "."):
            param_vec = np.zeros(1, b - a + 1)
        else:
            data_i = key_vector[keyword]
            data_i = movmean(data_i[index_i,:], np.floor(win_sub_size / 2).astype(int))
            param_vec = data_i[a:b]


        if(search_op == 1):
            scores[i,:] = 1 - param_vec
        elif(search_op == 2):
            scores[i,:] = param_vec

    brackets_scores = norm_1_sig(sum(scores, 1))

    return brackets_scores

def grouped_followed_by_extract_score(X, KEY_VECTOR2, KEY_VECTOR3, keywords, win_size):
    if (len(keywords) < 3):
        key_vector = KEY_VECTOR2
    elif(len(keywords) > 2):
        key_vector = KEY_VECTOR3

    win_sub_size = int(win_size / (len(keywords)))
    scores = np.zeros((len(keywords), len(X) - win_size + 1))

    for i in range(len(keywords)):
        # check for operators on keyword
        if ("!" in (keywords[i])):
            keyword = str(keywords[i]).lower()
            keyword = keyword[1:]
            search_op = 1
        else:
            keyword = str(keywords[i]).lower()
            search_op = 2

        a = i * win_sub_size
        b = (len(X) - win_size + 1) + (i * win_sub_size)

        # Check which keyword is written
        if (keyword == "."):
            param_vec = np.zeros(b - a)
        else:
            data_i = key_vector[keyword]
            mm_data_i = movmean(data_i[a:], win_sub_size)
            param_vec = np.zeros(b - a)
            if(len(param_vec)>len(mm_data_i)):
                param_vec[:len(mm_data_i)] = mm_data_i
            else:
                param_vec = mm_data_i[:b-a]

            # param_vec = data_i[a:]
            # print(len(param_vec))
            # plt.plot(param_vec)

        if (search_op == 1):
            scores[i, :] = 1 - param_vec[:b-a]
        elif(search_op == 2):
            scores[i, :] = param_vec[:b-a]

    scores = norm_1_sig(np.mean(scores, 0))
    # plt.plot(scores)
    # plt.plot(norm_1_sig(X))
    # plt.show()
    return scores

def multi_query_search(X, K1, K2, K3, query_groups, signal_indxs, win_size):
    scores_sig_i = np.zeros(np.shape(X)[0]-win_size+1)
    for i in range(len(query_groups)):
        #search the query on the previous signal's index
        sig_i = signal_indxs[i]
        scores_sig_i = scores_sig_i + norm_1_sig(uni_query_search(X[:, int(sig_i)-1], K1[sig_i], K2[sig_i], K3[sig_i], query_groups[i], win_size))

    return scores_sig_i