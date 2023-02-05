from crawling import get_link
from crawling import data_collect
from keyword_extraction import book_keyword

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

import pandas as pd

def hash_tag_filter(hash_filter, hash_tag):
    hash_to_list = hash_tag[1:].split('#')
    filtered = ''
    for temp in hash_to_list:
        cnt = 0
        for i in hash_filter:
            if not i in temp:
                cnt += 1
        if cnt == len(hash_filter):
            filtered += ' ' + temp
    return filtered[1:]

def sentence_filter(sentence_filter_list, intro):
    try: # 문장이 존재할 경우
        cnt = 0
        for idx, value in enumerate(intro):
            if value == "\n":
                if not (intro[idx+cnt-1] == "!" or intro[idx+cnt-1] == "." \
                        or intro[idx+cnt-1] == "?" or intro[idx+cnt-1] == "\n"):
                    intro = intro[:idx + cnt] + '.' + intro[idx + cnt:]
                    cnt += 1
        intro = intro.replace("\n", "")
        filtered_intro = ''
        prior = 0
        for idx, value in enumerate(intro):
            if value == '!' or value == '?' or value == '.':
                temp = intro[prior:idx+1]
                cnt = 0
                for i in sentence_filter_list:
                    if not i in temp:
                        cnt += 1
                if cnt == len(sentence_filter_list):
                    filtered_intro += temp
                prior = idx + 1
        return filtered_intro
    except:
        return intro

def modify(df):
    book_title = list(df.columns)
    book_attribute = list(df.index)
    nan_list = []
    for title in book_title: # 모든 책
        if df[title]['hash_tag'] != 'False':
            temp = df[title]['hash_tag']
            filtered = hash_tag_filter(hash_filter, temp)
            df[title]['hash_tag'] = filtered
        else:
            df[title]['hash_tag'] = ''
    for title in book_title: # 모든 책
        cnt = 0
        for attribute in book_attribute[1:]: # hash tag 제외
            temp = df[title][attribute]
            filtered = sentence_filter(sentence_filter_list, temp)
            df[title][attribute] = filtered
            if df[title][attribute] == '':
                cnt += 1
        if cnt == 4:
            nan_list.append(title)
    if nan_list:
        df_final = df.drop(nan_list, axis = 'columns')

    return df_final

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# # 책 링크 수집하기
url = 'http://www.yes24.com/24/category/bestseller?CategoryNumber=001001026&sumgb=06&PageNumber='
df = get_link(driver, url, 1, 11)
df.to_csv('self_dev_link.csv', encoding='cp949')

# 링크를 바탕으로 책 정보 수집하기
# df = pd.read_csv('./travel_link.csv')
df_data = data_collect(driver, df)
df_data.to_csv('self_dev_raw.csv', encoding='cp949')

# 책 정보 필터링하기
hash_filter = ['스페셜', '에디션', '추천', '북클러버', 'TV', '소개', '북클럽', '분철', '북클럽',\
                '올해의', '책읽아웃', '쿠폰', '유퀴즈', '방탄', '새해', '북카페', '읽은'\
                    '영원', '알쓸', '뭐읽지', '유튜', '나다움', '북스타그램', '제본']
sentence_filter_list = ['셀러', '한정판', '출간', '하드커버', 'edition', '초판', \
                       '『', '★', '저자', '전자책', '유튜', '포스트', '타임즈', '선물'\
                        '*', '신작', '에디션']
df_data_m = pd.read_csv('D:\\book_data\\self_dev_raw.csv', encoding='cp949', index_col=0)
modified_df = modify(df_data_m)
modified_df.to_csv('self_dev_data.csv', encoding='cp949')

# # df = pd.read_csv('./travel_data.csv', encoding='cp949', index_col=0)
# travel_1_200 = book_keyword(modified_df)
# travel_1_200.to_csv('ss2.csv', encoding='cp949')