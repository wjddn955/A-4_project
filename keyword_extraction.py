#pip install sentence_transformers
#pip install konlpy

import numpy as np
import itertools

from konlpy.tag import Okt, Hannanum
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import pandas as pd
df = pd.read_csv('./travel_data.csv', encoding='cp949', index_col=0)

def input_doc(doc, range_1, range_2, top_n):
    okt = Okt()
    tokenized_doc = okt.pos(doc)
    tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])

    n_gram_range = (range_1, range_2)
    count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
    candidates = count.get_feature_names_out()

    #model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('jhgan/ko-sroberta-nli') # 좋다
    doc_embedding = model.encode([doc])
    candidate_embeddings = model.encode(candidates)

    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

    return candidates, doc_embedding, candidate_embeddings, keywords

from collections import Counter
def book_keyword(df):
    book_title = list(df.columns)
    book_attribute = list(df.index)
    hannanum = Hannanum()

    for title in book_title:
        cnt = 0
        total_keyword = []
        intro_final = []
        yes24_final = []
        pub_final = []
        rec_final = []
        intro_keyword = []
        yes24_keyword = []
        pub_keyword = []
        rec_keyword = []

        doc = df[title]['book_introduction']
        if type(doc) == str:
            cnt += 1
            intro = []
            intro_keyword = []
            intro_final = []
            nouns_init = hannanum.nouns(doc)
            nouns = [i for i in nouns_init if len(i) > 1]
            if type(df[title]['hash_tag']) == str: # 해시태그가 있다면
                doc += df[title]['hash_tag']
            candidates, doc_embedding, candidate_embeddings, keyword = input_doc(doc, 2, 3, 20)
            for i in keyword:
                tmp = i.split()
                for j in tmp:
                    intro.append(j)
            for word in intro:
                for noun in list(set(nouns)):
                    if word == noun:
                        intro_keyword.append(word)
            intro_counter = Counter(intro_keyword).most_common()
            for key, value in enumerate(intro_counter):
                if value[1] > 3: # parameter
                    intro_final.append(value[0])
                    
        doc = df[title]['yes24_review']
        if type(doc) == str:
            cnt += 1
            yes24 = []
            yes24_keyword = []
            yes24_final = []
            nouns_init = hannanum.nouns(doc)
            nouns = [i for i in nouns_init if len(i) > 1]
            if type(df[title]['hash_tag']) == str: # 해시태그가 있다면
                doc += df[title]['hash_tag']
            candidates, doc_embedding, candidate_embeddings, keyword = input_doc(doc, 2, 3, 20)
            for i in keyword:
                tmp = i.split()
                for j in tmp:
                    yes24.append(j)
            for word in yes24:
                for noun in list(set(nouns)):
                    if word == noun:
                        yes24_keyword.append(word)
            yes24_counter = Counter(yes24_keyword).most_common()
            for key, value in enumerate(yes24_counter):
                if value[1] > 3: # parameter
                    yes24_final.append(value[0])

        doc = df[title]['pub_review']
        if type(doc) == str:
            cnt += 1
            pub = []
            pub_keyword = []
            pub_final = []
            nouns_init = hannanum.nouns(doc)
            nouns = [i for i in nouns_init if len(i) > 1]
            if type(df[title]['hash_tag']) == str: # 해시태그가 있다면
                doc += df[title]['hash_tag']
            candidates, doc_embedding, candidate_embeddings, keyword = input_doc(doc, 2, 3, 20)
            for i in keyword:
                tmp = i.split()
                for j in tmp:
                    pub.append(j)
            for word in pub:
                for noun in list(set(nouns)):
                    if word == noun:
                        pub_keyword.append(word)
            pub_counter = Counter(pub_keyword).most_common()
            for key, value in enumerate(pub_counter):
                if value[1] > 3: # parameter
                    pub_final.append(value[0])

        doc = df[title]['recommend']
        if type(doc) == str:
            cnt += 1
            rec = []
            rec_keyword = []
            rec_final = []
            nouns_init = hannanum.nouns(doc)
            nouns = [i for i in nouns_init if len(i) > 1]
            if type(df[title]['hash_tag']) == str: # 해시태그가 있다면
                doc += df[title]['hash_tag']
            candidates, doc_embedding, candidate_embeddings, keyword = input_doc(doc, 2, 3, 20)
            for i in keyword:
                tmp = i.split()
                for j in tmp:
                    rec.append(j)
            for word in rec:
                for noun in list(set(nouns)):
                    if word == noun:
                        rec_keyword.append(word)
            rec_counter = Counter(rec_keyword).most_common()
            for key, value in enumerate(rec_counter):
                if value[1] > 3: # parameter
                    rec_final.append(value[0])

        total_keyword += intro_final
        total_keyword += yes24_final
        total_keyword += pub_final
        total_keyword += rec_final
        counter = Counter(total_keyword).most_common()
        final_keyword = []
        for key, value in enumerate(counter):
            #if value[1] > cnt-2:
            final_keyword.append(value[0])
        keyword_list = [0 for i in range(10)]
        try:
            keyword_list[0] = final_keyword[0]
            keyword_list[1] = final_keyword[1]
            keyword_list[2] = final_keyword[2]
            keyword_list[3] = final_keyword[3]
            keyword_list[4] = final_keyword[4]
            keyword_list[5] = final_keyword[5]
            keyword_list[6] = final_keyword[6]
            keyword_list[7] = final_keyword[7]
            keyword_list[8] = final_keyword[8]
            keyword_list[9] = final_keyword[9]
        except:
            pass

        if keyword_list[0] == 0: # 겹치는 키워드가 하나도 없을 경우
            minimum = intro_keyword + yes24_keyword + pub_keyword + rec_keyword
            minimum_counter = Counter(minimum).most_common()
            mini = []
            rank = minimum_counter[0][1]
            for key, value in enumerate(minimum_counter):
                if rank == value[1]:
                    final_keyword.append(value[0])
                else:
                    break
            try:
                keyword_list[0] = final_keyword[0]
                keyword_list[1] = final_keyword[1]
                keyword_list[2] = final_keyword[2]
                keyword_list[3] = final_keyword[3]
                keyword_list[4] = final_keyword[4]
                keyword_list[5] = final_keyword[5]
                keyword_list[6] = final_keyword[6]
                keyword_list[7] = final_keyword[7]
                keyword_list[8] = final_keyword[8]
                keyword_list[9] = final_keyword[9]
            except:
                pass
            
        df_series = pd.Series(keyword_list, name=title)
        try:
            df_stitched = pd.concat([df_stitched, df_series], axis = 1)
        except:
            df_stitched = df_series.to_frame(name=title)
    return df_stitched
   
def diary_keyword(diary):
    diary_key = []
    diary_noun = []
    hannanum = Hannanum()
    diary_final = []
    nouns_init = hannanum.nouns(diary)
    nouns = [i for i in nouns_init if len(i) > 1]
    candidates, doc_embedding, candidate_embeddings, keyword = input_doc(diary, 2, 3, 20)
    for i in keyword:
        tmp = i.split()
        for j in tmp:
            diary_key.append(j)
    for word in diary_key:
        for noun in list(set(nouns)):
            if word == noun:
                diary_noun.append(word)
    diary_counter = Counter(diary_noun).most_common()
    for key, value in enumerate(diary_counter):
        if value[1] > 3:
            diary_final.append(value[0])
    return diary_final
