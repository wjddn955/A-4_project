import pandas as pd
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

# 책 제목
def title(driver):
    title_id = driver.find_element(By.ID, "yDetailTopWrap")
    title_right = title_id.find_element(By.CLASS_NAME, "topColRgt")
    title = title_right.find_element(By.CLASS_NAME, "gd_titArea")
    main_title = title.find_element(By.CLASS_NAME, "gd_name")
    try:
        sub_title = title.find_element(By.CLASS_NAME, "gd_nameE")
        final_title = main_title.text + '(' + sub_title.text + ')'
    except:
        final_title = main_title.text
    return final_title

# 이 상품의 태그
def hash_tag(driver):
    try:
        tag = driver.find_element(By.CLASS_NAME, "gd_tagArea")
        tag_list = tag.find_elements(By.CLASS_NAME, "tag ")
        hashtag = tag_list[0].text
        for i in range(1, len(tag_list)):
            hashtag += tag_list[i].text
    except:
        return False
    return hashtag

# 책 소개
def book_introduction(driver):
    try:
        yes = driver.find_element(By.ID, "infoset_introduce")
        info = yes.find_element(By.CLASS_NAME, "infoSetCont_wrap")
        intro = info.find_element(By.CLASS_NAME, "infoWrap_txtInner")
    except:
        return False
    return intro.text

# 책 속으로
def into_the_book(driver):
    try:
        yes = driver.find_element(By.ID, "infoset_inBook")
        info = yes.find_element(By.CLASS_NAME, "infoSetCont_wrap")
        into = info.find_element(By.CLASS_NAME, "infoWrap_txtInner")
    except:
        return False
    return into.text

# yes24 리뷰
def yes24_review(driver):
    try:
        yes = driver.find_element(By.ID, "infoset_yesReivew")
        info = yes.find_element(By.CLASS_NAME, "infoSetCont_wrap")
        review = info.find_element(By.CLASS_NAME, "infoWrap_txt")
    except:
        return False
    return review.text

# 출판사 리뷰
def pub_review(driver):
    try:
        pub = driver.find_element(By.ID, "infoset_pubReivew")
        info = pub.find_element(By.CLASS_NAME, "infoSetCont_wrap")
        review = info.find_element(By.CLASS_NAME, "infoWrap_txt")
    except:
        return False
    return review.text

# 추천평
def recommend(driver):
    try:
        pub = driver.find_element(By.ID, "infoset_nomiCmt")
        info = pub.find_element(By.CLASS_NAME, "infoSetCont_wrap")
        review = info.find_element(By.CLASS_NAME, "infoWrap_txt")
    except:
        return False
    return review.text

# 올해의 책 추천평
def year_recommend(driver):
    try:
        year_rec = driver.find_element(By.ID, "infoset_boyRecommendCommentList")
        total = year_rec.find_element(By.CLASS_NAME, "yesUI_pagenS")
        tag_a = total.find_elements(By.TAG_NAME, 'a')
        href = tag_a[-1].get_attribute("href")
        for i in range(1, len(href)):
            if href[-1 * i] == "=":
                point = i
                break
        end_page, address = href[-1 * point + 1:], href[:-1 * point + 1]
        total_review = ''
        for i in range(1, int(end_page) + 1):
            link = address + str(i)
            driver.get(link)
            total_rec = driver.find_element(By.ID, "infoset_boyRecommendCommentList")
            each_rec = total_rec.find_element(By.CLASS_NAME, "infoSetCont_wrap.rvCmtRow_cont.tp_boy.clearfix")
            # 빈칸이 있는 경우 .으로 대체해야 함
            odd = total_rec.find_elements(By.CLASS_NAME, 'cmtInfoGrp.cmt_odd')
                
            for temp in odd:
                review = temp.find_element(By.CLASS_NAME, 'cmt_cont')
                total_review += review.text + ' '
                
            even = total_rec.find_elements(By.CLASS_NAME, 'cmtInfoGrp.cmt_even')
            for temp in even:
                review = temp.find_element(By.CLASS_NAME, 'cmt_cont')
                total_review += review.text +' '
    except:
        return False
    return total_review

# 회원 리뷰 추천순
def review_recommend(driver):
    try:
        customer_rev = driver.find_element(By.ID, "infoset_reivew")
        total_rev = customer_rev.find_element(By.CLASS_NAME, "infoSetCont_wrap.reviewRow_cont.tp_tab.yesComLoadingArea")
        rev_sort = total_rev.find_element(By.CLASS_NAME, "review_sortRgt")
        tag_a = rev_sort.find_elements(By.TAG_NAME, 'a')
        href_list = []
        for temp in tag_a:
            href_list.append(temp.get_attribute("href"))
        review_recommend = href_list[1]
        driver.get(review_recommend)
        end_check = driver.find_element(By.CLASS_NAME, "review_sortLft")
        end = end_check.find_element(By.CLASS_NAME, "yesUI_pagenS")
        tag_a = end.find_elements(By.TAG_NAME, 'a')
        href = tag_a[-1].get_attribute("href")
        for i in range(1, len(href)):
            if href[-1 * i] == "=":
                point = i
                break
        end_page, address = href[-1 * point + 1:], href[:-1 * point + 1]
        total_review = []
        for i in range(1, int(end_page) + 1):
            link = address + str(i)
            driver.get(link)
            total_rev = driver.find_element(By.CLASS_NAME, "infoSetCont_wrap.reviewRow_cont.tp_tab.yesComLoadingArea")
            each_rev = total_rev.find_elements(By.CLASS_NAME, "reviewInfoGrp.lnkExtend")
            for temp2 in each_rev:
                total_review.append(temp2.text)
    except:
        return False
    return total_review

# 회원 리뷰 별점순 기존
def review_stars_first(driver):
    try:
        customer_rev = driver.find_element(By.ID, "infoset_reivew")
        total_rev = customer_rev.find_element(By.CLASS_NAME, "infoSetCont_wrap.reviewRow_cont.tp_tab.yesComLoadingArea")
        rev_sort = total_rev.find_element(By.CLASS_NAME, "review_sortRgt")
        tag_a = rev_sort.find_elements(By.TAG_NAME, 'a')
        href_list = []
        for temp in tag_a:
            href_list.append(temp.get_attribute("href"))
        review_star = href_list[2]
        driver.get(review_star)
        end_check = driver.find_element(By.CLASS_NAME, "review_sortLft")
        end = end_check.find_element(By.CLASS_NAME, "yesUI_pagenS")
        tag_a = end.find_elements(By.TAG_NAME, 'a')
        href = tag_a[-1].get_attribute("href")
        for i in range(1, len(href)):
            if href[-1 * i] == "=":
                point = i
                break
        end_page, address = href[-1 * point + 1:], href[:-1 * point + 1]
        total_review = []
        for i in range(1, int(end_page) + 1):
            link = address + str(i)
            driver.get(link)
            total_rev = driver.find_element(By.CLASS_NAME, "infoSetCont_wrap.reviewRow_cont.tp_tab.yesComLoadingArea")
            each_rev = total_rev.find_elements(By.CLASS_NAME, "reviewInfoGrp.lnkExtend")
            for temp2 in each_rev:
                total_review.append(temp2.text)
    except:
        return False
    return total_review

# 한줄평
def one_line(driver):
    try:
        driver.implicitly_wait(2)
        customer_rev = driver.find_element(By.ID, "infoset_rvCmt")
        customer_rev_into = customer_rev.find_element(By.ID, "infoset_oneCommentList")
        total_rev = customer_rev_into.find_element(By.CLASS_NAME, "rvCmt_sortRgt")
        tag_a = total_rev.find_elements(By.TAG_NAME, 'a')
        href_list = []
        for temp in tag_a:
            href_list.append(temp.get_attribute("href"))
        review_star = href_list[2]
        driver.get(review_star)
        end_check = driver.find_element(By.CLASS_NAME, "rvCmt_sortLft")
        end = end_check.find_element(By.CLASS_NAME, "yesUI_pagenS")
        tag_a = end.find_elements(By.TAG_NAME, 'a')
        href = tag_a[-1].get_attribute("href")
        for i in range(1, len(href)):
            if href[-1 * i] == "=":
                point = i
                break
        end_page, address = href[-1 * point + 1:], href[:-1 * point + 1]
        total_review = []
        for i in range(1, int(end_page) + 1):
            link = address + str(i)
            driver.get(link)
            total_rev = driver.find_element(By.CLASS_NAME, "infoSetCont_wrap.rvCmtRow_cont.clearfix")
            
            odd = total_rev.find_elements(By.CLASS_NAME, 'cmtInfoGrp.cmt_odd')
            for temp in odd:
                review = temp.find_element(By.CLASS_NAME, 'cmt_cont')
                rate = temp.find_element(By.CLASS_NAME, 'cmt_rating')
                total_review.append(rate.text + ' ' + review.text)
                        
            even = total_rev.find_elements(By.CLASS_NAME, 'cmtInfoGrp.cmt_even')
            for temp in even:
                review = temp.find_element(By.CLASS_NAME, 'cmt_cont')
                rate = temp.find_element(By.CLASS_NAME, 'cmt_rating')
                total_review.append(rate.text + ' ' + review.text)
    except:
        return False
    return total_review

# 회원 리뷰 별점순
def review_stars(driver):
    try:
        customer_rev = driver.find_element(By.ID, "infoset_reivew")
        total_rev = customer_rev.find_element(By.CLASS_NAME, "infoSetCont_wrap.reviewRow_cont.tp_tab.yesComLoadingArea")
        rev_sort = total_rev.find_element(By.CLASS_NAME, "review_sortRgt")
        tag_a = rev_sort.find_elements(By.TAG_NAME, 'a')
        href_list = []
        for temp in tag_a:
            href_list.append(temp.get_attribute("href"))
        review_star = href_list[2]
        driver.get(review_star)
        end_check = driver.find_element(By.CLASS_NAME, "review_sortLft")
        end = end_check.find_element(By.CLASS_NAME, "yesUI_pagenS")
        tag_a = end.find_elements(By.TAG_NAME, 'a')
        href = tag_a[-1].get_attribute("href")
        for i in range(1, len(href)):
            if href[-1 * i] == "=":
                point = i
                break
        end_page, address = href[-1 * point + 1:], href[:-1 * point + 1]
        total_review = []
        for i in range(1, int(end_page) + 1):
            link = address + str(i)
            driver.get(link)
            total_rev = driver.find_element(By.CLASS_NAME, "infoSetCont_wrap.reviewRow_cont.tp_tab.yesComLoadingArea")
            each_rev = total_rev.find_elements(By.CLASS_NAME, "reviewInfoGrp.lnkExtend")
            for temp2 in each_rev:
                rate = temp2.find_element(By.CLASS_NAME, "reviewInfoTop")
                rate_detail = rate.find_element(By.CLASS_NAME, "review_rating")
                only_rev = temp2.find_element(By.CLASS_NAME, "reviewInfoBot.origin")
                detail = only_rev.find_element(By.CLASS_NAME, "review_cont")
                total_review.append(rate_detail.text[:21] + ' ' + detail.text)
    except:
        return False
    return total_review

def find_error(error, data):
    for idx, key in enumerate(data):
        temp = key.find(error)
        if temp != -1: # error
            break
    return idx, temp

# 한줄평_sum
def one_line_str(driver):
    try:
        driver.implicitly_wait(2)
        customer_rev = driver.find_element(By.ID, "infoset_rvCmt")
        customer_rev_into = customer_rev.find_element(By.ID, "infoset_oneCommentList")
        total_rev = customer_rev_into.find_element(By.CLASS_NAME, "rvCmt_sortRgt")
        tag_a = total_rev.find_elements(By.TAG_NAME, 'a')
        href_list = []
        for temp in tag_a:
            href_list.append(temp.get_attribute("href"))
        review_star = href_list[2]
        driver.get(review_star)
        end_check = driver.find_element(By.CLASS_NAME, "rvCmt_sortLft")
        end = end_check.find_element(By.CLASS_NAME, "yesUI_pagenS")
        tag_a = end.find_elements(By.TAG_NAME, 'a')
        href = tag_a[-1].get_attribute("href")
        for i in range(1, len(href)):
            if href[-1 * i] == "=":
                point = i
                break
        end_page, address = href[-1 * point + 1:], href[:-1 * point + 1]
        total_review = ''
        for i in range(1, int(end_page) + 1):
            link = address + str(i)
            driver.get(link)
            total_rev = driver.find_element(By.CLASS_NAME, "infoSetCont_wrap.rvCmtRow_cont.clearfix")
            
            odd = total_rev.find_elements(By.CLASS_NAME, 'cmtInfoGrp.cmt_odd')
            for temp in odd:
                review = temp.find_element(By.CLASS_NAME, 'cmt_cont')
                # rate = temp.find_element(By.CLASS_NAME, 'cmt_rating')
                total_review += review.text
                        
            even = total_rev.find_elements(By.CLASS_NAME, 'cmtInfoGrp.cmt_even')
            for temp in even:
                review = temp.find_element(By.CLASS_NAME, 'cmt_cont')
                # rate = temp.find_element(By.CLASS_NAME, 'cmt_rating')
                total_review += review.text
    except:
        return False
    return total_review

def get_link(driver, url, start, end):
    link_list = []
    for i in range(start,end):
        temp = []
        link = url + str(i)
        driver.get(link)
        driver.implicitly_wait(1)
        book_list = driver.find_element(By.ID, "category_layout")
        href_list = book_list.find_elements(By.CLASS_NAME, "goodsTxtInfo")
        for i in href_list:
            tag_a = i.find_element(By.TAG_NAME, 'a')
            href = tag_a.get_attribute("href")
            temp.append(href)
        link_list += temp
    df = pd.DataFrame(link_list, columns = ['link'])

    return df

def data_collect(driver, df):
    book_attribute = ['hash_tag', 'book_introduction', \
    'yes24_review', 'pub_review', 'recommend']
    for i in range(len(df)):
        link = df['link'][i]
        driver.get(link)
        time.sleep(1)
        try:
            book_title = title(driver)
            book_hash_tag = hash_tag(driver)
            book_book_introduction = book_introduction(driver)
            if book_book_introduction:
                for idx, value in enumerate(book_book_introduction):
                        if value == '\u2023':
                            book_book_introduction = book_book_introduction[:idx] + book_book_introduction[idx + 1:]
            if book_book_introduction == '':
                book_book_introduction = False
            book_yes24_review = yes24_review(driver)
            if book_yes24_review:
                for idx, value in enumerate(book_yes24_review):
                        if value == '\u2023':
                            book_yes24_review = book_yes24_review[:idx] + book_yes24_review[idx + 1:]
            if book_yes24_review == '':
                book_yes24_review = False
            book_pub_review = pub_review(driver)
            if book_pub_review:
                for idx, value in enumerate(book_pub_review):
                        if value == '\u2023':
                            book_pub_review = book_pub_review[:idx] + book_pub_review[idx + 1:]
            if book_pub_review == '':
                book_pub_review = False
            book_recommend = recommend(driver)
            if book_recommend:
                for idx, value in enumerate(book_recommend):
                        if value == '\u2023':
                            book_recommend = book_recommend[:idx] + book_recommend[idx + 1:]
            if book_recommend == '':
                book_recommend = False
            book_list = [0 for i in range(5)]
            book_list[0], book_list[1], book_list[2],\
            book_list[3], book_list[4]= \
            book_hash_tag, book_book_introduction, book_yes24_review, \
            book_pub_review, book_recommend
            if not (book_list[1] == 0 and book_list[2] == 0 and book_list[3] == 0 and book_list[4] == 0):
                df_series = pd.Series(book_list, index=book_attribute, name=book_title)
                try:
                    df_stitched = pd.concat([df_stitched, df_series], axis = 1)
                except:
                    df_stitched = df_series.to_frame(name=book_title)
        except:
            pass
    return df_stitched

def error(df):
    book_title = list(df.columns)
    book_attribute = list(df.index)
    for title in book_title: # 모든 책
        for attribute in book_attribute[1:]: # hash tag 제외
            temp = df[title][attribute]
            if type(temp) == str:
                for idx, value in enumerate(temp):
                    if value == '\u2023':
                        df[title][attribute] = temp[:idx] + temp[idx + 1:]
    return df