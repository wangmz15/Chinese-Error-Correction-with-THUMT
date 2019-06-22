#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import json
import requests
import urllib

# 文件翻译api
def file_translation(input_path, output_path, port, ip='127.0.0.1'):
    with open(input_path, 'r') as f:
        sources = f.readlines()
    sources = [line.strip() for line in sources]
    targets = translate_by_api(sources, port, ip)
    targets = [(line.encode('utf-8')+'\n') for line in targets]
    with open(output_path, 'w') as f:
        f.writelines(targets)

    return 'translated %d lines successful' % len(targets)


# 句子列表翻译api
def translate_by_api(sent_list, port, ip='127.0.0.1'):

    json_str = json.dumps({'sent_list': sent_list})
    quote_str = urllib.quote_plus(json_str)
    quote_str = urllib.quote_plus(quote_str)

    url = 'http://'+ip.strip()+':'+str(port)+'/translate_batch_api/'+quote_str

    r = requests.get(url)

    if r.status_code != 200:
        #print(sent_list)
        #print(json_str)
        #print(quote_str)
        raise Exception("fail to request")
        return None

    json_data = json.loads(r.text)
    return json_data['result_list']

def seg_sentence(sentence,language):
    idx = []
    if language == "zh" or language == "ja":
        punc = ["，",',']
    else:
        punc = [","]
    words = sentence.split(" ")
    pos = 0
    num_punc = 0
    num = 0
    idx_end = 0
    while pos < len(words):
        num += 1
        if words[pos] in punc:
            num_punc += 1
            if num_punc == 3 or num >= 30:
                words[pos] = "。"
                num_punc = 0
                num = 0
                idx.append(idx_end)
                idx_end += 1
        elif words[pos] == "。":
            num_punc = 0
            num = 0
            idx_end += 1
        else:
            pass
        pos += 1
    sentence = " ".join(words)
    return sentence,idx

# 切分子句，子句为一个单独翻译句
def split_sentence(sentence, language):
    if language == 'zh':
        punc = ['。','！','？','；']
    elif language == 'ja':
        punc = ['。']
    else:
        punc = ['.','!','?',';']
    de_month = ["Januar","Februar","März","April","Mai","Juni","Juli","August","September","Oktober","November","Dezember"]
    words = sentence.split(' ')
    pos = 0
    start = pos
    sentence_list = []
    quote_odevity = 0
    while pos < len(words):
        if words[pos] == '&quot;':
            quote_odevity = 1 - quote_odevity
        if words[pos] in punc:
            if language == 'de' and words[pos] == '.' and pos-1>0 and words[pos-1].isdigit() and pos+1<len(words) and words[pos+1] in de_month:
                pos += 1
            elif pos+1 < len(words) and words[pos+1] == '”':
                sentence_list.append(' '.join(words[start:pos+2]))
                pos += 2
                start = pos
            elif pos+1 < len(words) and words[pos+1] == '&quot;' and quote_odevity == 1:
                sentence_list.append(' '.join(words[start:pos+2]))
                pos += 2
                start = pos
            else:
                sentence_list.append(' '.join(words[start:pos+1]))
                pos += 1
                start = pos
        else:
            pos += 1
    if start < pos:
        sentence_list.append(' '.join(words[start:pos]))
                
    return sentence_list

# de-seg : 去除中文字符之间的空格，保留两个英文单词之间的空格
def deseg(sentence):
    #def judge_en_word(word):
    #    return all(ord(c) < 128 for c in word)  
    senlist = sentence.split()
    result = senlist[0]
    for i in range(1, len(senlist)):
        if senlist[i-1][-1].isalpha() and senlist[i][0].isalpha():
            result += ' ' + senlist[i]
        else:
            result += senlist[i]
    return result

# 修正中英文标点
def normPunc(sentence, language):
    punc_en = ['!','?',';'] # '.',
    punc_zh = ['！','？','；'] # '。',
    if language == 'zh':
        for i in range(len(punc_en)):
            print(sentence)
            sentence = sentence.replace(punc_en[i], punc_zh[i])
        for punc_list in [['\'','‘','’'],['\"','“','”']]:
            new_sent = ''
            j = 0
            for i in range(len(sentence)):
                if sentence[i] == punc_list[0]:
                    if j == 0:
                        new_sent += punc_list[1]
                    else:
                        new_sent += punc_list[2]
                    j = 1-j
                else:
                    new_sent += sentence[i]
            sentence = new_sent
    else :
        for i in range(len(punc_zh)):
            sentence = sentence.replace(punc_zh[i], punc_en[i])
        sentence = sentence.replace('“', '"')
        sentence = sentence.replace('”', '"')
        sentence = sentence.replace('‘', '\'')
        sentence = sentence.replace('’', '\'')
        sentence = sentence.replace('。', '.')

    return sentence

# 修正中英文标点，去除重复的标点，加上漏掉的标点
def validity_check(sentence, language):
    sentence = normPunc(sentence, language)
    senlist = sentence.split()
    if len(senlist) == 0:
        return sentence

    # step 1
    bug_puncs = ['.', '-', '"', "'", "?", ',', '!', '_', ';', ':']

    s = 0
    while s < len(senlist)-1:
        if senlist[s] in bug_puncs:
            e = s
            while(e + 1 < len(senlist) and senlist[e+1] == senlist[s]):
                e += 1
            del senlist[s+1:e+1]
            s += 1
        else:
            s += 1

    # step 2
    punc_en = ['.','!','?',';']
    punc_zh = ['。','！','？','；']
    
    if language == 'zh':
        if senlist[-1] not in punc_zh and len(senlist)>=2 and senlist[-2] not in punc_zh:
            senlist += ['。']
    else:
        if senlist[-1] not in punc_en and len(senlist)>=2 and senlist[-2] not in punc_en:
            senlist += ['.']
    return ' '.join(senlist)

def lower_first(content):
    items = content.split()
    items[0] = items[0].lower()
    res = " ".join(items)
    return res

# norm-char : 去除行首和行尾空格（python3也会去除全角空格）-> A3区全解转半角
def normChar(istring):
    rstring = ""
    for uchar in istring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if inside_code < 0x0020 or inside_code > 0x7e:      #转完之后不是半角字符返回原来的字符
            rstring += uchar
        else:
            rstring += chr(inside_code)
    rstring = re.sub(r'\s+', ' ', rstring)
    return rstring
