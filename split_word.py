import os, sys, random, re
import thulac
from tqdm import tqdm
import math, time, threading
import multiprocessing as mp
from multiprocessing import Pool, Process
from bs4 import BeautifulSoup
# import synonyms
import numpy as np

data_root = '/data/disk1/private/wangmuzi'


def gen_vocab_lang8(dirs, vocab_out='', current_vocab=''):
    thu1 = thulac.thulac(T2S=True, seg_only=True)  # 默认模式
    dic = {}
    if current_vocab:
        with open(current_vocab) as fv:
            for line in fv:
                dic[line.split()[0]] = int(line.split()[1])
            fv.close()
    cnt = 0
    for dir in dirs:
        with open(dir + '.src.txt') as fs, open(dir + '.trg.txt') as ft:
            doc_num = fs.readlines()
            for line_src in tqdm.tqdm(doc_num):
                # cnt += 1
                # if cnt > 10:
                #     break
                line_trg = ft.readline()
                texts = [k[0] for k in thu1.cut(line_src, text=False)]
                texts.extend([k[0] for k in thu1.cut(line_trg, text=False)])
                for t in set(texts):
                    if t in dic:
                        dic[t] += 1
                    else:
                        dic[t] = 1
    with open(vocab_out, 'w') as fvw:
        for k in dic:
            if k.strip():
                fvw.write(k + ' ' + str(dic[k]) + '\n')


def gen_vocab(dirs, vocab_out='', okdir='', current_vocab=''):
    thu1 = thulac.thulac(T2S=True, seg_only=True)  # 默认模式
    dic = {}
    if current_vocab:
        with open(current_vocab) as fv:
            for line in fv:
                try:
                    dic[line.split()[0]] = int(line.split()[1])
                except:
                    print(line)
            fv.close()
    # cnt1 = 0
    ok_file = []

    with open(okdir, 'a+') as f_okdir:
        f_okdir.seek(0)
        for l in f_okdir:
            ok_file.append(l)
        f_okdir.close()
    fw_okdir = open(okdir, 'a')
    # print(ok_file)
    for dir in tqdm(dirs):
        if dir in ok_file or '1066066' in dir:
            continue
        with open(dir) as f:
            for line in f:
                try:
                    texts = [k[0] for k in thu1.cut(line, text=False)]
                except IndexError:
                    print(dir)
                    # print(thu1.cut(line, text=False))
                    with open(vocab_out, 'w') as fvw:
                        for k in dic:
                            if k.strip():
                                fvw.write(k + ' ' + str(dic[k]) + '\n')
                    exit(0)
                for t in set(texts):
                    if t in dic:
                        dic[t] += 1
                    else:
                        dic[t] = 1
            fw_okdir.write(dir + '\n')
    with open(vocab_out, 'w') as fvw:
        for k in dic:
            if k.strip():
                fvw.write(k + ' ' + str(dic[k]) + '\n')
        fvw.close()


def listFilename(path1):
    results = []
    fileList = os.listdir(path1)
    for file in fileList:
        mypath = path1 + '/' + file
        if os.path.isdir(mypath):
            results.extend(listFilename(mypath)) # 递归
        else:
            results.append(mypath)
            # print(mypath)
    return results

LARGEST_PROPOTION = 10.0
CHARACTOR_LEAST_APPEARENCE = 5
WORD_LEAST_APPEARENCE = 3


def sort_words(dir):
    f = open(dir)
    dic = {}
    for line in f:
        word = ''.join(line.strip().split()[:-1])
        cnt = line.strip().split()[-1]
        if word == '' or cnt == '':
            continue
        cnt = int(cnt)
        dic[word] = cnt
    dic = sorted(dic.items(), key=lambda item: item[1], reverse=True)
    fw = open(dir + '_sorted', 'w')
    fc = open(dir + '_c', 'w')
    fword = open(dir + '_w', 'w')
    dic_w = {}
    dic_c = {}
    for k, v in dic:
        fw.write(k + ' ' + str(v) + '\n')
        if len(k) == 1:
            dic_c[k] = v
        else:
            dic_w[k] = v
    for (k, v) in dic_c.items():
        appear_in_word = 0
        for (k1, v1) in dic_w.items():
            if k in k1:
                appear_in_word += v1
        percent = float(appear_in_word / v)
        dic_c[k] = [v, appear_in_word, percent]
    for (k, v) in dic_w.items():
        if v < WORD_LEAST_APPEARENCE:
            continue
        fword.write(k + ' ' + str(v) + '\n')
    dic_c = sorted(dic_c.items(), key=lambda item: item[1][2], reverse=True)
    for (k, [v, a, p]) in dic_c:
        if p > LARGEST_PROPOTION or v < CHARACTOR_LEAST_APPEARENCE:
            continue
        fc.write(k + ' ' + str(v) + ' ' + str(a) + ' ' + str(p) + '\n')

def merge_words(in_dir, out_dir, keyword):
    dirs = listFilename(in_dir)
    total_dic = {}
    for dir in tqdm(dirs):
        if keyword not in dir:
            continue
        for line in open(dir):
            line.strip()
            token = ''.join(line.split()[:-1])
            cnt = int(line.split()[-1])
            if token not in total_dic:
                total_dic[token] = cnt
            else:
                total_dic[token] += cnt
    print(len(total_dic))
    with open(out_dir, 'w') as fw:
        for k,v in total_dic.items():
            fw.write(k +' '+str(v) + '\n')


def cut_sent(para):
    para = re.sub('([。\t！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

def get_tokenized_files(dirs, out, okdir=''):
    thu1 = thulac.thulac(T2S=True, seg_only=True)  # 默认模式
    seg_sents = []
    # fok = open(okdir, 'a')
    for dir in tqdm(dirs):
        with open(dir) as f:
            for para in f:
                para = para.strip()
                if not para:
                    continue
                for sent in cut_sent(para):
                    sent = sent.replace('\t', '').replace('\n', '').replace(' ','').strip()
                    if sent:
                        seg_sents.append(thu1.cut(sent, text=True).strip())
        # fok.write(dir)
    fw = open(out, 'w')
    for seg_sent in seg_sents:
        fw.write(seg_sent+'\n')

def merge_tokenized_files(dirs, outdir):
    fw = open(outdir, 'w')
    lines = []
    for dir in tqdm(dirs):
        with open(dir) as f:
            for line in f:
                # line = line.replace('\t', '').replace('\n', '').strip()
                # if not line:
                #     continue
                lines.append(line)
    for line in lines:
        fw.write(line)

def merge_tokenized_files_rmrb(dirs, outdir):
    fws = open(outdir+'.src', 'w')
    fwt = open(outdir+'.trg', 'w')
    src = []
    trg = []
    for i in range(11):
        for dir in tqdm(dirs):
            if '.seg'+str(i) in dir and 'trg' in dir:
                with open(dir) as f:
                    for line in f:
                        src.append(line)
            if '.seg'+str(i) in dir and 'trg' not in dir:
                with open(dir) as f:
                    for line in f:
                        trg.append(line)
    for line_s, line_t in zip(src, trg):
        fws.write(line_s)
        fwt.write(line_t)

def get_tokenized_files_lang8(dir, out):
    thu1 = thulac.thulac(T2S=True, seg_only=True)  # 默认模式
    src_sents = []
    trg_sents = []
    # fok = open(okdir, 'a')
    with open(dir+'.src') as fsrc, open(dir+'.trg') as ftrg:
        for src,trg in tqdm(zip(fsrc,ftrg)):
            src = src.replace('\t', '').replace('\n', '').replace(' ', '').strip()
            trg = trg.replace('\t', '').replace('\n', '').replace(' ', '').strip()
            if src and trg:
                src_sents.append(thu1.cut(src, text=True))
                trg_sents.append(thu1.cut(trg, text=True))
        # fok.write(dir)
    fwsrc = open(out+dir.split('/')[-1]+'.src.seg', 'w')
    fwtrg = open(out+dir.split('/')[-1]+'.trg.seg', 'w')
    for src,trg in zip(src_sents,trg_sents):
        fwsrc.write(src+'\n')
        fwtrg.write(trg+'\n')


def get_tokenized_files_cged(dirs, out):
    thu1 = thulac.thulac(T2S=True, seg_only=True)  # 默认模式
    src_sents = []
    trg_sents = []
    # fok = open(okdir, 'a')
    for dir in dirs:
        all_content = ''.join(open(dir).readlines())
        soup = BeautifulSoup(all_content, "html.parser")
        docs = soup.find_all('doc')
        for doc in tqdm(docs):
            text = doc.find('text')
            correction = doc.find('correction')
            if text and correction:
                text = text.text.strip().replace('\n', '')
                correction = correction.text.strip().replace('\n', '')
                src_sents.append(thu1.cut(text, text=True))
                trg_sents.append(thu1.cut(correction, text=True))
    fwsrc = open(out+'/'+out.split('/')[-1]+'.src', 'w')
    fwtrg = open(out+'/'+out.split('/')[-1]+'.trg', 'w')
    for src,trg in zip(src_sents,trg_sents):
        fwsrc.write(src+'\n')
        fwtrg.write(trg+'\n')

def get_valid_files_cged(src_dir):
    all = []
    train = []
    valid = []
    with open(src_dir+'.src') as src, open(src_dir+'.trg') as trg, \
            open(src_dir + '_train.src', 'w') as train_src, open(src_dir + '_train.trg', 'w') as train_trg, \
            open(src_dir + '_valid.src', 'w') as valid_src, open(src_dir + '_valid.trg', 'w') as valid_trg:
        for src_line, trg_line in zip(src, trg):
           all.append((src_line,trg_line))

        random.shuffle(all)
        for i in range(int(0.1*len(all))):
            valid = random.choice(all)
            valid_src.write(valid[0])
            valid_trg.write(valid[1])
            all.remove(valid)
        for pair in all:
            train_src.write(pair[0])
            train_trg.write(pair[1])


def get_valid_files_rmrb(src_dir):
    all = []
    train = []
    valid = []
    with open(src_dir+'.src') as src, open(src_dir+'.trg') as trg, \
            open(src_dir[:-4] + 'train.src', 'w') as train_src, open(src_dir[:-4] + 'train.trg', 'w') as train_trg, \
            open(src_dir[:-4] + 'valid.src', 'w') as valid_src, open(src_dir[:-4] + 'valid.trg', 'w') as valid_trg:
        for src_line, trg_line in tqdm(zip(src.readlines(), trg.readlines())):
           all.append((src_line,trg_line))

        random.shuffle(all)
        for i in range(int(0.001*len(all))):
            valid = random.choice(all)
            valid_src.write(valid[0])
            valid_trg.write(valid[1])
            all.remove(valid)
        for pair in tqdm(all):
            train_src.write(pair[0])
            train_trg.write(pair[1])


def generate_paral_text(directions):
    """

    :type directions: list
    """
    four_oper = ['s', 'r', 'w', 'm', 'keep']
    np.random.seed(0)
    p = np.array([0.4*0.8, 0.2*0.8, 0.1*0.8, 0.3*0.8, 0.2])
    print(directions,len(directions))
    for dir in directions:
        with open(dir) as f, open(dir+'.trg', 'w') as fw:
            for line in tqdm(f.readlines()):
                words = line.split()
                cnt = 0
                i = 0
                mistake_cnt = max(int(len(words) * 0.1), 1)
                for oper in [np.random.choice(four_oper, p=p.ravel()), np.random.choice(four_oper, p=p.ravel())]:
                    if oper == 'keep' or len(words) == 0:
                        continue
                    elif oper == 's':
                        while i in range(mistake_cnt):
                            if cnt > 10:
                                break
                            word_idx = random.choice(range(len(words)))
                            candidates = synonyms.nearby(words[word_idx])[0][1:9]
                            if words[word_idx] in ['。','，','、','《','》','？','；','’','‘','：','“','”','【','】','{','}','、','|','！','@','#','￥','%','……','&','*'] \
                                    or len(candidates) == 0:
                                cnt += 1
                                continue
                            # print(words[word_idx],candidates)
                            words[word_idx] = random.choice(candidates) #  随机替换掉0.1次近义词
                            i += 1
                    elif oper == 'r':
                        while i in range(mistake_cnt):
                            if cnt > 10:
                                break
                            word_idx = random.choice(range(len(words)))
                            candidates = synonyms.nearby(words[word_idx])[0][1:9]
                            if words[word_idx] in ['。', '，', '、', '《', '》', '？', '；', '’', '‘', '：', '“', '”', '【', '】', '{', '}',
                                                   '、', '|', '！', '@', '#', '￥', '%', '……', '&', '*'] \
                                    or len(candidates) == 0:
                                cnt += 1
                                continue
                            insert_pos = max(word_idx + random.choice([0, 1]), 0) # 随机插入一个近义词在前或者在后
                            words.insert(insert_pos, random.choice(candidates))
                            i += 1
                    elif oper == 'w':
                        word_idx = random.choice(range(len(words)))
                        word_to_exchange = words[word_idx]
                        exchange_pos = min(max(word_idx + random.choice([-1, 1, -2, 2]), 0), len(words)-1)  # 随机在前或者在后替换一个词
                        words[word_idx] = words[exchange_pos]
                        words[exchange_pos] = word_to_exchange
                    else:
                        while i in range(mistake_cnt):
                            if cnt > 10:
                                break
                            word_idx = random.choice(range(len(words)))
                            if words[word_idx] in ['。','，','、','《','》','？','；','’','‘','：','“','”','【','】','{','}','、','|','！','@','#','￥','%','……','&','*']:
                                cnt += 1
                                continue
                            del words[word_idx:word_idx+1] # 随机删掉一个非标点符号
                            i += 1
                new_line = ' '.join(words)
                fw.write(new_line+'\n')


if __name__ == "__main__":
    # ############################# cged #######################################

    # get_tokenized_files_cged([data_root+'/raw_data/cged16/Training/tocfl'], data_root+'/data/THUMT/data/cged/tocfl')
    # get_tokenized_files_cged([data_root+'/raw_data/cged16/Training/CGED16_HSK_TrainingSet.txt'], data_root+'/data/THUMT/data/cged/hsk')
    # merge_tokenized_files([data_root+'/data/THUMT/data/cged/tocfl/tocfl.src',data_root+'/data/THUMT/data/cged/hsk/hsk.src'], data_root+'/data/THUMT/data/cged/cged.src')
    # merge_tokenized_files([data_root+'/data/THUMT/data/cged/tocfl/tocfl.trg',data_root+'/data/THUMT/data/cged/hsk/hsk.trg'], data_root+'/data/THUMT/data/cged/cged.trg')
    # get_valid_files_cged(data_root+'/data/THUMT/data/cged/cged')


    # ############################# Lang8 #######################################

    # gen_vocab(vocab_out=data_root+'/data/OpenNMT-py/dic/rmrb.vocab1', dirs=res, current_vocab='', okdir=data_root+'/data/OpenNMT-py/dic/ok.txt')
    # for i in listFilename(data_root+'/data/lang8'):
    #     print(i)
    # get_tokenized_files_lang8(data_root+'/data/lang8/lang8_train', data_root+'/data/THUMT/data/lang8_seg/')
    # get_tokenized_files_lang8(data_root+'/data/lang8/lang8_valid', data_root+'/data/THUMT/data/lang8_seg/')

    # ############################# RMRB #######################################
    # res = listFilename(data_root + '/raw_data/rmrb1994-2003')
    # pro_num = 10
    # batch_size = int(len(res) / pro_num)
    # print(len(res), pro_num, batch_size)
    # r = [res[i * batch_size:(i + 1) * batch_size] for i in range(pro_num)]
    # r.append(res[pro_num * batch_size:])
    # #
    # if len(res) % pro_num:
    #     pro_num += 1
    #
    # for i in range(pro_num):
    #     if i != 7:
    #         continue
    #     t = threading.Thread(target=gen_vocab, args=(r[i], data_root+'/data/OpenNMT-py/dic/rmrb/rmrb.vocab{}_plus'.format(str(i)),
    #                                                  data_root+'/data/OpenNMT-py/dic/rmrb/ok{}'.format(str(i))))
    #     t.start()


    # merge_words(data_root+'/data/OpenNMT-py/dic/rmrb', data_root+'/data/OpenNMT-py/dic/rmrb/rmrb', 'vocab')
    # sort_words(data_root+'/data/OpenNMT-py/dic/rmrb/rmrb')


    # res = listFilename(data_root + '/raw_data/rmrb1994-2003')
    # pro_num = 10
    # batch_size = int(len(res) / pro_num)
    # print(len(res), pro_num, batch_size)
    # r = [res[i * batch_size:(i + 1) * batch_size] for i in range(pro_num)]
    # r.append(res[pro_num * batch_size:])
    # #
    # for i,dirs in enumerate(r):
    #     if 9 <= i:
    #         t = threading.Thread(target=get_tokenized_files, args=(dirs, data_root+'/data/THUMT/data/rmrb_seg/rmrb.seg{}'.format(str(i)),
    #                                                                data_root+'/data/THUMT/data/rmrb_seg/ok{}'.format(str(i))))
    #         t.start()

    # merge_tokenized_files(listFilename(data_root+'/data/THUMT/data/rmrb/rmrb_seg/'), data_root+'/data/THUMT/data/rmrb/rmrb')

    # res = listFilename(data_root + '/data/THUMT/data/rmrb/rmrb_seg')
    # pro_num = 10
    # batch_size = int(len(res) / pro_num)
    # print(len(res), pro_num, batch_size)
    # r = [res[i * batch_size:(i + 1) * batch_size] for i in range(pro_num)]
    # r.append(res[pro_num * batch_size:])
    # # print(r)
    # for i,dirs in enumerate(r):
    #     if 9 <= i and i < 11:
    #         t = threading.Thread(target=generate_paral_text, args=[dirs])
    #         t.start()

    # merge_tokenized_files_rmrb(listFilename(data_root+'/data/THUMT/data/rmrb/rmrb_seg/'), data_root+'/data/THUMT/data/rmrb/rmrb')


    # get_valid_files_rmrb(data_root + '/data/THUMT/data/rmrb/rmrb')
    # get_valid_files_rmrb(data_root + '/data/THUMT/data/lang8/lang8')


    ##############################################################
    # 多进程 不管用
    # for i in range(pro_num):
    #     proc = Process(target=gen_vocab, args=(r[i], 'rmrb.vocab{}'.format(str(i)), 'ok{}'.format(str(i))))
    #     proc.start()
    #     proc.join()

    # process = [mp.Process(target=gen_vocab, args=(dirs, 'rmrb.vocab{}'.format(str(i)), '', 'ok{}'.format(str(i)))) for i, dirs in enumerate(r)]
    # for p in process:
    #     p.start()
    # for p in process:
    #     p.join()

    # p = Pool(pro_num)
    # a = []
    # for i in range(pro_num):
    #     p.apply_async(gen_vocab, args=(r[i], 'rmrb.vocab{}'.format(str(i)), 'ok{}'.format(str(i))))
    #     print(str(i) + ' processor started !')
    # p.close()
    # p.join()
