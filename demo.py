#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, render_template, request, redirect, send_from_directory
from flask_cors import CORS
import codecs
import os
import re
import operator
import urllib
import random
import sys
import logging

import argparse
import itertools

import numpy as np
import time
import json
import tensorflow as tf
import thumt.data.dataset as dataset
import thumt.data.vocab as vocabulary
import thumt.models as models
import thumt.utils.parallel as parallel
import thumt.utils.inference as inference
from utils import *  # common function
from resource_mt.apply_bpe import BPE, read_vocabulary
from resource_mt.moses import MosesTokenizer, MosesDetokenizer
import thulac
#import enchant
import nltk

thu = thulac.thulac(seg_only=True,T2S=True)

def merge_dict(_sum, _one):
    for key, value in _one.items():
        if _sum.has_key(key):
            _sum[key] += value
        else:
            _sum[key] = value
    return _sum


def gettext(vocab, idx_list, idx_eos):
    result = ''
    for idx in idx_list:
        if idx == idx_eos:
            break
        result += vocab[idx] + ' '
    return result.strip()


def getid(ivocab, text):
    words = text.split(' ')
    result = []
    for word in words:
        if not word in ivocab:
            result.append(ivocab['<unk>'])
        else:
            result.append(ivocab[word])
    result.append(ivocab['<eos>'])
    return result


def build_ivocab(vocab):
    result = {}
    for num, word in enumerate(vocab):
        result[word] = num
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translation demo server",
        usage="demo.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Path of trained models")
    parser.add_argument("--vocabulary", type=str, nargs=2, required=True,
                        help="Path of source and target vocabulary")
    parser.add_argument("--port", type=int, required=True,
                        help="Port to launch")
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Maximum batch size to translate in one iter")

    # model and configuration
    parser.add_argument("--models", type=str, required=True, nargs="+",
                        help="Name of the model")
    parser.add_argument("--parameters", type=str,
                        help="Additional hyper parameters")

    # bpe init
    parser.add_argument(
        '--pair', type=argparse.FileType('r'), metavar='PATH',
        required=True,
        help="File with BPE codes (created by learn_bpe.py).")
    parser.add_argument(
        '--merges', type=int, default=-1,
        metavar='INT',
        help="Use this many BPE operations (<= number of learned symbols)" +
             "default: Apply all the learned merge operations")
    parser.add_argument(
        '--separator', type=str, default='##', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument(
        '--glossaries', type=str, nargs='+', default=None,
        metavar="STR",
        help="Glossaries. The strings provided in glossaries will not be affected" +
             "by the BPE (i.e. they will neither be broken into subwords, nor concatenated with other subwords")

    return parser.parse_args()


def default_parameters():
    params = tf.contrib.training.HParams(
        input=None,
        output=None,
        vocabulary=None,
        model=None,
        # vocabulary specific
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        mapping=None,
        append_eos=False,
        # decoding
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_beta=0.2,
        decode_length=80,
        decode_batch_size=32,
        decode_constant=5.0,
        decode_normalize=False,
        device_list=[0],
        num_threads=1
    )

    return params


def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

    for (k, v) in params1.values().items():
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in params2.values().items():
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def override_parameters(params, args):
    if args.parameters:
        params.parse(args.parameters)

    params.vocabulary = {
        "source": vocabulary.load_vocabulary(args.vocabulary[0]),
        "target": vocabulary.load_vocabulary(args.vocabulary[1])
    }
    params.vocabulary["source"] = vocabulary.process_vocabulary(
        params.vocabulary["source"], params
    )
    params.vocabulary["target"] = vocabulary.process_vocabulary(
        params.vocabulary["target"], params
    )

    control_symbols = [params.pad, params.bos, params.eos, params.unk]

    params.mapping = {
        "source": vocabulary.get_control_mapping(
            params.vocabulary["source"],
            control_symbols
        ),
        "target": vocabulary.get_control_mapping(
            params.vocabulary["target"],
            control_symbols
        )
    }

    return params


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=False)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    config.gpu_options.allow_growth = True
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config


def set_variables(var_list, value_dict, prefix):
    print("6")
    ops = []
    for var in var_list:
        for name in value_dict:
            var_name = "/".join([prefix] + list(name.split("/")[1:]))
            if var.name[:-2] == var_name:
                tf.logging.info("restoring %s -> %s" % (name, var.name))
                with tf.device("/cpu:0"):
                    op = tf.assign(var, value_dict[name])
                    ops.append(op)
                break
                print("8")

    return ops


def shard_features(features, placeholders, predictions):
    num_shards = len(placeholders)
    feed_dict = {}
    n = 0
    for name in features:
        feat = features[name]
        batch = feat.shape[0]
        shard_size = (batch + num_shards - 1) // num_shards

        for i in range(num_shards):
            shard_feat = feat[i * shard_size:(i + 1) * shard_size]
            if shard_feat.shape[0] != 0:
                feed_dict[placeholders[i][name]] = shard_feat
                n = i + 1
            else:
                break


    if isinstance(predictions, (list, tuple)):
        predictions = [item[:n] for item in predictions]

    return predictions, feed_dict


# main start here
if True:
    args = parse_args()
    tf.logging.set_verbosity(tf.logging.INFO)
    # Load configs
    model_cls_list = [models.get_model(model) for model in args.models]
    params_list = [default_parameters() for _ in range(len(model_cls_list))]
    params_list = [
        merge_parameters(params, model_cls.get_parameters())
        for params, model_cls in zip(params_list, model_cls_list)
    ]
    params_list = [
        import_params(args.checkpoints[i], args.models[i], params_list[i])
        for i in range(len(args.checkpoints))
    ]
    params_list = [
        override_parameters(params_list[i], args)
        for i in range(len(model_cls_list))
    ]

    # Build Graph
    with tf.Graph().as_default():
        model_var_lists = []

        # Load checkpoints
        for i, checkpoint in enumerate(args.checkpoints):
            print("Loading %s" % checkpoint)
            var_list = tf.train.list_variables(checkpoint)
            values = {}
            reader = tf.train.load_checkpoint(checkpoint)

            for (name, shape) in var_list:
                if not name.startswith(model_cls_list[i].get_name()):
                    continue

                if name.find("losses_avg") >= 0:
                    continue

                tensor = reader.get_tensor(name)
                values[name] = tensor

            model_var_lists.append(values)
        # Build models
        model_list = []

        for i in range(len(args.checkpoints)):
            name = model_cls_list[i].get_name()
            model = model_cls_list[i](params_list[i], name + "_%d" % i)
            model_list.append(model)

        params = params_list[0]

        # Create placeholders
        placeholders = []
        for i in range(len(params.device_list)):
            placeholders.append({
                "source": tf.placeholder(tf.int32, [None, None],
                                         "source_%d" % i),
                "source_length": tf.placeholder(tf.int32, [None],
                                                "source_length_%d" % i)
            })

        # Create parallel predictions
        inference_fn = inference.create_inference_graph
        predictions = parallel.data_parallelism(
            params.device_list, lambda f: inference_fn(model_list, f, params),
            placeholders)

        # Create assign ops
        assign_ops = []

        all_var_list = tf.trainable_variables()

        for i in range(len(args.checkpoints)):
            un_init_var_list = []
            name = model_cls_list[i].get_name()
            print("1")
            for v in all_var_list:
                if v.name.startswith(name + "_%d" % i):
                    un_init_var_list.append(v)
            print("2")
            ops = set_variables(un_init_var_list, model_var_lists[i],
                                name + "_%d" % i)
            print("3")
            assign_ops.extend(ops)
        assign_op = tf.group(*assign_ops)

        sess_creator = tf.train.ChiefSessionCreator(
            config=session_config(params)
        )

        # Create vocab dictionary
        vocab_en = params.vocabulary["target"]
        ivocab_zh = build_ivocab(params.vocabulary["source"])  ## type == <type 'str'>
        # print(len(vocab_en), len(ivocab_zh))

        # Create session
        sess = tf.train.MonitoredSession(session_creator=sess_creator)
        sess.run(assign_op)
        print('run tf_session success.')

        # bpe
        args.pair = codecs.open(args.pair.name, encoding='utf-8')
        bpe_ins = BPE(args.pair, args.merges, args.separator, None, args.glossaries)
        print('run bpe init success.')

        # tokenizer and detokenizer
        tokenizer = MosesTokenizer('zh')
        detokenizer = MosesDetokenizer('zh')


        # flask
        app = Flask(__name__)
        CORS(app)
        print('run flask success.')



# batched translate
def tf_translate(sen_ids_list):
    num_sent = len(sen_ids_list)
    max_len = max(map(len, sen_ids_list))

    padded_input = np.ones([num_sent, max_len], dtype=np.int32) * ivocab_zh['<pad>']
    for i in range(num_sent):
        padded_input[i][:len(sen_ids_list[i])] = sen_ids_list[i]

    features = {}
    features["source"] = padded_input
    features["source"] = np.array(features["source"])
    features["source_length"] = [len(sen_ids) for sen_ids in sen_ids_list]
    features["source_length"] = np.array(features["source_length"])

    op, feed_dict = shard_features(features, placeholders, predictions)
    results, _ = sess.run(op, feed_dict=feed_dict)  # results -> (n_shards, batch_size, top_beams, length)
    results = list(itertools.chain(*results))  # results -> (n, top_beams, length)
    results = [x[0] for x in results]  # results -> (n, length)

    trg_list = [gettext(vocab_en, x, params.mapping["target"][params.eos]) for x in results]

    return trg_list


def rbpe(sentence):
    result = sentence.replace('@@ ', '')
    return result

'''def spelling_check(sentence):
    d = enchant.Dict("en_US")
    check_tokens = list()
    tokens = sentence.strip().split()
    print(tokens)
    for w in tokens:
        if d.check(w) or not w.isalpha():
            check_tokens.append(w)
        else:
            suggestions = d.suggest(w)
            if (len(suggestions) != 0):
                check_tokens.append(suggestions[0])
            else:
                check_tokens.append(w)
    return " ".join(check_tokens)'''

def bpe(sentence):
    # sentence = sentence.decode('utf-8')
    result = bpe_ins.segment(sentence.strip())
    # result = result.encode('utf-8')
    return result


def tokenize(sentence):
    sentence = sentence.decode('utf-8')
    result = tokenizer.tokenize(sentence.strip(), return_str=True)
    result = result.encode('utf-8')
    return result


def split_char(sentence):
    sentence = sentence.decode("utf-8")
    res = ""
    for char in sentence:
        res = res + " " + char
    res = res.strip()
    res = res.encode("utf-8")
    return res


def detokenize(sentence):
    # sentence = sentence.decode('utf-8')
    result = detokenizer.detokenize(sentence.strip().split(), return_str=True)
    # result = result.encode('utf-8')
    return result


def print_list(op, sent_list):
    sys.stdout.write(op)
    for sen_i in sent_list:
        sys.stdout.write(sen_i + ' // ')
    sys.stdout.write('\n')


def preprocess(sentence):
    # norm_punctuation
    sentence = normPunc(sentence, 'zh')
    print('normPunc: ', sentence)
    # spelling check
    #sentence = spelling_check(sentence)`
    #print('spelling: ', sentence)
    # tokenize
    # sentence = tokenize(sentence)
    # sentence = ' '.join(nltk.word_tokenize(sentence.strip()))
    sentence = thu.cut(sentence, text=True)
    print('tokenized: ', sentence)
    # split
    sentence_list = split_sentence(sentence, 'zh')
    print_list('split: ', sentence_list)
    # bpe
    for i in range(len(sentence_list)):
        sentence_list[i] = bpe(sentence_list[i])
    print_list('bpe: ', sentence_list)
    # map to id
    sen_ids_list = [getid(ivocab_zh, sentence) for sentence in sentence_list]
    return sen_ids_list


def postprocess(result_list, idx):
    # validity check
    for i in range(len(result_list)):
        result_list[i] = validity_check(result_list[i], 'zh')
    print_list('validity check: ', result_list)
    if len(idx) > 0:
        for i in range(len(result_list)):
            if i in idx:
                result_list[i] = result_list[i][:-1] + ","
                if i + 1 < len(result_list):
                    words = result_list[i + 1].split()
                    result_list[i + 1] = " ".join(words)
    print_list("after lower: ", result_list)

    # concat
    for i in range(len(result_list)):
        if i == 0 or (i > 0 and result_list[i - 1][-1] != ','):
            if len(result_list[i]) > 0:
                result_list[i] = result_list[i][0].upper() + result_list[i][1:]
    result = ' '.join(result_list)
    # rbpe
    result = rbpe(result)
    print('reverse bpe: ', result)
    # detok
    result = detokenize(result)
    result = result.replace('( ', '(').replace('<unk>', '').replace(' ', '')
    print('detok: ', result)
    return result


@app.route('/translateapi/<quote_str>', methods=['GET', 'POST'])
def translateapi(quote_str):
    if len(quote_str) > 1000:  # limit the quote_str length
        return json.dumps({'trg': 'sentence is toooo loooong!'})

    time_stamp_all = time.time()
    print('-' * 60)

    # quote_str = quote_str.encode('utf-8')  # transform unicode to str
    print('quote_str:', quote_str)

    sentence = urllib.parse.unquote(quote_str,encoding='utf-8')
    # sentence = quote_str
    print('source_sentence:', sentence)

    # pre_process
    idx = []

    sen_ids_list = preprocess(sentence)

    # translate
    time_stamp_trans = time.time()
    result_list = tf_translate(sen_ids_list)
    print_list('translated(%.2fs): ' % (time.time() - time_stamp_trans), result_list)

    # post_process
    result = postprocess(result_list, idx)

    print('target_sentence: ', result)
    print('total time = (%.2fs)' % (time.time() - time_stamp_all))
    result.replace('<unk>','')
    return json.dumps({'trg': result})


@app.route('/')
def demo():
    return render_template('index.html')


if __name__ == '__main__':
    app.debug = True
    app.run('0.0.0.0', port=args.port)

