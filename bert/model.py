import json
import os
from abc import ABC
from random import shuffle, random, choice, randrange
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dropout, Dense
import sentencepiece as spm
from tqdm import tqdm


def load_vocab():
    vocab_file = "../kowiki/kowiki.model"
    vocab = spm.SentencePieceProcessor()
    vocab.load(vocab_file)
    return vocab


vocab = load_vocab()


def get_attn_pad_mask(self, seq_q, seq_k, i_pad):
    # seq_q : (batch_size, len_q)
    # seq_k : (batch_size, len_k)
    batch_size = tf.shape(seq_q)[0]
    len_q = tf.shape(seq_q)[1]
    len_k = tf.shape(seq_k)[1]

    pad_attn_mask = tf.equal(seq_k, i_pad)

    # 텐서 반복확장
    pad_attn_mask = tf.broadcast_to(tf.expand_dims(pad_attn_mask, axis=1), [batch_size, len_q, len_k])
    # or
    # pad_attn_mask = tf.tile(tf.reshape(pad_attn_mask, (batch_size, 1, len_k)), multiples=(batch_size, len_q, len_k))
    return pad_attn_mask


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        assert embedding_dim % self.num_heads == 0

        self.projection_dim = embedding_dim // num_heads  # 하나의 헤드에 존재하는 차원수
        self.query_dense = tf.keras.layers.Dense(embedding_dim)
        self.key_dense = tf.keras.layers.Dense(embedding_dim)
        self.value_dense = tf.keras.layers.Dense(embedding_dim)
        self.dense = tf.keras.layers.Dense(embedding_dim)

    def scaled_dot_product_attention(self, query, key, value, mask):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)  # Head의 projection_dim
        logits = matmul_qk / tf.math.sqrt(depth)

        # 마스킹. 에턴션 스코어 행렬의 마스킹 할 위치에 매우 작은 음수값을 넣는다.
        # 매우 작은 값이므로 소프트맥스 함수를 지나면 행렬의 해당 위치의 값은 0이 됨.
        if mask is not None:
            logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(logits, axis=-1)

        # output 크기 : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]

        # (batch_size, seq_len, embedding_dim)
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # (batch_size, num_heads, seq_len, projection_dim)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, attn_prob = self.scaled_dot_product_attention(query, key, value, mask)

        # head 다시 concatenate
        # (batch_size, seq_len, num_heads, projection_dim)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len, embedding_dim)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embedding_dim))

        outputs = self.dense(concat_attention)
        return outputs, attn_prob


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.config = config

        self.att = MultiHeadAttention(self.config.d_model, self.config.num_heads)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(self.config.dff, activation="relu"),
             tf.keras.layers.Dense(self.config.embedding_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=self.config.epsilon)
        self.layernorm2 = LayerNormalization(epsilon=self.config.epsilon)
        self.dropout1 = Dropout(self.config.dropout)
        self.dropout2 = Dropout(self.config.dropout)

    def call(self, inputs, mask, training):
        attn_output, attn_prob = self.att(inputs, mask)  # 첫번째 서브층 : 멀티 헤드 어텐션
        # training 옵션 : 학습시에만 적용되고 추론시에는 적용되지 않도록 하는 옵션
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Add & norm
        ffn_output = self.ffn(out1)  # 두번째 서브층 : 포지션 와이즈 피드 포워드 신경망
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output), attn_prob  # Add & norm


class Encoder(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

        self.enc_emb = tf.keras.layers.Embedding(self.config.n_enc_vocab, self.config.embedding_dim)
        # n_enc_seq에 +1 하는 이유가??
        self.pos_emb = tf.keras.layers.Embedding(self.config.n_enc_seq + 1, self.config.embedding_dim)
        self.seg_emb = tf.keras.layers.Embedding(self.config.n_seg_type, self.config.embedding_dim)

    def call(self, inputs, segments):
        # 포지션 임베딩 inputs 생성 (크기는 inputs의 seq_length와 동일)
        positions = tf.keras.layers.Input(shape=(None, self.config.n_enc_seq), name="positions")
        # i_pad (패딩으로 간주되는 토큰의 정수) 와 inputs를 비교하여 패딩마스크 생성 -> 패딩토큰:True, 단어토큰:False 으로 구성된 패딩마스크 생성
        pos_mask = tf.equal(inputs, self.config.i_pad)
        # 패딩마스크를 positions 에 덮어씌우기. (패딩토큰이 아니라면 보존, 패딩토큰은 0으로 업데이트)
        positions = tf.where(pos_mask, 0, positions)

        # (batch_size, n_enc_seq, embedding_dim)
        outputs = self.enc_emb(inputs) + self.pos_emb(positions) + self.seg_emb(segments)

        # (batch_size, n_enc_seq, n_enc_seq)
        attn_mask = get_attn_pad_mask(inputs, inputs, self.config.i_pad)

        attn_probs = []

        for i in range(self.config.n_layer):
            # (batch_size, n_enc_seq, embedding_dim), (batch_size, num_heads, n_enc_seq, n_enc_seq)
            outputs, attn_prob = EncoderLayer(self.config)(outputs, attn_mask, training=True)
            attn_probs.append(attn_prob)

        return outputs, attn_probs


class BERT(tf.keras.layers.Layer):
    """
    bert
    """

    def __init__(self, config):
        super(BERT, self).__init__()
        self.config = config
        self.encoder = Encoder(self.config)
        self.dense = Dense(units=self.config.embedding_dim, activation=tf.keras.activations.tanh)

    def call(self, inputs, segments):
        # outputs : (batch_size, n_seq, embedding_dim)
        # self_attn_probs : (batch_size, num_heads, n_enc_seq, n_enc_seq)
        outputs, self_attn_probs = self.encoder(inputs, segments)

        # (batch_size, embedding_dim)
        # 첫번째([CLS]) Token을 저장.
        # outputs_cls = outputs[:,0].contiguous()
        outputs_cls = outputs[:, 0]
        # outputs_cls에 Dense 및 tanh activation 통과.
        outputs_cls = self.dense(outputs_cls)

        return outputs, outputs_cls, self_attn_probs


class BERTPretrain(tf.keras.layers.Layer):
    def __init__(self, config):
        super(BERTPretrain, self).__init__()
        self.config = config
        self.bert = BERT(self.config)

        # NSP(Next Sentence Prediction) Classifier
        self.projection_cls = Dense(units=2)
        # MLM(Masked Language Model) classifier
        self.projection_lm = Dense(units=self.config.n_enc_vocab)
        # 왜 MLM weight를 enc_emb weight 와 share하는것인지?
        # 초기 weight를 share하는것인지?
        self.projection_lm.weights = self.bert.encoder.enc_emb.weights

    def call(self, inputs, segments):
        # outputs : (batch_size, n_enc_seq, embedding_dim)
        # outputs_cls : (batch_size, embedding_dim)
        # attn_probs : (batch_size, num_heads, n_enc_seq, n_enc_seq)
        outputs, outputs_cls, attn_probs = self.bert(inputs, segments)

        # (batch_size, 2)
        # cls토큰으로 문장 A와 문장 B의 관계를 예측 (NSP)
        # A의 다음문장이 B가 맞을경우는 True, 아닐경우는 False로 예측
        logits_cls = self.projection_cls(outputs_cls)

        # (batch_size, n_enc_seq, n_enc_vocab)
        # MLM 예측
        logits_lm = self.projection_lm(outputs)

        return logits_cls, logits_lm, attn_probs


def create_pretrain_mask(tokens, mask_cnt, vocab_list):
    """
    Pretrain Data 생성 - [MASK] 토큰 생성
    :param tokens: [CLS]+doc1+[SEP]+doc2+[SEP] 구조.  문장을 토크나이저로 나눈 토큰 리스트.
    :param mask_cnt: mask_cnt는 전체 token개수의 15%에 해당하는 개수.
    :param vocab_list: 단어집합리스트
    :return:
    """
    cand_idx = []

    # 1. token을 단어별로 index 배열 형태로 저장
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # u"\u2581" 는 단어의 시작을 의미하는 값임. 없다면 이전 토큰과 연결된 subword 이라는 의미.
        if 0 < len(cand_idx) and not token.startswith(u"\u2581"):
            cand_idx[-1].append(i)  # 이전 인덱스리스트에 연결하여 인덱스 추가.
        else:
            cand_idx.append([i])

    # 2. 랜덤선택을 위해 단어의 index를 섞는다
    shuffle(cand_idx)

    mask_lms = []
    for index_set in cand_idx:

        # 3. mask_lms 개수가 mask_cnt를 넘지 않도록 함. mask_cnt는 전체 token개수의 15%에 해당하는 개수.
        if len(mask_lms) >= mask_cnt:
            break
        if len(mask_lms) + len(index_set) > mask_cnt:
            continue

        for index in index_set:  # 단어가 여러 토큰으로 이루어져있음을 가정.
            # 토큰마다 똑같은 확률을 적용하여 [MASK] 토큰 생성.
            masked_token = None
            if random() < 0.8:  # 80% replace with [MASK]
                masked_token = "[MASK]"
            else:
                if random() < 0.5:  # 10% keep original
                    masked_token = tokens[index]
                else:  # 10% random word
                    masked_token = choice(vocab_list)
            # mask된 index 값과 정답 label을 mask_lms에 저장
            mask_lms.append({"index": index, "label": tokens[index]})
            # token index의 값을 mask한다.
            tokens[index] = masked_token

    # [CLS]와 [SEP]을 제외하고 랜덤하게 마스크 된 값 정렬
    mask_lms = sorted(mask_lms, key=lambda x: x["index"])

    # 정렬된 값을 이용해 mask_index, mask_label을 생성.
    mask_idx = [p["index"] for p in mask_lms]
    mask_label = [p["label"] for p in mask_lms]

    return tokens, mask_idx, mask_label


def trim_tokens(tokens_a, tokens_b, max_seq):
    """
    최대 길이 초과하는 토큰 자르기. tokenA, tokenB의 길이의 합이 특정 길이보다 길 경우 길이 축소.
    :param tokens_a:
    :param tokens_b:
    :param max_seq: 최대 허용 길이
    :return:
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_seq:
            break
        if len(tokens_a) > len(tokens_b):
            del tokens_a[0]
        else:
            tokens_b.pop()


def create_pretrain_instances(docs, doc_idx, doc, n_seq, mask_prob, vocab_list):
    """
    * doc 별 pretrain 데이터 생성. 단락 최대 길이(n_seq) 만큼 재구성된 학습데이터를 만듦.
    * 단락 구성 : [CLS]+doc1+[SEP]+doc2+[SEP]

    * doc이라는 단락에는 문장단위로 다시 쪼개진다는 구분이 없고, 단지 단락이 단어토큰의 집합으로 이루어져있음. 즉 문장과 동일하게 인식.
    * 따라서 단락내 특정문장의 중간부터, 다음문장의 중간까지의 텍스트가  tokens_a 나 tokens_b로 구성될수 있음을 인식.

    학습데이터 구성방식
        * is_next=0(다음문장이 실제문장이 아님)으로 구성되는 학습데이터는 tokens_a와 tokens_b 모두 길이를 랜덤선택하므로 학습데이터길이가 너무 짧아질수 있음
        * tokens_a의 구성 문장길이가 짧으면 tokens_b에서 가능한 많이 채울수도 있을텐데 그러진 않음.
    :param docs: 단락(문장) 모음
    :param doc_idx: 현재 단락(문장)의 인덱스
    :param doc: 현재 단락(문장)내용
    :param n_seq: 단락(문장) 최대 길이
    :param mask_prob: 단락(문장) 내 토큰에 [MASK] 토큰을 씌울 비중. 통상 0.15 즉 15%
    :param vocab_list: 단어집합
    :return:
    """
    # for [CLS], [SEP], [SEP]
    max_seq = n_seq - 3
    tgt_seq = max_seq

    instances = []
    current_chunk = []
    current_length = 0
    for i in range(len(doc)):
        current_chunk.append(doc[i])  # line
        current_length += len(doc[i])
        # 마지막 단어이거나, 단락(문장) 길이가 문장 최대길이에 도달하면 단락(문장)을 스캔하여 학습데이터를 만듦.
        if i == len(doc) - 1 or current_length >= tgt_seq:
            if 0 < len(current_chunk):  # 현재문장 빈 값 예외처리
                a_end = 1
                # 단락에 구성된 단어가 한개 이상이라면 current_chunk에 있는 단어 중 하나의 인덱스 랜덤선택
                if 1 < len(current_chunk):
                    a_end = randrange(1, len(current_chunk))

                tokens_a = []
                for j in range(a_end):  # 처음단어부터 랜덤 선택된 인덱스의 단어까지 모두 tokens_a에 추가.
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # 단어가 한개밖에 없거나 (똑같은 단어를 두개 쓸수는 없으므로), 50프로의 확률로 다른단락(문장)에서 tokens_b 생성.
                if len(current_chunk) == 1 or random() < 0.5:
                    is_next = 0  # False (다음문장이 실제 문장이 아님)

                    # trim_tokens 메소드를 사용하면 되므로 필요없음
                    # tokens_b_len = tgt_seq - len(tokens_a)

                    random_doc_idx = doc_idx
                    # 다른 단락 랜덤선택
                    while doc_idx == random_doc_idx:
                        random_doc_idx = randrange(0, len(docs))
                    random_doc = docs[random_doc_idx]

                    # 꼭 최대 단락길이를 다 채울필요는 없고, 나중에 최대단락길이만 넘지 않으면 됨.
                    random_start = randrange(0, len(random_doc))  # 다른단락에서 랜덤으로 시작단어 인덱스 get
                    for j in range(random_start, len(random_doc)):  # 랜덤선택한 단어부터 끝까지 tokens_b에 추가.
                        tokens_b.extend(random_doc[j])

                else:  # 50프로의 확률로 동일한 단락에서 tokens_b 생성.
                    is_next = 1  # True (다음문장이 실제 문장임)
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                # tokens_a의 단락 길이가 더 길면 a의 "앞쪽"부터 단어제거
                # tokens_b의 단락 길이가 더 길면 b의 "뒤쪽"부터 단어제거
                trim_tokens(tokens_a, tokens_b, max_seq)
                assert 0 < len(tokens_a)
                assert 0 < len(tokens_b)

                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]

                # len(tokens_a) + 2 : [CLS]와 [SEP] 포함
                # len(tokens_b) + 1 : [SEP] 포함
                # 앞쪽 단락은 0으로 segment 지정, 뒤쪽 단락은 1로 segment 지정
                segment = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

                # Mask Token 개수는 전체 Token수([CLS],[SEP],[SEP]은 제외) 에 mask_prob(통상 0.15 즉, 15%)를 곱하여 구함.
                tokens, mask_idx, mask_label = create_pretrain_mask(tokens, int((len(tokens) - 3) * mask_prob),
                                                                    vocab_list)
                # 단락(문장)별 학습데이터 더미 생성.
                instance = {
                    "tokens": tokens,
                    "segment": segment,  # segment label
                    "is_next": is_next,  # NSP label
                    "mask_idx": mask_idx,  # mask된 토큰의 단락(문장) 내 인덱스
                    "mask_label": mask_label  # mask된 토큰의 정답값
                }
                instances.append(instance)

            # 변수 초기화
            current_chunk = []
            current_length = 0
    return instances


def make_pretrain_data(vocab, in_file, out_file, count, n_seq, mask_prob):
    """
    pretrain 데이터 생성
    :param vocab:
    :param in_file:
    :param out_file:
    :param count:
    :param n_seq:
    :param mask_prob:
    :return:
    """
    vocab_list = []
    # 단어목록 vocab_list를 생성. 생성 시 unknown은 제거
    # vocab_list는 create_pretrain_mask 함수의 입력으로 사용.
    for id in range(vocab.get_piece_size()):
        if not vocab.is_unknown(id):
            vocab_list.append(vocab.id_to_piece(id))

    # 말뭉치 파일 라인수를 확인.
    line_cnt = 0
    with open(in_file, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            line_cnt += 1

    docs = []
    with open(in_file, 'r', encoding='utf-8') as f:
        doc = []
        with tqdm(total=line_cnt, desc=f"Loading") as pbar:  # 진행률 표시
            for i, line in enumerate(f):
                line = line.strip()
                if line == "":  # 줄에 빈값이라면 단락이 종료된것으로 인식. doc를 docs에 추가하고 doc를 새로 만듭니다.
                    if 0 < len(doc):
                        docs.appned(doc)
                        doc = []
                else:
                    # 줄에 구성된 문자를 vocab을 이용해 tokenize한 후 doc에 추가.
                    pieces = vocab.encode_as_pieces(line)
                    if 0 < len(pieces):
                        doc.append(pieces)
                pbar.update(1)
            if doc:
                docs.append(doc)

        # BERT는 Mask를 15%만 하므로 MLM을 학습시에 한번에 전체 단어를 학습할수 없음.
        # 한 말뭉치에 대해 통상 Pretrain 데이터 10개(count로 설정가능) 정도 만들어서 학습하도록 함.
        for index in range(count):
            output = out_file.format(index)
            if os.path.isfile(output): continue

            with open(output, 'w') as out_f:
                with tqdm(total=len(docs), desc=f"Masing") as pbar:
                    # 단락모음을 돌면서 모든 단락에 대해 Pretrain data를 생성하여 하나의 파일에 쓰기.
                    for i, doc in enumerate(docs):
                        # doc은 encode된 토큰(토큰의 인덱스가 아닌 토큰단어)으로 이루어짐.
                        instances = create_pretrain_instances(docs, i, doc, n_seq, mask_prob, vocab_list)
                        for instance in instances:
                            out_f.write(json.dumps(instance))
                            out_f.write("\n")
                        pbar.update(1)


def wrapper_make_pretrain_data():
    in_file = "../kowiki-data/kowiki.txt"
    out_file = "../kowiki-data/kowiki_bert_{}.json"
    count = 10
    # kowiki 학습데이터를 살펴보니 단락이 꽤나 길어서 256개로는 데이터 뒷부분을 제대로 활용하기 힘들수도 있지만... 일단 적용
    n_seq = 256  # 단락(문장) 최대길이.
    mask_prob = 0.15

    make_pretrain_data(vocab, in_file, out_file, count, n_seq, mask_prob)


"""
TODO:
tensorflow 기준
1. train을 하려면 Layer에 정의된 output에 정답데이터가 들어가야하는데, attn_probs에 대한 정답값은 넣을수 없다. 
    -> output인자 attn_probs 빼고 재구성해야함.
2. DataSet에서 tensor_from_slices 이용할때 {'inputs': xx , 'segments' : xx}, {'logits_cls' : xx , 'logits_lm' : xx} 로 구성하기.

"""
class PretrainDataSet(tf.data.Dataset):
    def __init__(self, vocab, in_file):
        super(PretrainDataSet, self).__init__()

        self.vocab = vocab
        self.labels_cls = []
        self.labels_lm = []
        self.sentences = []
        self.segments = []
        # TODO : 데이터 세팅

    def from_tensor_slices(self):
        super().from_tensor_slices(({'inputs': self.sentences, 'segments': self.segments},
                                    {'logits_cls': self.labels_cls, 'logits_lm': self.labels_lm}))
