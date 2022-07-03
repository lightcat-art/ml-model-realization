import os
import sys

import sentencepiece as spm


def make_vocab_model():
    corpus = "../kowiki-data/kowiki.txt"
    output = "../kowiki-data"
    prefix = "kowiki"
    vocab_size = 8000
    print("makevocab : train start")
    spm.SentencePieceTrainer.train(
        # input : 입력 corpus
        # prefix : 저장할 모델 이름
        # vocab_size : vocab 개수 ( 기본 8000에 스페셜 토큰 7개 더해서 8007개 )
        f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" +
        " --model_type=bpe" +  # 서브워드 토크나이저 알고리즘
        " --max_sentence_length=999999" +  # 문장 최대 길이
        " --pad_id=0 --pad_piece=[PAD]" +  # pad token id, 값
        " --unk_id=1 --unk_piece=[UNK]" +  # unknown token id, 값
        " --bos_id=2 --bos_piece=[BOS]" +  # begin of sequence token id, 값
        " --eos_id=3 --eos_piece=[EOS]" +  # end of sequence token is, 값
        " --user_defined_symbols=[SEP],[CLS],[MASK]"  # 사용자 정의 토큰
    )
    print("makevocab : train end")


def test_vocab():
    vocab_file = "kowiki.model"
    vocab = spm.SentencePieceProcessor()
    vocab.load(vocab_file)

    lines = [
        "겨울이 되어서 날씨가 무척 추워요.",
        "이번 성탄절은 화이트 크리스마스가 될까요?",
        "겨울에 감기 조심하시고 행복한 연말 되세요."
    ]
    for line in lines:
        pieces = vocab.encode_as_pieces(line)
        ids = vocab.encode_as_ids(line)
        print(line)
        print(pieces)
        print(ids)
        print()


if __name__ == "__main__":
    args = os.sys.argv

    # xx.py 인자까지 argument로 인식함
    if not len(args) == 2:
        sys.exit('not enough argument length.')

    option = args[1]
    if option == "--train":
        make_vocab_model()
    if option == "--test":
        test_vocab()
