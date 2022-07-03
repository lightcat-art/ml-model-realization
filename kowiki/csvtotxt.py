import pandas as pd
import sys
import csv

if __name__ == "__main__":
    in_file = "../kowiki-data/kowiki_20220703.csv"
    out_file = "../kowiki-data/kowiki.txt"
    SEPARATOR = u"\u241D"
    print("csvtotxt : read_csv start")
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)

    df = pd.read_csv(in_file, sep=SEPARATOR, engine="python", encoding="utf-8")
    print("csvtotxt : write outfile start")
    with open(out_file, "w", encoding='utf-8') as f:
        for index, row in df.iterrows():
            f.write(row["text"])  # title 과 text를 중복 되므로 text만 저장 함
            f.write("\n\n\n\n")  # 구분자
