#!/usr/bin/env bash

BASE_URL="https://huggingface.co/datasets/McGill-NLP/TopiOCQA-wiki-corpus/resolve/main/data"


# download 0 to 26, total 27 files
for i in $(seq -w 0 26); do
    FILENAME="train-${i}-of-00027.parquet"
    URL="${BASE_URL}/${FILENAME}"

    echo "↓ do $FILENAME ..."

    # wget with continue option
    wget -c "$URL" -O "${FILENAME}"
    
    # check wget exit status
    if [[ $? -ne 0 ]]; then
        echo "⚠️  : $FILENAME"
    else
        echo "✔️  download ok: $FILENAME"
    fi
done

echo "🎉 all done"
