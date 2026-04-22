#!/usr/bin/env bash

BASE_URL="https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v2.1"


# download 0 to 26, total 27 files
for i in $(seq -w 0 6); do
    FILENAME="train-0000${i}-of-00007.parquet"
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
