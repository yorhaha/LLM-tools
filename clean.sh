#!/bin/bash

SEARCH_DIR="."

# Delete "global_step*" directories
find "$SEARCH_DIR" -type d -name 'global_step*' -exec echo {} \; -exec rm -rf {} +

# Delete "rng_state*" files
find "$SEARCH_DIR" -type f -name 'rng_state*' -exec echo {} \; -exec rm -rf {} +

# Make "config.json" files use cache
find "$SEARCH_DIR" -type f -name "config.json" -print0 | while IFS= read -r -d '' file
do
    sed -i 's/"use_cache": false,/"use_cache": true,/g' "$file"
done

# Make "tokenizer_config.json" files use left padding
find "$SEARCH_DIR" -type f -name "tokenizer_config.json" -print0 | while IFS= read -r -d '' file
do
    sed -i 's/"padding_side": "right",/"padding_side": "left",/g' "$file"
done
