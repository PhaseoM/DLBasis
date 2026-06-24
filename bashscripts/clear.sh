#!/bin/bash

TARGET_DIR="$1"

if [ -z "$TARGET_DIR" ]; then
    echo "usage: $0 <directory>"
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: $TARGET_DIR does not exist or is not a directory"
    exit 1
fi

# 防止误删根目录
if [ "$TARGET_DIR" = "/" ]; then
    echo "Dangerous Operation: refusing to clear the root directory"
    exit 1
fi

find "$TARGET_DIR" -mindepth 1 -delete

echo "Cleared folder: $TARGET_DIR"