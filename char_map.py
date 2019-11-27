import os
# -*- coding: UTF-8 -*-
char_map = {}
index_map = {}
for line in open("dict_4562.txt"):
    line = line.strip("\n")
    ch, index = line.split(" ")
    char_map[ch] = int(index)
    index_map[int(index)] = ch
index_map[0] = ' '
char_map[' '] = 0
