# -*- coding: utf-8 -*-
# 4.1.1 文字の出現頻度を数えるには
from collections import Counter
string = "This is a pen."
cnt = Counter(string)
print(cnt)
print( cnt['i'] )
