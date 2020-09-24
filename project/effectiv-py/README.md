# Effective Python

### walrus演算子(セイウチ)
Python3.8の新構文で以下のような変数.
```
a := b
```
今までは以下のようなコードの場合、不要な変数に再度アクセスされる可能性がありまた値を書き換えられる心配がある為
```python
・・・

count = get_count() # 問題点1:不要なアクセスにより値が書き換わる可能性がある.
if count >= 4:
    make_cider(count)
else:
    out_of_stock()

・・・
# 問題点2:この段階でもifブロック内だけで使用するcount変数が見えてしまう。

```

walrus演算子を使用すると以下のように問題点を修正できる。

```python
・・・

if (count := get_count()) >= 4 :
    make_cider(count)
else:
    out_of_stock()

・・・
```

以下のような例も改善できます。

```python
count = fresh_fruit.get('banana', 0)
if count >= 2 :
    pieces = slice_bananas(count)
    to_enjoy = make_smoothies(pieces)
else:
    count = fresh_fruit.get('apple', 0)
    if count >= 4:
        to_enjoy =  make_cider(count)
    else:
        count = fresh_fruit.get('lemon', 0)
        if count:
            to_enjoy = make_lemonade(count)
        else:
            to_enjoy = 'Nothing'
```
walrus演算子を使用すると以下のように改善される
```python
if (count := fresh_fruit.get('banana', 0)) >= 2 :
    pieces = slice_bananas(count)
    to_enjoy = make_smoothies(pieces)
elif (count := fresh_fruit.get('apple', 0)) >= 4:
    to_enjoy = make_cider(count)
elif count := fresh_fruit.get('lemon', 0):
    to_enjoy = make_lemonade(count)
else:
    to_enjoy = 'Nothing'
```