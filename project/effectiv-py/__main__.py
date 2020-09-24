#!python3.8
snacks = [('bacon', 350), ('donut', 240), ('muffin', 190)]

for rank, (name, calories) in enumerate(snacks):
    print(f'#{rank}: {name} has {calories} calories')


if (count := 1):
    print(f'count = {count}')
else:
    print('false')


# catch-all アンパック
car_ages = [0, 9 ,3, 5, 20, 19, 1, 6, 15]
print(car_ages)
oldest, *others, youngest = car_ages
print(oldest, others, youngest)

# *others = car_ages １つ指も指定部分が存在しない為SyntaxErrorになる。

short_list = [1, 2]
first, second, *other = short_list
print(first, second, other) # 1, 2, []