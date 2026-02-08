from collections import Counter
from collections import defaultdict
count_dict = defaultdict(int)

list1 = [1, 2, 3, 4, 5, 6]
list2 = [2, 7, 9, 0, 4]
list3 = [100, 3, 2, 19, 56]

for id in list1:
    count_dict[id] += 1
print(count_dict)
for id in list2:
    count_dict[id] += 1
print(count_dict)
for id in list3:
    count_dict[id] += 1
print(count_dict)

cnt1, cnt2 = 0, 0
for key, value in count_dict.items():
    cnt1 += 1
    cnt2 += value
print(cnt2 / cnt1)