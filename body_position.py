from konlpy.tag import Kkma

import pymysql
import sys

conn = pymysql.connect(host='localhost', user='root', password='0000', db='cheongwadae', charset='utf8')
curs = conn.cursor()
SQL = "SELECT * FROM petition"
curs.execute(SQL)
result = curs.fetchall()

kkma = Kkma()

body_pos = {}
for i in range(len(result)):
    body = result[i][2]
    kkma_body = kkma.pos(body)
    if body_pos.get(result[i][3]) == None:
        body_pos[result[i][3]] = []
        body_pos[result[i][3]].append(kkma_body)
    else:
        body_pos[result[i][3]].append(kkma_body)

keys = list(body_pos.keys())


final_dict = {}
for key in keys:
    pos_dict = {}
    for body_list in body_pos[key]:
        for i in range(len(body_list)):
            pos_keys = list(pos_dict.keys())
            if body_list[i] not in pos_keys:
                pos_dict[body_list[i]] = 1
                pos_keys = list(pos_dict.keys())
            else:
                pos_dict[body_list[i]] += 1
    final_dict[key] = pos_dict

'''
for key in list(final_dict.keys()):
    print(key)
    print('---------------------------------------------------------------------')
    new = sorted(final_dict[key].items(), key=lambda t : t[1])
    for item in new:
        print(item)
    # for item in list(final_dict[key].items()):
    #     print(item)
    print('---------------------------------------------------------------------')
'''


print(keys[7])
print('------------')
new = sorted(final_dict[keys[7]].items(), key=lambda x : x[1])
for item in new:
    print(item)
    if item[1] > 3:
        break
print('------------')

print(len(keys))