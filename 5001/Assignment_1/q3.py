import re

# question 1
f = open("blocklist.xml", encoding="utf-8")
bl = f.read()
regular1 = r'\<emItem\sblockID=\"[i|d].*[0-9]\"\s.*>'
pattern1 = re.compile(regular1)
result1 = pattern1.findall(bl)
for i in range(len(result1)):
    print(result1[i])
print('There are', len(result1), 'text lines satisfy the requirement in question 1.')

# question 2
f = open("blocklist.xml", encoding="utf-8")
bl = f.read()
regular2 = r'<.*\sid=\"[^/\^]*[c|o][o|r][m|g]\".*>'
pattern2 = re.compile(regular2)
result2 = pattern2.findall(bl)
for i in range(len(result2)):
    print(result2[i])
print('There are', len(result2), 'text lines satisfy the requirement in question 2.')

