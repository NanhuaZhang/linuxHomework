import re


def prase(diffName):
    # datas=re.sub("([\d]+[\w][\d]+)+",'',diffInfo).split('\n')
    # datas = diffInfo.split('\n')
    # print(datas)
    with open(diffName, 'r') as f:
        diffs = f.read()
    diffs = diffs.split('\n')
    f.close()
    LineNumber = {}
    pcontent = []
    count = 0
    lineMerge = []
    for diff in diffs:
        if diff == '---' or diff == '':
            continue
        if diff[0] == '>':
            continue
        if diff[0] == '<':
            pcontent.append(diff[2:])
            continue
        if diff[0] != '<' or diff[0] != '>':
            lineMerge.append(pcontent)
            # print(lineMerge)
            pcontent = []
        lineMerge = []
        d = re.split('[a-z]', diff)
        dp = d[1].split(',')
        if 'd' in diff :
            dp.append('-1')
            lineMerge.append(dp)
        else:
            lineMerge.append(dp)
        LineNumber[count] = lineMerge
        count += 1
    lineMerge.append(pcontent)
    print(LineNumber)
    return LineNumber


def singleModify(temp, fileContent, datas):
    # datas = fileContent.split("\n")
    i = int(temp[0][0])-1
    # if temp[0][-1] == 'd':
    #     k = i+len(temp[1])
    # else:
    if temp[0][-1] == '-1':
        k = int(temp[0][-2])-2
    else:
        k = int(temp[0][-1]) - 1
    count = 0
    if temp[0][-1] == '-1':
        str = datas[k]+'\n'
        r = '\n'.join(temp[1])
        str += r
        fileContent = fileContent.replace(datas[k], str, 1)
    elif temp[1] == []:
        while i <= k:
            fileContent = fileContent.replace(datas[i], '', 1)
            i += 1
    else:
        # while i <= k:
        #     fileContent = fileContent.replace(datas[i], temp[1][count], 1)
        #     count += 1
        #     i += 1
        oldStr = ""
        while i <= k:
            oldStr += datas[i]
            i += 1
        newStr = '\n'.join(temp[1])
        fileContent = fileContent.replace(oldStr, newStr, 1)
    return fileContent


def modify(fileContent, modifyNumber):
    datas = fileContent.split("\n")

    for index in modifyNumber:
        temp = modifyNumber[index]
        # print(index)
        fileContent = singleModify(temp, fileContent, datas)
    with open("oldFile.py", "w") as f:
        f.write(fileContent)
    f.close()
    # print(fileContent)

def run(filename, diffname):
    with open(filename, 'r') as f:
        fileContent = f.read()
        # print(fileContent)
        number = prase(diffname)
        modify(fileContent, number)
    f.close()

run("2.py", "test.txt")

