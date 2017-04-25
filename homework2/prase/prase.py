import re


def prase(diffName):
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
    i = int(temp[0][0])-1
    if temp[0][-1] == '-1':
        k = int(temp[0][-2]) - 1
    else:
        k = int(temp[0][-1]) - 1
    count = 0
    if temp[0][-1] == '-1':
        r = '\n'
        for str in temp[1]:
            r = r + str + '\n'
        fileContent = fileContent.replace(datas[k] + '\n', r, 1)
    elif temp[1] == []:
        while i <= k:
            fileContent = fileContent.replace(datas[i]+'\n', '', 1)
            i += 1
    else:
        oldStr = ""
        while i <= k:
            oldStr = oldStr + datas[i] + '\n'
            i += 1
        newStr = '\n'.join(temp[1])
        newStr += '\n'
        fileContent = fileContent.replace(oldStr, newStr, 1)
    return fileContent


def modify(fileContent, modifyNumber):
    datas = fileContent.split("\n")
    for index, data in enumerate(datas):
        datas[index] = data + " entrypt line"+str(index)
    fileContent = '\n'.join(datas)
    for index in modifyNumber:
        temp = modifyNumber[index]
        # print(index)
        fileContent = singleModify(temp, fileContent, datas)
    datas = fileContent.split("\n")
    for index, data in enumerate(datas):
        length = len(str(index))
        if "entrypt line" in data:
            datas[index] = data[:-(13+length)]
    fileContent = '\n'.join(datas)
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

