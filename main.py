# importing libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree

# print("Hello")
# reading dataset txt file
data = pd.read_csv('data_37,2,500.txt', sep=" ", header=None)
input_no = 37
output_no = 2
instances = 500
rc = []
for i in range(input_no):
    rc.append('i' + str(i))
for i in range(output_no):
    rc.append('o' + str(i))
data.columns = rc
# print(data)

# declaring input-output data
input = rc[:input_no]
output = rc[input_no:]
# print(input)
# print(output)
X = data[input]
Y = data[output]
# print(X)
# print(Y)

# splitting training-testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
# print(X_train)
# print(Y_train)
# print(X_test, 'x_test')
# print(Y_test)
test_size = round(0.1 * instances)
# print(instances, 'instances')
# print(test_size, 'test_size')
test_matrix = [[0 for i in range(input_no)] for j in range(test_size)]
varify_matrix = [[0 for i in range(output_no)] for j in range(test_size)]
for i in range(input_no):
    cnt1 = 0
    for j in X_test['i' + str(i)]:
        test_matrix[cnt1][i] += int(j)
        cnt1 += 1
for i in range(output_no):
    cnt1 = 0
    for j in X_test['i' + str(i)]:
        varify_matrix[cnt1][i] += int(j)
        cnt1 += 1
# print(test_matrix)

# writing files for input testing and output testing
with open('test_data.txt', 'w') as test_file:
    for i in range(test_size):
        for j in range(input_no):
            test_file.write(str(test_matrix[i][j]) + ' ')
        test_file.write('\n')
with open('varify_data.txt', 'w') as varify_file:
    for i in range(test_size):
        for j in range(output_no):
            varify_file.write(str(varify_matrix[i][j]) + ' ')
        varify_file.write('\n')


# visiting every leaf of decision tree to get the minterms
def visit_every_leaf(gate, id, clf, rightChildDict, SOP):
    # print('called')
    if clf.tree_.children_left[id] == clf.tree_.children_right[id]:  # condition to check whether a node is a leaf
        if rightChildDict[id] == 1:  # checking whether leaf is labelled with '1'
            SOP.append(gate)
        return;
    visit_every_leaf(gate + str(clf.tree_.feature[id]) + "- and ", clf.tree_.children_left[id], clf, rightChildDict,
                     SOP)  # - to remember negation
    visit_every_leaf(gate + str(clf.tree_.feature[id]) + "+ and ", clf.tree_.children_right[id], clf, rightChildDict,
                     SOP)  # + to remember not a negation


# Converting SOP into circuit
def ConvertSopStringToCircuit(all_gates, andSOP, input_no, output_no, cnt, SOP, all_cir):
    # for keys in all_gates:
    # print(keys,all_gates[keys])
    # Converting SOP string into feasible Gate-circuit
    GateList = andSOP.split()
    if len(GateList) != 0:
        GateList.pop()
    # print(GateList)
    lim = len(GateList)
    for i in range(lim):
        if GateList[i] != 'and' and GateList[i] != 'or':
            if GateList[i][-1] == '-':
                GateList[i] = 'G' + str(int(GateList[i][:-1]) + input_no)
            else:
                GateList[i] = 'G' + GateList[i][:-1]
    # print(GateList)
    # print(int(lim/2))
    while len(GateList) > 1:
        first = GateList.pop()
        second = GateList.pop()
        third = GateList.pop()
        # print(first,second,third,'f','s','t')#defining gates into dictionary
        if second == 'and':
            if 'AND' + '(' + first + ',' + third + ')' not in all_cir:
                all_gates['G' + str(cnt[0])] = 'AND' + '(' + first + ',' + third + ')'
                all_cir['AND' + '(' + first + ',' + third + ')'] = 'G' + str(cnt[0])
                GateList.append('G' + str(cnt[0]))
                cnt[0] += 1
            else:
                GateList.append(all_cir['AND' + '(' + first + ',' + third + ')'])
        # print(cnt,'#cnt')
        # print('G' + str(cnt),'a')
        # print(GateList)
    # print(GateList)
    if len(GateList) != 0:
        SOP.append(GateList.pop())
    # print(cnt,'!!!!in func cnt')
    # cnt = cnt
    # for i in all_gates:
    # print(i, ':', all_gates[i])


# applying or gates among all the minterms
def EvaluateOrGates(orGateList, all_gates, all_cir, output_no, cnt):
    while len(orGateList) > 1:
        first = orGateList.pop()
        sec = orGateList.pop()
        all_gates['G' + str(cnt[0])] = 'OR(' + first + ',' + sec + ')'
        orGateList.append('G' + str(cnt[0]))
        cnt[0] += 1
    if len(orGateList) != 0:
        all_gates['OUTPUT' + str(output_no)] = orGateList.pop()


# training model for each output and storing them into one circuit
accuracies = np.zeros((output_no, 1), dtype=int)
all_gates = {}  # dictionary to store definitions of all gates
all_cir = {}
cnt = []
cnt.append(0)
for i in range(input_no):
    all_gates['G' + str(cnt[0])] = 'input' + str(i)
    all_cir['input' + str(i)] = 'G' + str(cnt[0])
    cnt[0] += 1
for i in range(input_no):
    all_gates['G' + str(cnt[0])] = 'NOT(' + 'G' + str(i) + ')'
    all_cir['NOT(' + 'G' + str(i) + ')'] = 'G' + str(cnt[0])
    cnt[0] += 1
for i in range(output_no):
    classifier = DecisionTreeClassifier(criterion='entropy')
    classifier = classifier.fit(X_train, Y_train['o' + str(i)])
    Y_predict = classifier.predict(X_test)
    accuracies[i] = 100 * metrics.accuracy_score(Y_test['o' + str(i)], Y_predict)
    id = 0
    parentID = -1
    SOP = []
    rightChildDict = {}
    right_children = classifier.tree_.children_right
    for j in range(classifier.tree_.node_count + 1):
        rightChildDict[j] = 0
    for right in right_children:
        if right != -1:
            rightChildDict[right] = 1
    or_gates = []
    gate = ""
    visit_every_leaf(gate, id, classifier, rightChildDict, SOP)
    if len(SOP) == 0:
        if Y_predict[0] == 0:
            all_gates['G' + str(cnt[0])] = 'OR(G0,G' + str(input_no) + ')'
            all_gates['OUTPUT' + str(i)] = 'G' + str(cnt[0])
            cnt[0] += 1
        elif Y_predict[0] == 1:
            all_gates['G' + str(cnt[0])] = 'AND(G0,G' + str(input_no) + ')'
            all_gates['OUTPUT' + str(i)] = 'G' + str(cnt[0])
            cnt[0] += 1

    for k in SOP:
        # print(k, i)
        if len(k) != 0:
            # print(cnt,'////cnt')
            ConvertSopStringToCircuit(all_gates, k, input_no, i, cnt, or_gates, all_cir)
            # print(cnt,'\\\\cnt')
    # print(or_gates,'sop')
    EvaluateOrGates(or_gates, all_gates, all_cir, i, cnt)
    # print(final_SOP)
    # print(cnt[0], 'cnt')

with open('circuit.txt', 'w') as circuit_file:
    for i in all_gates:
        v = all_gates[i]
        if v[0] == 'i':
            circuit_file.write('INPUT(' + str(int(i[1:]) + 1) + ')' + '\n')

    circuit_file.write('\n')

    for i in all_gates:
        v = all_gates[i]
        if i[0] == 'O':
            circuit_file.write('OUTPUT(' + str(int(v[1:]) + 1) + ')' + '\n')

    circuit_file.write('\n')
    for i in all_gates:
        v = all_gates[i]
        if v[0] == 'N':
            circuit_file.write(
                (str(int(i[1:]) + 1) + ' = ' + 'NOT(' + str(int(v[v.find('G') + 1: v.find(',')]) + 1) + ')') + '\n')
        elif v[0] == 'A':
            circuit_file.write(
                str(int(i[1:]) + 1) + ' = ' + 'AND(' + str(int(v[v.find('G') + 1: v.find(',')]) + 1) + ', ' + str(
                    int(v[v.find(',') + 2: v.find(')')]) + 1) + ')' + '\n')
        elif v[0] == 'O':
            circuit_file.write(
                str(int(i[1:]) + 1) + ' = ' + 'OR(' + str(int(v[v.find('G') + 1: v.find(',')]) + 1) + ', ' + str(
                    int(v[v.find(',') + 2: v.find(')')]) + 1) + ')' + '\n')

for j in accuracies:
    print(j, '%')
