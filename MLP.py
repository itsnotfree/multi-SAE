import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


dos_type = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'processtable', 'udpstorm', 'mailbomb', 'apache2']
probing_type = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
r2l_type = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'warezmaster', 'warezclient', 'spy', 'sendmail',
            'xlock', 'snmpguess', 'named', 'xsnoop', 'snmpgetattack', 'worm']
u2r_type = ['buffer_overflow', 'loadmodule', 'perl', 'rootkit', 'xterm', 'ps', 'httptunnel', 'sqlattack']
type2id = {'normal': 0}

for i in dos_type:
    type2id[i] = 1
for i in probing_type:
    type2id[i] = 1
for i in r2l_type:
    type2id[i] = 1
for i in u2r_type:
    type2id[i] = 1

# protocol -> id
all_protocol = ['tcp', 'udp', 'icmp']
protocol_dict = {}
for id, name in enumerate(all_protocol):
    protocol_dict[name] = id

# service -> id total:70
all_service = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo',
               'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http',
               'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link',
               'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u',
               'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell',
               'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i',
               'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
service_dict = {}
for id, name in enumerate(all_service):
    service_dict[name] = id

# flag -> id
all_flag = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
flag_dict = {}
for id, name in enumerate(all_flag):
    flag_dict[name] = id

####################
# read training data
import csv

all_train_data = []
trainX = []
trainY = []
with open('KDDTrain+.txt', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        all_train_data.append(row)

for row in all_train_data:
    row[1] = protocol_dict[row[1]]
    row[2] = service_dict[row[2]]
    row[3] = flag_dict[row[3]]
    row[-2] = type2id[row[-2]]
    trainX.append(row[:41])
    trainY.append(row[-2])
train_label = []
for i in trainY:
    label_list = [0 for num in range(2)]
    label_list[int(i)] = 1
    train_label.append(label_list)

print(np.array(trainX).shape)
print(np.array(trainY).shape)


# parameters
training_epoch = 20
learning_rate = 0.001
batch_size = 32
total_batch = int(len(trainY) / batch_size)

################
# read test data
all_test_data = []
testX = []
testY = []
with open('KDDTest+.txt', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        all_test_data.append(row)
for row in all_test_data:
    row[1] = protocol_dict[row[1]]
    row[2] = service_dict[row[2]]
    row[3] = flag_dict[row[3]]
    row[-2] = type2id[row[-2]]
    testX.append(row[:41])
    testY.append(row[-2])
test_label =[]
for i in testY:
    label_list = [0 for num in range(2)]
    label_list[int(i)] = 1
    test_label.append(label_list)

print(np.array(testX).shape)
print(np.array(testY).shape)

# placeholder
x = tf.placeholder(tf.float32, [None, 41])
y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)

# MLP
W1 = tf.get_variable('W1', shape=[41, 30], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([30]))
L1 = tf.nn.relu(tf.matmul(x, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable('W2', shape=[30, 30], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([30]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable('W3', shape=[30, 30], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([30]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable('W4', shape=[30, 30], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([30]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable('W5', shape=[30, 2], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([2]))
hypothesis = tf.nn.relu(tf.matmul(L4, W5) + b5)

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print('==Training started!!!==')
for epoch in range(training_epoch):
    avg_cost = 0

    for i in range(total_batch):
        x_train = trainX[i * batch_size:(i+1) * batch_size]
        y_train = train_label[i * batch_size:(i+1) * batch_size]
        c, _ = sess.run([cost, optimizer], feed_dict={x: x_train, y: y_train, keep_prob: 0.7})
        avg_cost += c / total_batch

    print('Epoch', '%03d' % (epoch + 1), 'cost=', '%.9f' % avg_cost)
print('==Training finished!!!==')

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy', sess.run(accuracy, feed_dict={x: testX, y: test_label, keep_prob: 1}))