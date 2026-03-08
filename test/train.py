from time import perf_counter, sleep
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
def task():
    print('Starting a task')

    sleep(1)

    print('task is done')

def task2():
    print('Starting a task')

    sleep(1)

    print('task is done')

start_time = perf_counter()

with ThreadPoolExecutor(max_workers=2) as executor:
    future1 = executor.submit(task)
    future2 = executor.submit(task2)

    print(future1.result())
    print(future2.result())

end_time = perf_counter()

#     executor = ThreadPoolExecutor(max_workers=2)

#     start_time = perf_counter()

#     a = executor.submit(task)
#     b = executor.submit(task2)

# end_time = perf_counter()


print(f'It took {end_time - start_time: 0.2f} seconds to complete')


# x = numpy.array([12, 3, 5, 6])

'''

scalar_tensor = tf.constant(5)
vector_tensor = tf.constant([
    1,2,3,4,5
])
matrix_tensor = tf.constant([
    [1,2], [3,4]
])


matrix_ones_tensor = tf.ones_like([2,2]) # Ορισμος πινακα 2D-tensor με 4 στοιχεια , 2 γραμμες και 2 στηλες

tensor_one = tf.constant([
    [1,2], [3,4]
])
_3d_tensor = tf.constant([[[
    1,2
],
[
    3,4
],
[
    5,6
]]])
tensor_ones = tf.ones([2, 2], tf.int32)
add_output_tensor = tf.add(tensor_one, tensor_ones)
mul_output_tensor = tf.multiply(tensor_one, tensor_ones)

matrix_slice_matrix = matrix_tensor[:, :]
matrix_slice_vector = vector_tensor[0]
# print(tf.reshape(tensor = matrix_slice_matrix, shape = [1, 4]))


X_train = tf.random.normal([100, 10]) # To μοντελο υπολογιζει y (number) = x1xw1 + x2xw2 +...x10xw10 + b, οπου x = entry του n vector απο 2d tensor
Y_train = tf.random.normal([100, 1]) # 100 y (1 output for each sample/100) σε μια στηλη

model = Sequential()
model.add(keras.Input(shape = (10,)))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1))

# model.compile(optimizer='adam', loss='mse')
# model.fit(X_train, Y_train, epochs=10)

x = numpy.array([[
    1,2,3

],
[
    4,5,6
]])
y = numpy.array([10,20,30])

z = numpy.matmul(x, y)

def naive_vector(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z

def naive_matrix_vector(matrix, vector):
    assert len(matrix.shape) == 2
    assert len(vector.shape) == 1
    assert matrix.shape[1] == vector.shape[0]
    
    z = numpy.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            z[i] += matrix[i, j] * vector[j]
    return z
print(naive_matrix_vector(matrix=x, vector=y))
print(x.shape[0])
'''