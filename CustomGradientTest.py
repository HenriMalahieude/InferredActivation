import tensorflow as tf

a = tf.Variable(tf.ones([2]))
b = tf.Variable(tf.ones([4]))

@tf.custom_gradient
def fnfn(x):
    y = a[1] * x + b[2]
    def grad_fn(dx, variables):
        dy_dx = dx * a[1]
        
        gscalar = tf.reduce_sum(dy_dx)

        da = tf.constant([0, 1], dtype=x.dtype)
        db = tf.constant([0, 0, 1, 0], dtype=x.dtype)

        grad_vars = [da*gscalar, db*gscalar]
        #print(dy_dx, gscalar, da, db, grad_vars, sep=" - ")
        return dy_dx, grad_vars
    
    return y, grad_fn

input = tf.constant([1.,2.,3.])
with tf.GradientTape(persistent=True) as tape:
    tape.watch(input)
    result = fnfn(input)

print()
print(tape.gradient(result, input))
print(tape.gradient(result, a))
print(tape.gradient(result, b))