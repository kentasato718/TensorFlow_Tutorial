# Import 'tensorflow'
import tensorflow as tf

# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

# Initialize the session
sess = tf.Session()

# Print the result
print(sess.run(result))

# Close the session
sess.close()

