import tensorflow as tf

cluster_spec = tf.train.ClusterSpec({'worker' : ['localhost:2223'], 'ps' : ['localhost:2222']})
server = tf.train.Server(cluster_spec,job_name='worker')
server = tf.train.Server(cluster_spec,job_name='ps')

tf.reset_default_graph()

#create local graph like normal specifying the local device
with tf.device('/job:worker/task:0'):
    a = tf.Variable([0.],name='a',collections=[tf.GraphKeys.LOCAL_VARIABLES])
    b = tf.constant([100.])
    loss = tf.abs(a-b)
    
    optimizer = tf.train.GradientDescentOptimizer(.1)
    grads,local_vars = zip(*optimizer.compute_gradients(loss,var_list=tf.local_variables()))
    local_update = optimizer.apply_gradients(zip(grads,local_vars))
    
    
    init_local = tf.local_variables_initializer()
    
#create the globabl copies on the ps
with tf.device('/job:ps/task:0'):
    for v in tf.local_variables():
        v_g = tf.get_variable('g/'+v.op.name,
                            shape = v.shape,
                            dtype = v.dtype,
                            trainable=True,
                            collections=[tf.GraphKeys.GLOBAL_VARIABLES,tf.GraphKeys.TRAINABLE_VARIABLES])
        
#global updates
with tf.device('/job:worker/task:0'):
    #this needs to be updated.  Clearly not robust for any graph more complext
    global_vars = tf.global_variables()
    global_update = optimizer.apply_gradients(zip(grads,global_vars))
    
#create init op on the chief node
with tf.device('/job:worker/task:0'):
    init_global = tf.global_variables_initializer()
    
a_global = tf.global_variables()[0]

print(a.device)
print(b.device)
print(loss.device)
print(local_update.device)
print(global_update.device)
print(init_global.device)
print(init_local.device)
print(a_global.device)

sess = tf.Session(target=server.target)
sess.run([init_local,init_global])

sess.run([a,a_global])

sess.run(local_update)

sess.run([a,a_global])

sess.run(global_update)

sess.run([a,a_global])