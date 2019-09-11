import tensorflow as tf

from models.model_constructor import Model_constructor


class Model_ResNet34:
    """ResNet34 arxiv.org/1512.03385
    """
    model_name = 'ResNet34'
    input_size = 224
    input_depth = 3
    output_depth = 2

    def __init__(self,input_size=224,input_depth=3,output_depth=2,model_name='ResNet34'):
        self.model_name = model_name
        assert input_size%32 == 0
        self.input_size = input_size
        self.input_depth = input_depth
        self.output_depth = output_depth

    def get_graph(self,mode,exploit_params):
        graph = tf.Graph()
        with graph.as_default():
            Model_constructor._static_init_({'mode':mode,'bn':'own0'})
            self.set_exploit_params(exploit_params)

            # batch_images = preprocess(self.batch_size)
            batch_images = tf.placeholder(tf.float32,
                                          shape=[self.batch_size, self.input_size, self.input_size, self.input_depth],
                                          name='batch_images')
            batch_labels = tf.placeholder(tf.int8,
                                          shape=[self.batch_size, self.output_depth],
                                          name='batch_labels')

            conv0 = Model_constructor.conv_bn_relu_layer(batch_images,(7,7,self.input_depth,64),2,[0])
            # this is as in article in the Table1, in the beginning of the conv2_x block
            pool0 = tf.nn.max_pool(conv0, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            res_block = Model_constructor.residual_block_compact_v4
            id = [0]
            stack_of_blocks_1 = Model_constructor.stack_of_blocks(pool0,res_block,64,3,id)
            stack_of_blocks_2 = Model_constructor.stack_of_blocks(stack_of_blocks_1,res_block,128,4,id)
            stack_of_blocks_3 = Model_constructor.stack_of_blocks(stack_of_blocks_2,res_block,256,6,id)
            stack_of_blocks_4 = Model_constructor.stack_of_blocks(stack_of_blocks_3,res_block,512,3,id)

            pool4 = tf.reduce_mean(stack_of_blocks_4, [1, 2])
            if Model_constructor.get__MODE() == Model_constructor.cons__MODE_train():
                pool4 = tf.nn.dropout(pool4, keep_prob=self.dropout_keep_prob)

            fc_output = Model_constructor.fc_logits(pool4, self.output_depth, [0])

            batch_logits_trn = fc_output
            batch_predictions_trn = tf.nn.softmax(batch_logits_trn, name='batch_predictions')
            batch_loss_trn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=batch_labels, logits=batch_logits_trn), name='batch_loss')

            if Model_constructor.get__MODE() == Model_constructor.cons__MODE_train():
                if self.optimizer == Model_ResNet34.const__optimizers_list()[0]:
                    optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
                elif self.optimizer == Model_ResNet34.const__optimizers_list()[1]:
                    optimizer = tf.train.AdamOptimizer(self.learning_rate)
                beta = self.beta_l2
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    optimizer = optimizer.minimize(batch_loss_trn+Model_constructor.get__l2loss()*beta, name='optimizer')

        return graph

    def set_exploit_params(self, params_dict):
        self.set__batch_size(params_dict)

        if Model_constructor.get__MODE() == Model_constructor.cons__MODE_train():
            self.set__optimizer(params_dict)
            self.set__learning_rate(params_dict)
            self.set__beta_l2(params_dict)
            self.set__dropout(params_dict)

    def set__batch_size(self, params_dict):
        self.batch_size = params_dict['batch_size']
        assert type(self.batch_size) == int
        assert self.batch_size > 0

    def set__optimizer(self, params_dict):
        self.optimizer = params_dict['optimizer']
        assert self.optimizer in Model_ResNet34.const__optimizers_list()

    def set__learning_rate(self, params_dict):
        self.learning_rate = params_dict['learning_rate']
        assert type(self.learning_rate) == float and self.learning_rate > 0

    def set__beta_l2(self, params_dict):
        self.beta_l2 = params_dict['beta_l2']
        assert type(self.beta_l2) == float or self.beta_l2 == 0

    def set__dropout(self, params_dict):
        self.dropout_keep_prob = params_dict['dropout']
        assert type(self.dropout_keep_prob) == float or self.dropout_keep_prob == 0

    def get_exploit_params(self):
        exploit_params = {
            'model_name':self.model_name,
            'batch_size':self.batch_size,
            'optimizer':self.optimizer,
            'learning_rate':self.learning_rate,
            'beta_l2':self.beta_l2,
            'dropout_keep_prob':self.dropout_keep_prob,
        }
        return exploit_params

    @staticmethod
    def const__optimizers_list():
        return ('RMSProp','Adam')


def test():
    model = Model_ResNet34()
    graph = model.get_graph(mode='train',
                            exploit_params={'batch_size':64,'optimizer':'RMSProp','learning_rate':0.0001,
                                            'beta_l2':0.0001,'dropout':0.2})
    graph1 = model.get_graph(mode='inference',exploit_params={'batch_size':64})
    graph2 = model.get_graph('inference',{'batch_size':1})
    with tf.Session(graph=graph) as session:
        saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        saver.save(session,'./model.ckpt')
        with tf.Session(graph=graph1) as session1:
            saver = tf.train.Saver()
            saver.restore(session1,'./model.ckpt')
            with tf.Session(graph=graph2) as session2:
                saver = tf.train.Saver()
                saver.restore(session2,'./model.ckpt')
    pass
