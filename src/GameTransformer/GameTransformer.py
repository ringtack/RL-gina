import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class GameEncoder():

    def __init__(self, image_dim, value_size, latent_dim, action_size):
        super(GameEncoder, self).__init__()
        self.image_dim = image_dim
        self.action_size = action_size

        self.space_encoder = Encoder(image_dim, value_size, latent_dim)
        self.space_encoder.build(input_shape=(None, image_dim * 2))

        self.demon_encoder = Encoder(image_dim, value_size, latent_dim)
        self.demon_encoder.build(input_shape=(None, image_dim * 2))

        self.space_decoder = Decoder(image_dim, value_size, action_size)
        self.space_decoder.build(input_shape=(None, latent_dim))

        self.demon_decoder = Decoder(image_dim, value_size, action_size)
        self.demon_decoder.build(input_shape=(None, latent_dim))

        self.space_discriminator = Discriminator(image_dim, latent_dim)
        self.space_discriminator.build(input_shape=(None, image_dim * 2 + 2))

        self.demon_discriminator = Discriminator(image_dim, latent_dim)
        self.demon_discriminator.build(input_shape=(None, image_dim * 2 + 2))


        self.ae_learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=100, decay_rate=0.99)
        self.ae_optimizer = keras.optimizers.Adam(self.ae_learning_rate)

        self.trans_learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=100, decay_rate=0.99)
        self.trans_optimizer = keras.optimizers.Adam(self.trans_learning_rate)

    def save(self, path):
        self.space_encoder.save_weights(path + '/space_encoder')
        self.demon_encoder.save_weights(path + '/demon_encoder')
        self.space_decoder.save_weights(path + '/space_decoder')
        self.demon_decoder.save_weights(path + '/demon_decoder')
        self.space_discriminator.save_weights(path + '/space_discriminator')
        self.demon_discriminator.save_weights(path + '/demon_discriminator')

    def load(self, path):
        self.space_encoder.load_weights(path + '/space_encoder')
        self.demon_encoder.load_weights(path + '/demon_encoder')
        self.space_decoder.load_weights(path + '/space_decoder')
        self.demon_decoder.load_weights(path + '/demon_decoder')
        self.space_discriminator.load_weights(path + '/space_discriminator')
        self.demon_discriminator.load_weights(path + '/demon_discriminator')

    def experience_to_encode_inp(self, experience_list):
        return tf.convert_to_tensor([tf.concat([e[0], e[1], e[2], e[3]], axis=0) for e in experience_list])

    def decode_out_to_experience(self, decoder_output):
        images, reward_dist, action_dist = self.split_decoder_output(decoder_output)
        reward = tf.math.argmax(reward_dist, axis=1) - 1
        action = tf.math.argmax(action_dist, axis=1)

        out = []
        for i in range(images.shape[0]):
            out.append((images[i], reward[i], action[i]))
        return out

    def train(self, space_exp, demon_exp, batch_size=64):
        space_exp = self.experience_to_encode_inp(space_exp)
        demon_exp = self.experience_to_encode_inp(demon_exp)

        space_exp = tf.split(space_exp, space_exp.shape[0] // batch_size)
        demon_exp = tf.split(demon_exp, demon_exp.shape[0] // batch_size)

        for space_batch, demon_batch in zip(space_exp, demon_exp):
            sloss = self.train_autoencoder(self.space_encoder, self.space_decoder, space_batch)
            print("Space autoencoder loss:", sloss)

            dloss = self.train_autoencoder(self.demon_encoder, self.demon_decoder, demon_batch)
            print("Demon autoencoder loss:", dloss)

            discrm_loss, gen_loss = self.train_translator(self.demon_encoder, self.space_decoder,
                                                          self.space_discriminator, demon_batch, space_batch)
            print(f"Demon to space losses: discrimnator - {discrm_loss} generator - {gen_loss}")

            discrm_loss, gen_loss = self.train_translator(self.space_encoder, self.demon_decoder,
                                                          self.demon_discriminator, space_batch, demon_batch)
            print(f"Space to demon losses: discrimnator - {discrm_loss} generator - {gen_loss}")

    def split_experience_input(self, experience_batch):
        images = experience_batch[:, :self.image_dim * 2]
        rewards = experience_batch[:, self.image_dim * 2: self.image_dim * 2 + 1]
        actions = experience_batch[:, self.image_dim * 2 + 1:]
        return images, rewards, actions

    def split_decoder_output(self, decoder_output):
        images = decoder_output[:, :self.image_dim * 2]
        reward_dist = decoder_output[:, self.image_dim * 2: self.image_dim * 2 + 3]
        action_dist = decoder_output[:, self.image_dim * 2 + 3:]
        return images, reward_dist, action_dist

    def train_autoencoder(self, encode_layer, decode_layer, experience_batch):
        images, rewards, actions = self.split_experience_input(experience_batch)

        with tf.GradientTape() as tape:
            prediction = decode_layer.call(encode_layer.call(images))
            pred_images, pred_rewards, pred_actions = self.split_decoder_output(prediction)

            image_loss = tf.reduce_mean(tf.math.square(images - pred_images))
            pred_action_loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(rewards, pred_rewards) +
                                              keras.losses.sparse_categorical_crossentropy(actions, pred_actions))
            loss = image_loss + pred_action_loss

        grads = tape.gradient(loss, decode_layer.trainable_variables + encode_layer.trainable_variables)
        self.ae_optimizer.apply_gradients(
            zip(grads, decode_layer.trainable_variables + encode_layer.trainable_variables))

        return tf.reduce_mean(loss)

    def train_translator(self, encode_layer, decode_layer, discriminator_model, orig_exp, trans_exp):
        orig_images, _, _ = self.split_experience_input(orig_exp)
        _, trans_r, trans_a = self.split_experience_input(trans_exp)

        with tf.GradientTape(persistent=True) as tape:
            generated = decode_layer.call(encode_layer.call(orig_images))
            gen_image, gen_r, gen_a = self.split_decoder_output(generated)

            generated = tf.concat([gen_image,
                                   tf.cast(tf.expand_dims(tf.math.argmax(gen_r, axis=1) - 1, axis=1), dtype=tf.float32),
                                   tf.cast(tf.expand_dims(tf.math.argmax(gen_a, axis=1), axis=1), dtype=tf.float32)], axis=1)

            gen_discr = discriminator_model(generated)
            real_discr = discriminator_model(trans_exp)

            pred_action_loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(trans_r + 1, gen_r) +
                                              keras.losses.sparse_categorical_crossentropy(trans_a, gen_a))

            discr_loss = discriminator_model.loss_function(gen_discr, real_discr)
            gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_discr),
                                                                              logits=gen_discr))

        discr_grads = tape.gradient(discr_loss, discriminator_model.trainable_variables)
        self.trans_optimizer.apply_gradients(zip(discr_grads, discriminator_model.trainable_variables))

        translator_vars = decode_layer.trainable_variables + encode_layer.trainable_variables
        gen_grads = tape.gradient(gen_loss, translator_vars)
        pred_grads = tape.gradient(pred_action_loss, translator_vars)
        self.trans_optimizer.apply_gradients(zip(gen_grads, translator_vars))
        self.trans_optimizer.apply_gradients(zip(pred_grads, translator_vars))

        return discr_loss, gen_loss + tf.reduce_mean(pred_action_loss)


class Encoder(keras.Model):

    def __init__(self, image_dim, value_dim, latent_dim, **kwargs):
        super(Encoder, self).__init__()
        self.image_dim = image_dim
        self.latent_dim = latent_dim
        self.value_dim = value_dim
        self.state_encoder = Attention(self.image_dim, self.value_dim)
        self.delta_state_encoder = Attention(self.image_dim, self.value_dim)

        self.concat = layers.Concatenate()

        self.dual_encoder = Attention(2 * self.value_dim, self.latent_dim)

    def call(self, inputs, **kwargs):
        state_input, delta_state_input = tf.split(inputs, 2, axis=1)
        state_encoded = self.state_encoder(state_input)
        delta_state_encoded = self.delta_state_encoder(delta_state_input)
        concat = self.concat([state_encoded, delta_state_encoded])
        encoded = self.dual_encoder(concat)
        return encoded


class Decoder(keras.Model):
    def __init__(self, image_dim, latent_dim, action_size, **kwargs):
        super(Decoder, self).__init__()
        self.image_dim = image_dim
        self.latent_dim = latent_dim

        self.dense_1 = layers.Dense(latent_dim, activation='relu', kernel_initializer='he_uniform')
        self.dropout = layers.Dropout(0.3)
        self.dense_2 = layers.Dense(image_dim, activation='relu', kernel_initializer='he_uniform')

        self.att_state_delta = Attention(image_dim, 32)
        self.att_final_delta = layers.Dense(image_dim, activation='tanh')

        self.att_state = Attention(image_dim, 32)
        self.att_final_state = layers.Dense(image_dim, activation='sigmoid')

        self.reward_ff = keras.models.Sequential([
            layers.Dense(latent_dim, activation='relu', kernel_initializer='he_uniform'),
            layers.Dropout(0.3),
            layers.Dense(3,  activation='softmax')
        ])

        self.action_ff = keras.models.Sequential([
            layers.Dense(latent_dim, activation='relu', kernel_initializer='he_uniform'),
            layers.Dropout(0.3),
            layers.Dense(action_size, activation='softmax')
        ])

    def call(self, inputs, **kwargs):
        d_1 = self.dense_1(inputs)
        d_1 = self.dropout(d_1)
        d_2 = self.dense_2(d_1)

        state_attention = self.att_state(d_2)
        state_attention = self.att_final_state(state_attention)

        state_delta_attention = self.att_state_delta(d_2)
        state_delta_attention = self.att_final_delta(state_delta_attention)

        reward_space = self.reward_ff(d_2)
        action_space = self.action_ff(d_2)

        return tf.concat([state_attention, state_delta_attention, reward_space, action_space], axis=1)


class Discriminator(keras.Model):

    def __init__(self, image_dim, latent_dim):
        super(Discriminator, self).__init__()
        self.image_dim = image_dim
        self.latent_dim = latent_dim

        self.state_attention = Attention(image_dim, latent_dim)
        self.state_delta_attention = Attention(image_dim, latent_dim)

        self.dense_1 = layers.Dense(latent_dim, activation='relu', kernel_initializer='he_uniform')
        self.final_decision = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, **kwargs):
        state, state_delta, r, a = tf.split(inputs, [self.image_dim, self.image_dim, 1, 1], axis=1)

        state_attention = self.state_attention(state)
        state_delta_attention = self.state_delta_attention(state_delta)

        rejoined = tf.concat([state_attention, state_delta_attention, r, a], axis=1)
        d_1 = self.dense_1(rejoined)
        d_1 = layers.Dropout(0.3)(d_1)
        return self.final_decision(d_1)

    def loss_function(self, logits_fake, logits_real):
        D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake),
                                                                        logits=logits_fake))
        D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real),
                                                                         logits=logits_real))
        return D_loss


class Attention(layers.Layer):

    def __init__(self, input_size, output_size, **kwargs):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.p = PositionalEmbedding(self.input_size)
        self.atten = layers.MultiHeadAttention(2, 8, dropout=0.3)
        self.dense_1 = layers.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.dropout = layers.Dropout(0.3)
        self.dense_2 = layers.Dense(self.output_size)

    def call(self, inputs, **kwargs):
        pos = self.p(inputs)
        attention = self.atten(pos, pos)
        attention = layers.LayerNormalization()(attention + pos)
        d_1 = self.dense_1(attention)
        d_1 = self.dropout(d_1)
        d_1 = tf.math.reduce_sum(d_1, axis=2)
        d_2 = self.dense_2(d_1)
        return layers.LayerNormalization()(d_2)


class PositionalEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim=2):
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.value = tf

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = tf.stack([x] * self.embed_dim, axis=2)
        return x + positions
