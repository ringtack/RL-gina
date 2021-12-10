import tensorflow as tf


def populate_sets(space_env, demon_env, n_episodes=10, max_steps=1000):
    train_x = []
    train_y = []

    for i in range(n_episodes):

        def flatten_gray(disp):
            return tf.reshape(tf.cast(tf.convert_to_tensor(disp), tf.float32), [-1]) / 255

        space_obs = space_env.reset()
        demon_obs = demon_env.reset()

        demon_done = False
        space_done = False

        for t in range(max_steps):
            train_x.append(flatten_gray(space_obs))
            train_y.append(flatten_gray(demon_obs))
            if space_done or demon_done:
                break
            action = space_env.action_space.sample()
            space_obs, _, space_done, _ = space_env.step(action)
            demon_obs, _, demon_done, _ = demon_env.step(action)

    return tf.convert_to_tensor(train_x), tf.convert_to_tensor(train_y)


def populate_delta_sets(space_env, demon_env, n_episodes=10, max_steps=1000, time_sensitive=False):
    train_x = []
    train_y = []

    for i in range(n_episodes):

        def flatten_gray(disp, time=-1):
            flattened = tf.reshape(tf.cast(tf.convert_to_tensor(disp), tf.float32), [-1]) / 255
            if time_sensitive and time >= 0:
                flattened = tf.concat([flattened, tf.cast(tf.convert_to_tensor([time]), tf.float32)], axis=0)
            return flattened

        prev_space_obs = space_env.reset()
        prev_demon_obs = demon_env.reset()

        demon_done = False
        space_done = False

        for t in range(max_steps):

            if space_done or demon_done:
                break
            action = space_env.action_space.sample()
            space_obs, _, space_done, _ = space_env.step(action)
            demon_obs, _, demon_done, _ = demon_env.step(action)

            train_x.append(flatten_gray(space_obs - prev_space_obs, t))
            train_y.append(flatten_gray(demon_obs - prev_demon_obs))

            prev_space_obs = space_obs
            prev_demon_obs = demon_obs

    return tf.convert_to_tensor(train_x), tf.convert_to_tensor(train_y)

def populate_experience_dataset(space_env, demon_env, n_elts):
    train_space = []
    train_demon = []

    def flatten_gray(disp):
        return tf.reshape(tf.cast(tf.convert_to_tensor(disp), tf.float32), [-1]) / 255

    space_done = False
    demon_done = False

    space_obs = space_env.reset()
    demon_obs = demon_env.reset()

    for i in range(n_elts):
        if space_done:
            space_obs = space_env.reset()
        if demon_done:
            demon_obs = demon_env.reset()

        space_action = space_env.action_space.sample()
        next_space_obs, space_reward, space_done, _ = space_env.step(space_action)
        train_space.append((flatten_gray(space_obs), flatten_gray(next_space_obs) - flatten_gray(space_obs),
                            tf.convert_to_tensor([space_reward], dtype=tf.float32),
                            tf.convert_to_tensor([space_action], dtype=tf.float32)))
        space_obs = next_space_obs

        demon_action = demon_env.action_space.sample()
        next_demon_obs, demon_reward, demon_done, _ = demon_env.step(demon_action)
        train_demon.append((flatten_gray(demon_obs), flatten_gray(next_demon_obs) - flatten_gray(demon_obs),
                            tf.convert_to_tensor([demon_reward], dtype=tf.float32) ,
                            tf.convert_to_tensor([demon_action], dtype=tf.float32)))
        demon_obs = next_demon_obs

    return train_space, train_demon

