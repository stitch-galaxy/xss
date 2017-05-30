import numpy as np
import scipy.misc
import tensorflow as tf

import vgg

VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
CONTENT_LAYER = 'relu4_2'
GRID_PX = 11
LEARNING_RATE = 1e1
# COLOR_WEIGHT = 0
COLOR_WEIGHT = 100
ITERATIONS = 1000
ITER_PER_CHECKPOINT = 50

RGB_TO_YUV = np.array([[0.299, 0.587, 0.114],
                       [-0.14713, -0.28886, 0.436],
                       [0.615, -0.51499, -0.10001]], dtype=np.float32)

YUV_TO_RGB = np.array([[1., 0., 1.13983],
                       [1., -0.39465, -0.58060],
                       [1., 2.03211, 0.]], dtype=np.float32)


def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


def generate_pattern(content):
    content_features = {}

    shape = (1,) + content.shape

    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net, mean_pixel = vgg.net(VGG_PATH, image)

        content_pre = np.array([vgg.preprocess(content, mean_pixel)])
        content_features[CONTENT_LAYER] = net[CONTENT_LAYER].eval(
            feed_dict={image: content_pre})

    with tf.Graph().as_default():
        x_px = shape[1]
        y_px = shape[2]
        x_g_px = x_px // GRID_PX + (0 if x_px % GRID_PX == 0 else 1)
        y_g_px = y_px // GRID_PX + (0 if y_px % GRID_PX == 0 else 1)

        # fill initialization colors array
        init_colors = np.zeros((x_g_px, y_g_px, 3), dtype=np.float32)
        for i in range(x_g_px):
            for j in range(y_g_px):
                x_s_px = i * GRID_PX
                x_e_px = min((i + 1) * GRID_PX, x_px)
                y_s_px = j * GRID_PX
                y_e_px = min((j + 1) * GRID_PX, y_px)
                r_slice = content_pre[0, x_s_px: x_e_px, y_s_px: y_e_px, 0]
                g_slice = content_pre[0, x_s_px: x_e_px, y_s_px: y_e_px, 1]
                b_slice = content_pre[0, x_s_px: x_e_px, y_s_px: y_e_px, 2]
                init_colors[i, j, 0] = np.average(r_slice)
                init_colors[i, j, 1] = np.average(g_slice)
                init_colors[i, j, 2] = np.average(b_slice)


        indices = np.zeros((x_px, y_px), dtype=np.int32)
        colors = []
        t_c = 0

        def add_new_color(t_c, color):
            colors.append(color)
            return t_c + 1

        for i in range(x_px):
            for j in range(y_px):
                i_g = i // GRID_PX
                j_g = j // GRID_PX
                i_shift = i % GRID_PX
                j_shift = j % GRID_PX
                # get initial color
                color = init_colors[i_g, j_g, :].reshape((3))
                if i_shift == 0 and j_shift == 0:
                    # create color variable
                    indices[i][j] = t_c
                    t_c = add_new_color(t_c, color)
                else:
                    # copy color
                    indices[i][j] = indices[i_g * GRID_PX][j_g * GRID_PX]
                # backstitch
                # if i_shift == 0 and j_shift == 0:
                #     # create color variable
                #     indices[i][j] = t_c
                #     t_c = add_new_color(t_c, color)
                # elif i_shift == 0:
                #     if j_shift == 1:
                #         indices[i][j] = t_c
                #         t_c = add_new_color(t_c, color)
                #     else:
                #         # copy color
                #         indices[i][j] = indices[i][j_g * GRID_PX + 1]
                # elif j_shift == 0:
                #     if i_shift == 1:
                #         indices[i][j] = t_c
                #         t_c = add_new_color(t_c, color)
                #     else:
                #         # copy color
                #         indices[i][j] = indices[i_g * GRID_PX + 1][j]
                # else:
                #     if i_shift == 1 and j_shift == 1:
                #         indices[i][j] = t_c
                #         t_c = add_new_color(t_c, color)
                #     else:
                #         # copy color
                #         indices[i][j] = indices[i_g * GRID_PX + 1][j_g * GRID_PX + 1]

        colors_init = np.stack(colors)
        colors_v = tf.Variable(colors_init)
        flat_indices = tf.constant(indices.reshape((-1,1)))
        flat_image = tf.gather_nd(colors_v, flat_indices)
        image = tf.reshape(flat_image, shape=shape)

        # x_g_px = x_px // GRID_PX + (0 if x_px % GRID_PX == 0 else 1)
        # y_g_px = y_px // GRID_PX + (0 if y_px % GRID_PX == 0 else 1)
        #
        # init_r = np.zeros((x_g_px, y_g_px), dtype=np.float32)
        # init_g = np.zeros((x_g_px, y_g_px), dtype=np.float32)
        # init_b = np.zeros((x_g_px, y_g_px), dtype=np.float32)
        # for i in range(x_g_px):
        #     for j in range(y_g_px):
        #         x_s_px = i * GRID_PX
        #         x_e_px = min((i + 1) * GRID_PX, x_px)
        #         y_s_px = j * GRID_PX
        #         y_e_px = min((j + 1) * GRID_PX, y_px)
        #         r_slice = content_pre[0, x_s_px: x_e_px, y_s_px: y_e_px, 0]
        #         g_slice = content_pre[0, x_s_px: x_e_px, y_s_px: y_e_px, 1]
        #         b_slice = content_pre[0, x_s_px: x_e_px, y_s_px: y_e_px, 2]
        #         init_r[i, j] = np.average(r_slice)
        #         init_g[i, j] = np.average(g_slice)
        #         init_b[i, j] = np.average(b_slice)
        # image_r = tf.Variable(init_r)
        # image_g = tf.Variable(init_g)
        # image_b = tf.Variable(init_b)
        #
        # # image_r = tf.Variable(tf.random_normal((x_g_px, y_g_px)) * 0.256)
        # # image_g = tf.Variable(tf.random_normal((x_g_px, y_g_px)) * 0.256)
        # # image_b = tf.Variable(tf.random_normal((x_g_px, y_g_px)) * 0.256)
        #
        # h_mat = np.zeros((y_g_px, y_px))
        # for i in range(y_px):
        #     j = i // GRID_PX
        #     h_mat[j, i] = 1
        # tf_h_mat = tf.constant(h_mat, dtype=tf.float32)
        #
        # image_r = tf.matmul(image_r, tf_h_mat)
        # image_g = tf.matmul(image_g, tf_h_mat)
        # image_b = tf.matmul(image_b, tf_h_mat)
        #
        # v_mat = np.zeros((x_px, x_g_px))
        # for i in range(x_px):
        #     j = i // GRID_PX
        #     v_mat[i, j] = 1
        # tf_v_mat = tf.constant(v_mat, dtype=tf.float32)
        #
        # image_r = tf.matmul(tf_v_mat, image_r)
        # image_g = tf.matmul(tf_v_mat, image_g)
        # image_b = tf.matmul(tf_v_mat, image_b)
        #
        # image = tf.stack([image_r, image_g, image_b], axis=2)
        # image = tf.reshape(image, shape=shape)

        net, _ = vgg.net(VGG_PATH, image)

        # color loss
        final_image = image + tf.constant(mean_pixel.astype(np.float32).reshape((1, 1, 3)))
        flat_final_image = tf.reshape(final_image, (-1, 3))
        rgb_to_yuv = tf.constant(RGB_TO_YUV)

        flat_final_image_yuv = tf.transpose(tf.matmul(rgb_to_yuv, tf.transpose(flat_final_image)))
        flat_content_image = tf.reshape(tf.constant(content, dtype=tf.float32), (-1, 3))
        flat_content_yuv = tf.transpose(tf.matmul(rgb_to_yuv, tf.transpose(flat_content_image)))
        i_uv = tf.slice(flat_final_image_yuv, [0, 1], [-1, 2])
        c_uv = tf.slice(flat_content_yuv, [0, 1], [-1, 2])
        c_diff = tf.subtract(i_uv, c_uv)
        c_loss = tf.nn.l2_loss(c_diff) / x_px / y_px

        # content loss
        content_loss = tf.nn.l2_loss(net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_features[
            CONTENT_LAYER].size

        # overall loss
        loss = content_loss + COLOR_WEIGHT * c_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        # optimization
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            # sess.run(tf.initialize_all_variables())
            for i in range(ITERATIONS):
                loss_content, loss_color = sess.run([content_loss, c_loss])
                print('iteration {} content loss: {} color loss: {}'.format(i, loss_content, loss_color))
                sess.run(train_step)

                if i % ITER_PER_CHECKPOINT == 0:
                    result = image.eval()
                    result = result.reshape(shape[1:])
                    result = vgg.unprocess(result, mean_pixel)
                    imsave('./output/result_{}.png'.format(i), result)


def colorize_pattern(content, pattern):
    shape = (1,) + content.shape
    x_px = shape[1]
    y_px = shape[2]

    flat_content_image = content.reshape((-1, 3))
    flat_pattern_image = pattern.reshape((-1, 3))
    flat_content_image_yuv = np.transpose(np.matmul(RGB_TO_YUV, np.transpose(flat_content_image)))
    flat_pattern_image_yuv = np.transpose(np.matmul(RGB_TO_YUV, np.transpose(flat_pattern_image)))
    flat_pattern_y = flat_pattern_image_yuv[:, 0]
    flat_content_u = flat_content_image_yuv[:, 1]
    flat_content_v = flat_content_image_yuv[:, 2]
    pattern_y = flat_pattern_y.reshape((x_px, y_px))
    content_u = flat_content_u.reshape((x_px, y_px))
    content_v = flat_content_v.reshape((x_px, y_px))

    with tf.Graph().as_default():
        x_g_px = x_px // GRID_PX + (0 if x_px % GRID_PX == 0 else 1)
        y_g_px = y_px // GRID_PX + (0 if y_px % GRID_PX == 0 else 1)

        init_y = np.zeros((x_g_px, y_g_px), dtype=np.float32)
        init_u = np.zeros((x_g_px, y_g_px), dtype=np.float32)
        init_v = np.zeros((x_g_px, y_g_px), dtype=np.float32)
        for i in range(x_g_px):
            for j in range(y_g_px):
                x_s_px = i * GRID_PX
                x_e_px = min((i + 1) * GRID_PX, x_px)
                y_s_px = j * GRID_PX
                y_e_px = min((j + 1) * GRID_PX, y_px)

                y_slice = pattern_y[x_s_px: x_e_px, y_s_px: y_e_px]
                u_slice = content_u[x_s_px: x_e_px, y_s_px: y_e_px]
                v_slice = content_v[x_s_px: x_e_px, y_s_px: y_e_px]

                init_y[i, j] = np.average(y_slice)
                init_u[i, j] = np.average(u_slice)
                init_v[i, j] = np.average(v_slice)
        image_y = tf.constant(init_y)
        image_u = tf.Variable(init_u)
        image_v = tf.Variable(init_v)

        h_mat = np.zeros((y_g_px, y_px))
        for i in range(y_px):
            j = i // GRID_PX
            h_mat[j, i] = 1
        tf_h_mat = tf.constant(h_mat, dtype=tf.float32)

        image_y = tf.matmul(image_y, tf_h_mat)
        image_u = tf.matmul(image_u, tf_h_mat)
        image_v = tf.matmul(image_v, tf_h_mat)

        v_mat = np.zeros((x_px, x_g_px))
        for i in range(x_px):
            j = i // GRID_PX
            v_mat[i, j] = 1
        tf_v_mat = tf.constant(v_mat, dtype=tf.float32)

        image_y = tf.matmul(tf_v_mat, image_y)
        image_u = tf.matmul(tf_v_mat, image_u)
        image_v = tf.matmul(tf_v_mat, image_v)

        image = tf.stack([image_y, image_u, image_v], axis=2)
        image = tf.reshape(image, shape=shape)

        # color loss
        flat_image_yuv = tf.reshape(image, (-1, 3))
        i_uv = tf.slice(flat_image_yuv, [0, 1], [-1, 2])
        c_uv = tf.constant(flat_content_image_yuv[:, 1:].astype(np.float32))
        c_diff = tf.subtract(i_uv, c_uv)
        c_loss = tf.nn.l2_loss(c_diff) / x_px / y_px
        loss = c_loss

        yuv_to_rgb = tf.constant(YUV_TO_RGB)
        flat_final_image_rgb = tf.transpose(tf.matmul(yuv_to_rgb, tf.transpose(flat_image_yuv)))
        final_image_rgb = tf.reshape(flat_final_image_rgb, (x_px, y_px, 3))

        # optimizer setup
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        # optimization
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in range(ITERATIONS):
                loss_color = sess.run([c_loss])
                print('iteration {} color loss: {}'.format(i, loss_color))
                sess.run(train_step)

                if i % ITER_PER_CHECKPOINT == 0:
                    result = final_image_rgb.eval()
                    result = result.reshape(shape[1:])
                    imsave('./test/result_{}.png'.format(i), result)


def main():
    content = imread('./input/input.jpg')
    generate_pattern(content)

    # content = imread('./input/input.jpg')
    # pattern = imread('./output/result_50.png')
    # colorize_pattern(content, pattern)


if __name__ == '__main__':
    main()
