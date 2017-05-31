import numpy as np
import tensorflow as tf
import cv2

import vgg

VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
CONTENT_LAYER = 'relu4_2'
GRID_PX = 5
LEARNING_RATE = 1e1
ITERATIONS = 1000
ITER_PER_CHECKPOINT = 10
# CVT_TYPE = cv2.COLOR_BGR2YUV
# INV_CVT_TYPE = cv2.COLOR_YUV2BGR
CVT_TYPE = cv2.COLOR_BGR2LAB
INV_CVT_TYPE = cv2.COLOR_LAB2BGR
# CVT_TYPE = cv2.COLOR_BGR2LUV
# INV_CVT_TYPE = cv2.COLOR_LUV2BGR
# CVT_TYPE = cv2.COLOR_BGR2YCR_CB
# INV_CVT_TYPE = cv2.COLOR_YCR_CB2BGR



RGB_TO_YUV = np.array([[0.299, 0.587, 0.114],
                       [-0.14713, -0.28886, 0.436],
                       [0.615, -0.51499, -0.10001]], dtype=np.float32)

YUV_TO_RGB = np.array([[1., 0., 1.13983],
                       [1., -0.39465, -0.58060],
                       [1., 2.03211, 0.]], dtype=np.float32)


def imread(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img.astype(np.float32)
    #bgr to rgb
    img = img[..., ::-1]

    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    # rgb to bgr
    img = img[..., ::-1]
    cv2.imwrite(path, img)


def generate_pattern(content):
    content_bgr = content[..., ::-1] / 255.
    content_cvt = cv2.cvtColor(content_bgr, CVT_TYPE)

    content_features = {}

    shape = (1,) + content.shape

    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net, mean_pixel = vgg.net(VGG_PATH, image)

        content_features[CONTENT_LAYER] = net[CONTENT_LAYER].eval(
            feed_dict={image: np.array([vgg.preprocess(content, mean_pixel)])})

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

                xs_cvt = content_cvt[x_s_px: x_e_px, y_s_px: y_e_px, :]
                xs_init_cvt_color = np.mean(xs_cvt, axis=(0,1))
                xs_init_bgr_color = cv2.cvtColor(xs_init_cvt_color[np.newaxis, np.newaxis, :], INV_CVT_TYPE)
                xs_init_rgb_color = xs_init_bgr_color[..., ::-1] * 255.
                xs_init = vgg.preprocess(xs_init_rgb_color, mean_pixel)
                init_colors[i, j] = xs_init[0,0]

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

        colors_init = np.stack(colors)
        colors_v = tf.Variable(colors_init)
        flat_indices = tf.constant(indices.reshape((-1,1)))
        flat_image = tf.gather_nd(colors_v, flat_indices)
        image = tf.reshape(flat_image, shape=shape)

        net, _ = vgg.net(VGG_PATH, image)

        # content loss
        content_loss = tf.nn.l2_loss(net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_features[
            CONTENT_LAYER].size

        # overall loss
        loss = content_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        # optimization
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # sess.run(tf.initialize_all_variables())
            init_img = sess.run(image).reshape(content.shape)
            init_img = vgg.unprocess(init_img, mean_pixel)
            imsave('./output/init.png'.format(i), init_img)
            init_img_bgr = init_img[..., ::-1].astype(np.float32) / 255.
            init_img_cvt = cv2.cvtColor(init_img_bgr, CVT_TYPE)
            _, c2, c3 = cv2.split(init_img_cvt)

            for i in range(ITERATIONS):
                loss_content = sess.run([content_loss])
                print('iteration {} perceptual loss: {}'.format(i, loss_content))
                sess.run(train_step)

                if i % ITER_PER_CHECKPOINT == 0:
                    result = image.eval()
                    result = result.reshape(shape[1:])
                    result = vgg.unprocess(result, mean_pixel)
                    imsave('./output/result_{}.png'.format(i), result)

                    result_bgr = result[..., ::-1].astype(np.float32) / 255.

                    # rgb to bgr
                    result_cvt = cv2.cvtColor(result_bgr, CVT_TYPE)
                    c1, _, _ = cv2.split(result_cvt)
                    merged = cv2.merge((c1, c2, c3))
                    result = cv2.cvtColor(merged, INV_CVT_TYPE) * 255.
                    # bgr to rgb
                    result = result[..., ::-1]
                    imsave('./output/result_{}_color.png'.format(i), result)

def main():
    content = imread('./input/input.jpg')
    generate_pattern(content)

if __name__ == '__main__':
    main()
